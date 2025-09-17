"""Orchestrates thought lifecycle, scheduling, and streaming updates."""

from __future__ import annotations

import itertools
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mediapipe.framework.formats import landmark_pb2

from config import DEFAULT_THOUGHT_CONFIG, ThoughtConfig
from llm_client import LLMClient


logger = logging.getLogger(__name__)


ThoughtStatus = str


@dataclass
class ThoughtRequest:
    request_id: int
    emotion: str
    mode: str = "thought"


@dataclass
class ThoughtEvent:
    kind: str
    request_id: int
    payload: Optional[str] = None


@dataclass
class ThoughtState:
    request_id: int
    emotion: str
    mode: str = "thought"
    text: str = ""
    status: ThoughtStatus = "streaming"
    started_at: float = field(default_factory=time.monotonic)
    latched_at: Optional[float] = None
    box_origin: Tuple[int, int] = (0, 0)
    box_size: Tuple[int, int] = (0, 0)
    landmark_index: int = 0
    last_landmark_point: Tuple[int, int] = (0, 0)
    error_message: Optional[str] = None


@dataclass
class ThoughtRenderData:
    request_id: int
    text: str
    status: ThoughtStatus
    box_origin: Tuple[int, int]
    box_size: Tuple[int, int]
    line_start: Tuple[int, int]
    emotion: str
    hold_progress: float
    mode: str = "thought"


class ThoughtEngine:
    """Coordinates thought generation, random scheduling, and display state."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: ThoughtConfig = DEFAULT_THOUGHT_CONFIG,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._client = llm_client
        self._config = config
        self._rng = rng or random.Random()
        self._id_iter = itertools.count(1)
        self._request_queue: "queue.Queue[Optional[ThoughtRequest]]" = queue.Queue()
        self._event_queue: "queue.Queue[ThoughtEvent]" = queue.Queue()
        self._states: Dict[int, ThoughtState] = {}
        self._next_spawn_time = time.monotonic() + self._random_interval()
        self._worker: Optional[threading.Thread] = threading.Thread(
            target=self._run_worker,
            daemon=True,
        )
        self._worker.start()
        logger.debug(
            "ThoughtEngine worker thread started",
            extra={
                "next_spawn_time": self._next_spawn_time,
                "queue_size": self._request_queue.qsize(),
            },
        )

    def stop(self) -> None:
        if self._worker:
            self._request_queue.put(None)
            self._worker.join(timeout=1.0)
            logger.debug("ThoughtEngine worker stop requested")
            self._worker = None

    def update(
        self,
        emotion: Optional[str],
        face_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
        frame_shape: Tuple[int, int, int],
        available_landmarks: Optional[Sequence[int]] = None,
    ) -> List[ThoughtRenderData]:
        now = time.monotonic()
        self._drain_events(now)
        self._maybe_spawn_thought(
            emotion,
            face_landmarks,
            frame_shape,
            now,
            available_landmarks,
        )
        self._refresh_landmark_points(face_landmarks, frame_shape)
        self._maybe_expire(now)

        renders: List[ThoughtRenderData] = []
        for state in list(self._states.values()):
            hold_progress = 0.0
            if state.status == "latched" and state.latched_at:
                elapsed = now - state.latched_at
                hold_progress = min(elapsed / self._config.hold_duration, 1.0)

            if state.mode == "ascii":
                text = state.text.rstrip("\n") if state.text else ""
            else:
                text = state.text.strip() if state.text else ""
            if not text and state.error_message:
                text = state.error_message

            renders.append(
                ThoughtRenderData(
                    request_id=state.request_id,
                    text=text,
                    status=state.status,
                    box_origin=state.box_origin,
                    box_size=state.box_size,
                    line_start=state.last_landmark_point,
                    emotion=state.emotion,
                    hold_progress=hold_progress,
                    mode=state.mode,
                )
            )

        return renders

    def _maybe_spawn_thought(
        self,
        emotion: Optional[str],
        face_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
        frame_shape: Tuple[int, int, int],
        now: float,
        available_landmarks: Optional[Sequence[int]],
    ) -> None:
        if now < self._next_spawn_time:
            return

        self._schedule_next_spawn(now)

        if (
            not emotion
            or face_landmarks is None
            or len(self._states) >= self._config.max_active_thoughts
        ):
            return

        lowered = emotion.lower()
        mode = "thought"
        if lowered in {"neutral", "happy"} and self._rng.random() < 0.5:
            mode = "ascii"

        box_origin = self._pick_box_origin(face_landmarks, frame_shape)
        landmark_idx = self._pick_landmark_index(face_landmarks, available_landmarks)
        landmark_point = self._landmark_to_point(face_landmarks, landmark_idx, frame_shape)

        state = ThoughtState(
            request_id=next(self._id_iter),
            emotion=emotion,
            mode=mode,
            box_origin=box_origin,
            box_size=self._config.default_box_size,
            landmark_index=landmark_idx,
            last_landmark_point=landmark_point,
        )
        self._states[state.request_id] = state
        self._request_queue.put(ThoughtRequest(state.request_id, emotion, mode))
        logger.debug(
            "Scheduled thought request",
            extra={
                "request_id": state.request_id,
                "emotion": emotion,
                "active_thoughts": len(self._states),
                "queue_size": self._request_queue.qsize(),
                "mode": mode,
            },
        )

    def _refresh_landmark_points(
        self,
        face_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
        frame_shape: Tuple[int, int, int],
    ) -> None:
        if face_landmarks is None:
            return
        for state in self._states.values():
            state.last_landmark_point = self._landmark_to_point(
                face_landmarks,
                state.landmark_index,
                frame_shape,
            )

    def _drain_events(self, now: float) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break

            state = self._states.get(event.request_id)
            if state is None:
                logger.debug(
                    "Dropped event for unknown state",
                    extra={"request_id": event.request_id, "kind": event.kind},
                )
                continue

            if event.kind == "chunk" and event.payload:
                state.text += event.payload
                state.status = "streaming"
                logger.debug(
                    "Processed chunk event",
                    extra={
                        "request_id": event.request_id,
                        "text_length": len(state.text),
                        "mode": state.mode,
                    },
                )
            elif event.kind == "done":
                state.status = "latched"
                state.latched_at = now
                logger.debug(
                    "Thought latched",
                    extra={"request_id": event.request_id, "mode": state.mode},
                )
            elif event.kind == "error":
                state.status = "error"
                state.error_message = event.payload or "Thought generation failed."
                state.latched_at = now
                logger.debug(
                    "Thought errored",
                    extra={
                        "request_id": event.request_id,
                        "error_message": state.error_message,
                        "mode": state.mode,
                    },
                )

    def _maybe_expire(self, now: float) -> None:
        expired: List[int] = []
        for request_id, state in self._states.items():
            if state.status not in {"latched", "error"}:
                continue
            if not state.latched_at:
                continue
            if now - state.latched_at >= self._config.hold_duration:
                expired.append(request_id)
        for request_id in expired:
            self._states.pop(request_id, None)
            logger.debug("Expired thought state", extra={"request_id": request_id})

    def _run_worker(self) -> None:
        while True:
            request = self._request_queue.get()
            if request is None:
                logger.debug("Worker shutting down on sentinel")
                break
            logger.debug(
                "Worker dequeued request",
                extra={
                    "request_id": request.request_id,
                    "emotion": request.emotion,
                    "mode": request.mode,
                },
            )
            try:
                self._client.stream_thought(
                    request.emotion,
                    lambda chunk, req_id=request.request_id: self._event_queue.put(
                        ThoughtEvent("chunk", req_id, chunk)
                    ),
                    mode=request.mode,
                )
                logger.debug(
                    "Worker finished streaming",
                    extra={"request_id": request.request_id, "mode": request.mode},
                )
                self._event_queue.put(ThoughtEvent("done", request.request_id))
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.exception(
                    "Worker encountered error during streaming",
                    extra={"request_id": request.request_id, "mode": request.mode},
                )
                self._event_queue.put(
                    ThoughtEvent("error", request.request_id, str(exc))
                )

    def _pick_landmark_index(
        self,
        face_landmarks: landmark_pb2.NormalizedLandmarkList,
        allowed_indices: Optional[Sequence[int]] = None,
    ) -> int:
        candidates: Iterable[int]
        if allowed_indices:
            candidates = allowed_indices
        else:
            candidates = range(len(face_landmarks.landmark))

        viable_indices: List[int] = []
        for idx in candidates:
            landmark = face_landmarks.landmark[idx]
            if landmark.visibility and landmark.visibility < self._config.min_landmark_visibility:
                continue
            viable_indices.append(idx)

        if not viable_indices:
            viable_indices = list(range(len(face_landmarks.landmark)))

        return self._rng.choice(viable_indices)

    def _pick_box_origin(
        self,
        face_landmarks: landmark_pb2.NormalizedLandmarkList,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        height, width, _ = frame_shape
        margin = self._config.frame_margin
        box_w, box_h = self._config.default_box_size

        face_bounds = self._face_bounds(face_landmarks, frame_shape)

        for _ in range(self._config.max_box_attempts):
            max_x = max(margin, width - box_w - margin)
            max_y = max(margin, height - box_h - margin)
            x = self._rng.randint(margin, max_x)
            y = self._rng.randint(margin, max_y)
            candidate = (x, y, x + box_w, y + box_h)
            if not self._intersects(candidate, face_bounds):
                return (x, y)

        # Fallback to top corner
        return (margin, margin)

    def _face_bounds(
        self,
        face_landmarks: landmark_pb2.NormalizedLandmarkList,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        height, width, _ = frame_shape
        xs: List[int] = []
        ys: List[int] = []
        for landmark in face_landmarks.landmark:
            xs.append(int(landmark.x * width))
            ys.append(int(landmark.y * height))
        min_x = max(0, min(xs))
        max_x = min(width, max(xs))
        min_y = max(0, min(ys))
        max_y = min(height, max(ys))

        pad_x = int((max_x - min_x) * self._config.face_padding_ratio)
        pad_y = int((max_y - min_y) * self._config.face_padding_ratio)

        return (
            max(0, min_x - pad_x),
            max(0, min_y - pad_y),
            min(width, max_x + pad_x),
            min(height, max_y + pad_y),
        )

    @staticmethod
    def _intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    @staticmethod
    def _landmark_to_point(
        face_landmarks: landmark_pb2.NormalizedLandmarkList,
        index: int,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        height, width, _ = frame_shape
        landmark = face_landmarks.landmark[index]
        return int(landmark.x * width), int(landmark.y * height)

    def _random_interval(self) -> float:
        low, high = self._config.spawn_interval_range
        if high <= low:
            return max(0.1, low)
        return self._rng.uniform(low, high)

    def _schedule_next_spawn(self, base_time: float) -> None:
        self._next_spawn_time = base_time + self._random_interval()


__all__ = ["ThoughtEngine", "ThoughtRenderData"]
