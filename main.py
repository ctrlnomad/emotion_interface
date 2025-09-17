import logging
import os
import sys
import time
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from config import (
    DEFAULT_LLM_CONFIG,
    DEFAULT_THOUGHT_CONFIG,
    style_for_emotion,
)
from emotion_classifier import RuleBasedEmotionClassifier
from llm_client import LLMClient
from overlay import draw_thought_overlay
from thought_engine import ThoughtEngine


WINDOW_TITLE = "Face Landmarks"
FONT = cv2.FONT_HERSHEY_SIMPLEX
class FaceMeshDetector:
    """Wrapper around MediaPipe FaceMesh with sensible defaults."""

    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        # MediaPipe expects RGB input images.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._face_mesh.process(frame_rgb)

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class FPSCounter:
    def __init__(self, interval: float = 5.0) -> None:
        self.interval = interval
        self._last_time = time.time()
        self._frames = 0
        self.value = 0.0

    def update(self) -> float:
        self._frames += 1
        now = time.time()
        elapsed = now - self._last_time
        if elapsed >= self.interval:
            self.value = self._frames / elapsed
            self._frames = 0
            self._last_time = now
        return self.value


def _select_uniform_landmark_indices(
    face_landmarks, keep_ratio: float = 0.4
) -> List[int]:
    total = len(face_landmarks.landmark)
    if total == 0:
        return []

    target = max(1, int(total * keep_ratio))
    coords = [
        (landmark.x, landmark.y, landmark.z) for landmark in face_landmarks.landmark
    ]

    seed_index = 4 if total > 4 else 0
    selected = [seed_index]
    remaining = set(range(total))
    remaining.discard(seed_index)

    def distance_sq(a, b) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return dx * dx + dy * dy + dz * dz

    while len(selected) < target and remaining:
        best_idx = None
        best_dist = -1.0
        for idx in remaining:
            coord = coords[idx]
            nearest = min(distance_sq(coord, coords[s_idx]) for s_idx in selected)
            if nearest > best_dist:
                best_dist = nearest
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    selected.sort()
    return selected
def draw_landmarks(frame, face_landmarks, indices, color=(255, 255, 255)) -> None:
    height, width, _ = frame.shape
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 3, color, -1, lineType=cv2.LINE_AA)


def main() -> int:
    if not logging.getLogger().handlers:
        log_level = os.getenv("EMOTION_INTERFACE_LOG_LEVEL", "INFO").upper()
        try:
            resolved_level = getattr(logging, log_level, logging.INFO)
        except AttributeError:
            resolved_level = logging.INFO
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
        )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Ensure a camera is connected and accessible.", file=sys.stderr)
        return 1

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()
    classifier = RuleBasedEmotionClassifier()
    thought_engine: Optional[ThoughtEngine] = None
    llm_client = LLMClient(DEFAULT_LLM_CONFIG)
    thought_engine = ThoughtEngine(llm_client, DEFAULT_THOUGHT_CONFIG)

    if llm_client.init_error:
        print(llm_client.init_error, file=sys.stderr)
    show_landmarks = True
    use_black_background = False
    thoughts_enabled = True

    selected_landmark_indices: Optional[List[int]] = None

    try:
        with FaceMeshDetector() as detector:
            print(
                "Controls: press 'q' to quit, 'l' to toggle landmarks display, "
                "'b' to toggle background, 't' to toggle thoughts."
            )
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from webcam.", file=sys.stderr)
                    break

                results = detector.process(frame)
                annotated = (
                    np.zeros_like(frame)
                    if use_black_background
                    else frame.copy()
                )
                emotion_text = "Emotion: --"
                rule_text = "Rule: --"
                feature_lines = []
                current_emotion = None
                face_landmarks = None

                anchor_indices: Optional[List[int]] = selected_landmark_indices
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    if selected_landmark_indices is None:
                        selected_landmark_indices = _select_uniform_landmark_indices(
                            face_landmarks
                        )
                    emotion_result = classifier.classify(face_landmarks)
                    current_emotion = emotion_result.label
                    emotion_text = (
                        f"Emotion: {emotion_result.label}"
                        f" ({emotion_result.confidence:.2f})"
                    )
                    rule_text = f"Rule: {emotion_result.rule}"
                    feature_lines = [
                        f"{name}: {value:.3f}"
                        for name, value in emotion_result.features.items()
                    ]

                    if (
                        show_landmarks
                        and face_landmarks is not None
                        and selected_landmark_indices is not None
                    ):
                        draw_landmarks(
                            annotated,
                            face_landmarks,
                            selected_landmark_indices,
                        )
                        anchor_indices = selected_landmark_indices

                thought_renders = thought_engine.update(
                    current_emotion,
                    face_landmarks,
                    annotated.shape,
                    anchor_indices,
                )
                for thought_render in thought_renders:
                    style = style_for_emotion(thought_render.emotion)
                    draw_thought_overlay(annotated, thought_render, style)

                fps = fps_counter.update()
                status_text = (
                    f"Landmarks: {'on' if show_landmarks else 'off'}"
                    f" | Bg: {'black' if use_black_background else 'camera'}"
                    f" | Thoughts: {'on' if thoughts_enabled else 'off'}"
                )
                cv2.putText(
                    annotated,
                    status_text,
                    (10, 25),
                    FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 55),
                    FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    emotion_text,
                    (10, 85),
                    FONT,
                    0.7,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    rule_text,
                    (10, 115),
                    FONT,
                    0.7,
                    (200, 200, 200),
                    2,
                    lineType=cv2.LINE_AA,
                )

                feature_base_y = 145
                for idx, feat_text in enumerate(feature_lines):
                    cv2.putText(
                        annotated,
                        feat_text,
                        (10, feature_base_y + idx * 25),
                        FONT,
                        0.6,
                        (180, 220, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_TITLE, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("l"):
                    show_landmarks = not show_landmarks
                if key == ord("b"):
                    use_black_background = not use_black_background
                if key == ord("t"):
                    thoughts_enabled = not thoughts_enabled
                    thought_engine.set_streaming_enabled(thoughts_enabled)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if thought_engine is not None:
            thought_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
