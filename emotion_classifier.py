from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Tuple

from mediapipe.framework.formats import landmark_pb2


# MediaPipe FaceMesh landmark indices used for feature extraction.
LANDMARKS = {
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_brow_inner": 70,
    "left_brow_outer": 105,
    "right_brow_inner": 336,
    "right_brow_outer": 334,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "upper_lip": 13,
    "lower_lip": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "face_top": 10,
    "face_bottom": 152,
}


@dataclass
class EmotionResult:
    label: str
    raw_label: str
    confidence: float
    features: Dict[str, float]
    rule: str


class RuleBasedEmotionClassifier:
    """Heuristic facial emotion classifier using MediaPipe landmarks."""

    def __init__(self, history: int = 12) -> None:
        self._history: Deque[str] = deque(maxlen=history)

    def classify(self, face_landmarks: landmark_pb2.NormalizedLandmarkList) -> EmotionResult:
        coords = {name: self._get_landmark(face_landmarks, idx) for name, idx in LANDMARKS.items()}
        features = self._compute_features(coords)
        raw_label, confidence, rule = self._apply_rules(features)

        self._history.append(raw_label)
        smoothed = self._smooth_label()

        return EmotionResult(
            label=smoothed,
            raw_label=raw_label,
            confidence=confidence,
            features=features,
            rule=rule,
        )

    def _smooth_label(self) -> str:
        if not self._history:
            return "neutral"
        counts = Counter(self._history)
        label, _ = counts.most_common(1)[0]
        return label

    @staticmethod
    def _get_landmark(
        face_landmarks: landmark_pb2.NormalizedLandmarkList, index: int
    ) -> Tuple[float, float, float]:
        landmark = face_landmarks.landmark[index]
        return landmark.x, landmark.y, landmark.z

    @staticmethod
    def _compute_features(coords: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

        face_height = max(distance(coords["face_top"], coords["face_bottom"]), 1e-6)
        face_width = max(distance(coords["left_eye_outer"], coords["right_eye_outer"]), 1e-6)

        left_eye_open = (coords["left_eye_bottom"][1] - coords["left_eye_top"][1]) / face_height
        right_eye_open = (coords["right_eye_bottom"][1] - coords["right_eye_top"][1]) / face_height
        eye_openness = max((left_eye_open + right_eye_open) / 2.0, 0.0)

        left_eye_center_y = (coords["left_eye_top"][1] + coords["left_eye_bottom"][1]) / 2.0
        right_eye_center_y = (coords["right_eye_top"][1] + coords["right_eye_bottom"][1]) / 2.0
        left_brow_height = (left_eye_center_y - coords["left_brow_outer"][1]) / face_height
        right_brow_height = (right_eye_center_y - coords["right_brow_outer"][1]) / face_height
        brow_height_avg = (left_brow_height + right_brow_height) / 2.0
        brow_asymmetry = abs(left_brow_height - right_brow_height)

        mouth_width = distance(coords["mouth_left"], coords["mouth_right"]) / face_width
        mouth_open = (coords["lower_lip"][1] - coords["upper_lip"][1]) / face_height
        mouth_midpoint_y = (coords["upper_lip"][1] + coords["lower_lip"][1]) / 2.0
        mouth_corner_avg_y = (coords["mouth_left"][1] + coords["mouth_right"][1]) / 2.0
        mouth_smile = (mouth_midpoint_y - mouth_corner_avg_y) / face_height

        return {
            "eye_openness": eye_openness,
            "brow_height_avg": brow_height_avg,
            "brow_asymmetry": brow_asymmetry,
            "mouth_open": mouth_open,
            "mouth_smile": mouth_smile,
            "mouth_width": mouth_width,
        }

    @staticmethod
    def _apply_rules(features: Dict[str, float]) -> Tuple[str, float, str]:
        mouth_smile = features["mouth_smile"]
        mouth_open = features["mouth_open"]
        brow_height = features["brow_height_avg"]
        brow_asym = features["brow_asymmetry"]
        eye_open = features["eye_openness"]

        # Happy: smiling mouth with moderate openness and relaxed brows.
        if mouth_smile > 0.025 and 0.015 < mouth_open < 0.09 and brow_height > 0.02:
            confidence = min(1.0, (mouth_smile - 0.02) * 12 + 0.5)
            return "happy", confidence, "smile"

        # Distressed: wide open mouth or notable downward corners with tense brows and wide eyes.
        if mouth_open >= 0.09 or (mouth_smile < -0.015 and mouth_open > 0.03):
            score = max(mouth_open * 5, (-mouth_smile) * 20)
            if brow_height < 0.02 or eye_open > 0.045:
                score += 0.2
            confidence = min(1.0, 0.4 + score)
            return "distressed", confidence, "mouth_open"

        # Confused: noticeable brow asymmetry or single raised brow with neutral mouth.
        if brow_asym > 0.012 and mouth_open < 0.06:
            confidence = min(1.0, 0.5 + (brow_asym - 0.01) * 40)
            return "confused", confidence, "brow_asym"

        # Neutral fallback helps stabilise smoothing history.
        return "neutral", 0.2, "neutral"
