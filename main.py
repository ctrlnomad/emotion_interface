import sys
import time
import cv2
import mediapipe as mp

from emotion_classifier import RuleBasedEmotionClassifier


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


def draw_landmarks(frame, face_landmarks, color=(0, 255, 0)) -> None:
    height, width, _ = frame.shape
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 1, color, -1, lineType=cv2.LINE_AA)


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Ensure a camera is connected and accessible.", file=sys.stderr)
        return 1

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()
    classifier = RuleBasedEmotionClassifier()
    show_landmarks = True

    try:
        with FaceMeshDetector() as detector:
            print("Controls: press 'q' to quit, 'l' to toggle landmarks display.")
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from webcam.", file=sys.stderr)
                    break

                results = detector.process(frame)
                annotated = frame.copy()
                emotion_text = "Emotion: --"
                rule_text = "Rule: --"
                feature_lines = []

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    emotion_result = classifier.classify(face_landmarks)
                    emotion_text = (
                        f"Emotion: {emotion_result.label}"
                        f" ({emotion_result.confidence:.2f})"
                    )
                    rule_text = f"Rule: {emotion_result.rule}"
                    feature_lines = [
                        f"{name}: {value:.3f}"
                        for name, value in emotion_result.features.items()
                    ]

                    if show_landmarks:
                        draw_landmarks(annotated, face_landmarks)

                fps = fps_counter.update()
                status_text = f"Landmarks: {'on' if show_landmarks else 'off'}"
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
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
