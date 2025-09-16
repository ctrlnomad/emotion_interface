import sys
import time
import cv2
import mediapipe as mp


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

                if show_landmarks and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
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
