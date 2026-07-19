# Import OpenCV for webcam capture and display.
import sys
import time
import urllib.request
from pathlib import Path

import cv2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image_module
from mediapipe.tasks.python.vision.core import vision_task_running_mode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


def ensure_model() -> None:
    """Download the hand landmarker model on first run if it is missing."""
    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading hand landmarker model (one-time setup)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def create_hand_landmarker() -> hand_landmarker.HandLandmarker:
    """Create a MediaPipe hand landmarker configured for webcam video."""
    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hand_landmarker.HandLandmarker.create_from_options(options)


def draw_hands(
    frame,
    result: hand_landmarker.HandLandmarkerResult,
) -> None:
    """Draw hand skeletons and highlight index fingertips for sparkle anchors."""
    height, width = frame.shape[:2]

    for hand_landmarks in result.hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

        # Mark the index fingertip — this is where sparkle particles will spawn next.
        tip = hand_landmarks[hand_landmarker.HandLandmark.INDEX_FINGER_TIP]
        tip_x = int(tip.x * width)
        tip_y = int(tip.y * height)
        cv2.circle(frame, (tip_x, tip_y), 8, (0, 255, 255), -1)


def main():
    ensure_model()

    # Open the default webcam (device index 0).
    cap = cv2.VideoCapture(0)

    # Verify the webcam opened successfully before continuing.
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("  - On macOS: System Settings → Privacy & Security → Camera")
        print("    Enable access for the app running this script (Terminal, iTerm, or Cursor).")
        print("    Quit and reopen that app, then run this script again.")
        print("  - Also check that no other app is using the camera.")
        sys.exit(1)

    window_name = "Sparkly"
    landmarker = create_hand_landmarker()
    start_time_ms = int(time.time() * 1000)

    # Continuously read frames from the webcam until the user quits.
    try:
        while True:
            # Read one frame; ret is True when a frame was captured successfully.
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            # Flip horizontally so the feed behaves like a mirror.
            mirrored_frame = cv2.flip(frame, 1)

            # MediaPipe expects RGB; OpenCV gives us BGR.
            rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_image_module.Image(
                image_format=mp_image_module.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # Detect hands on the mirrored frame so overlays line up with the display.
            frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                draw_hands(mirrored_frame, result)

            # Show the mirrored frame with hand overlays in a window.
            cv2.imshow(window_name, mirrored_frame)

            # Wait 1 ms for a key press; exit cleanly when 'q' is pressed.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()

    # Release the webcam and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
