# Import OpenCV for webcam capture and display.
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
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

# BGR sparkle palette — shiny pink, gold, and silver only (a couple of shades
# of each so the sparkles feel varied but stay on-theme).
SPARKLE_COLORS = (
    (180, 105, 255),  # hot pink
    (203, 192, 255),  # soft pink
    (147, 20, 255),   # deep pink / magenta glow
    (0, 215, 255),    # gold
    (30, 195, 255),   # warm gold
    (0, 165, 255),    # deep amber gold
    (230, 230, 230),  # bright silver
    (192, 192, 192),  # classic silver
    (245, 245, 245),  # near-white silver glint
)


@dataclass
class Sparkle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    radius: int
    twinkle_offset: float


class SparkleSystem:
    """Simple particle pool that emits and fades sparkles at fingertip positions."""

    def __init__(self, max_particles: int = 900):
        self.particles: list[Sparkle] = []
        self.max_particles = max_particles

    def spawn_at(self, x: int, y: int, count: int = 9) -> None:
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break

            max_life = random.uniform(18, 36)
            self.particles.append(
                Sparkle(
                    x=x + random.uniform(-8, 8),
                    y=y + random.uniform(-8, 8),
                    vx=random.uniform(-2.2, 2.2),
                    vy=random.uniform(-3.2, -0.4),
                    life=max_life,
                    max_life=max_life,
                    color=random.choice(SPARKLE_COLORS),
                    radius=random.randint(2, 6),
                    twinkle_offset=random.uniform(0, 6.28),
                )
            )

    def update(self) -> None:
        alive: list[Sparkle] = []
        for particle in self.particles:
            particle.x += particle.vx
            particle.y += particle.vy
            particle.vy += 0.04
            particle.life -= 1
            if particle.life > 0:
                alive.append(particle)
        self.particles = alive

    def draw(self, frame) -> None:
        for particle in self.particles:
            fade = particle.life / particle.max_life

            # Twinkle: modulate brightness with a fast sine wave so the
            # sparkles shimmer instead of fading smoothly and flatly.
            twinkle = 0.65 + 0.35 * abs(
                (particle.life * 0.9 + particle.twinkle_offset) % 2 - 1
            )
            brightness = fade * twinkle

            radius = max(1, int(particle.radius * fade))
            color = tuple(min(255, int(channel * brightness * 1.15)) for channel in particle.color)
            center = (int(particle.x), int(particle.y))

            # Soft outer glow.
            if radius > 2:
                glow_color = tuple(int(c * 0.5) for c in color)
                cv2.circle(frame, center, radius + 2, glow_color, -1, lineType=cv2.LINE_AA)

            cv2.circle(frame, center, radius, color, -1, lineType=cv2.LINE_AA)

            # Bright core + tiny four-point star glint for the shiny look.
            if radius > 2:
                cv2.circle(frame, center, 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                glint_len = radius + 2
                cv2.line(
                    frame,
                    (center[0] - glint_len, center[1]),
                    (center[0] + glint_len, center[1]),
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )
                cv2.line(
                    frame,
                    (center[0], center[1] - glint_len),
                    (center[0], center[1] + glint_len),
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )


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
) -> list[tuple[int, int]]:
    """Draw hand skeletons and return index fingertip pixel positions."""
    height, width = frame.shape[:2]
    fingertip_positions: list[tuple[int, int]] = []

    for hand_landmarks in result.hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

        tip = hand_landmarks[hand_landmarker.HandLandmark.INDEX_FINGER_TIP]
        fingertip_positions.append((int(tip.x * width), int(tip.y * height)))

    return fingertip_positions


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
    sparkles = SparkleSystem()
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
                fingertip_positions = draw_hands(mirrored_frame, result)
                for tip_x, tip_y in fingertip_positions:
                    sparkles.spawn_at(tip_x, tip_y)

            sparkles.update()
            sparkles.draw(mirrored_frame)

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