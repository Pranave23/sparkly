import math
import random
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils, hand_landmarker
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode

# --- CONFIGURATION ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("hand_landmarker.task")

# Glitter Palette (Gold, Silver, Rose Gold, Magenta in BGR format)
GLITTER_COLORS = [
    (0, 215, 255),    # Gold
    (30, 195, 255),   # Warm Gold
    (180, 105, 255),  # Rose Gold / Pink
    (245, 245, 245),  # Bright Silver
    (147, 20, 255),   # Deep Magenta
]

class Particle:
    def __init__(self, x, y):
        self.x = x + random.uniform(-15, 15)
        self.y = y + random.uniform(-15, 15)
        self.color = random.choice(GLITTER_COLORS)
        
        # 30% chance for a particle to be a sharp 4-point glint
        self.is_glint = random.random() < 0.30

        self.vx = random.uniform(-4.0, 4.0)
        self.vy = random.uniform(-5.0, 1.0)
        
        self.max_life = random.uniform(20, 45)
        self.life = self.max_life
        self.radius = random.randint(2, 5)
        
        self.twinkle_speed = random.uniform(0.3, 0.8)
        self.twinkle_offset = random.uniform(0, math.pi * 2)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Gravity and Air resistance
        self.vy += 0.2 
        self.vx *= 0.95 

        self.life -= 1

    def draw(self, overlay):
        base_fade = max(0, self.life / self.max_life)
        twinkle_factor = 0.5 + 0.5 * math.sin(self.life * self.twinkle_speed + self.twinkle_offset)
        fade = base_fade * twinkle_factor
        
        # If the particle is essentially invisible, skip drawing it 
        if fade < 0.05: return
        
        r = max(1, int(self.radius * fade))
        c = tuple(min(255, int(ch * fade * 1.5)) for ch in self.color)
        center = (int(self.x), int(self.y))

        # We draw onto the black "overlay", NOT the camera frame directly
        if self.is_glint and r > 1:
            glint_len = r + 3
            cv2.line(overlay, (center[0] - glint_len, center[1]), (center[0] + glint_len, center[1]), c, 1, cv2.LINE_AA)
            cv2.line(overlay, (center[0], center[1] - glint_len), (center[0], center[1] + glint_len), c, 1, cv2.LINE_AA)
            cv2.circle(overlay, center, 1, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, center, r, c, -1, cv2.LINE_AA)

def main():
    if not MODEL_PATH.exists():
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2
    )
    landmarker = hand_landmarker.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    particles = []
    start_ms = int(time.time() * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = landmarker.detect_for_video(
            mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame),
            int(time.time() * 1000) - start_ms
        )

        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                drawing_utils.draw_landmarks(frame, landmarks, hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS)

                for tip_idx in [4, 8, 12, 16, 20]:
                    tx, ty = int(landmarks[tip_idx].x * w), int(landmarks[tip_idx].y * h)
                    for _ in range(30):
                        particles.append(Particle(tx, ty))

        particles = [p for p in particles if p.life > 0][-6000:] 
        
        # --- THE FIX ---
        # 1. Create a pitch-black canvas the exact size of the webcam feed
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # 2. Draw all the glowing particles onto the black canvas
        for p in particles:
            p.update()
            p.draw(overlay)

        # 3. Additive Blending: Add the lights from the canvas to the camera frame!
        # This completely ignores black pixels and makes the colors glow brightly.
        cv2.add(frame, overlay, dst=frame)
        # ---------------

        cv2.imshow("Glowing Fingertip Magic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()