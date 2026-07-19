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

# Ice & Glitter Palette (BGR format)
GLITTER_COLORS = [
    (255, 255, 255),  # Pure White
    (255, 245, 220),  # Pale Frost Blue
    (255, 210, 150),  # Bright Cyan
    (250, 200, 100),  # Deep Glacier Blue
    (180, 105, 255),  # Magic Magenta/Pink
]

class Particle:
    def __init__(self, x, y, is_burst=False):
        self.x = x + random.uniform(-15, 15)
        self.y = y + random.uniform(-15, 15)
        self.color = random.choice(GLITTER_COLORS)
        
        # 30% chance for a particle to be a sharp 4-point glint
        self.is_glint = random.random() < 0.30

        if is_burst:
            # THE UPWARD GUSH:
            # OpenCV y-axis goes DOWN, so negative 'vy' shoots straight UP into the air.
            self.vx = random.uniform(-8.0, 8.0)      # Wide horizontal spread
            self.vy = random.uniform(-15.0, -35.0)   # Violent upward velocity
            self.max_life = random.uniform(40, 70)   # Bursts last longer to rain back down
            self.radius = random.randint(3, 7)       # Burst particles are slightly bigger
        else:
            # Standard fingertip flutter
            self.vx = random.uniform(-4.0, 4.0)
            self.vy = random.uniform(-5.0, 1.0)
            self.max_life = random.uniform(20, 45)
            self.radius = random.randint(2, 5)
        
        self.life = self.max_life
        self.twinkle_speed = random.uniform(0.3, 0.8)
        self.twinkle_offset = random.uniform(0, math.pi * 2)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Gravity slowly pulls the magic back down, creating a gorgeous 
        # firework-like arc for the upward burst!
        self.vy += 0.4 
        # Air resistance slows horizontal movement over time
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

def get_hand_openness(landmarks):
    """Calculates if the hand is open or closed based on fingertip distance to wrist."""
    wrist = landmarks[0]
    scale_ref = landmarks[9] # Middle finger knuckle
    scale = math.hypot(wrist.x - scale_ref.x, wrist.y - scale_ref.y) or 1e-6
    
    tips = [4, 8, 12, 16, 20] # Fingertip indices
    distances = [math.hypot(wrist.x - landmarks[t].x, wrist.y - landmarks[t].y) for t in tips]
    return (sum(distances) / len(distances)) / scale

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
    hand_states = {} # Remembers if your hand was open or closed in the last frame
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
            for idx, landmarks in enumerate(result.hand_landmarks):
                # Optionally, comment this line out to hide the red/white skeleton lines
                drawing_utils.draw_landmarks(frame, landmarks, hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS)

                # Calculate hand state (open vs closed)
                openness = get_hand_openness(landmarks)
                prev_openness = hand_states.get(idx, openness)
                
                # THE BURST TRIGGER: Fist (< 1.3) transitions to Open Hand (> 1.9)
                if prev_openness < 1.3 and openness > 1.9:
                    palm_x = int(landmarks[9].x * w)
                    palm_y = int(landmarks[9].y * h)
                    # Spawn 400 particles instantly shooting upward!
                    for _ in range(400):
                        particles.append(Particle(palm_x, palm_y, is_burst=True))
                
                # Update memory for the next frame
                hand_states[idx] = openness

                # Only spawn standard fingertip magic if the hand is actually OPEN (> 1.5)
                # If your hand is closed/fist, this skips completely (NO SPARKLES)
                if openness > 1.5:
                    for tip_idx in [4, 8, 12, 16, 20]:
                        tx, ty = int(landmarks[tip_idx].x * w), int(landmarks[tip_idx].y * h)
                        for _ in range(25):
                            particles.append(Particle(tx, ty, is_burst=False))

        # Memory cap: 6000 particles 
        particles = [p for p in particles if p.life > 0][-6000:] 
        
        # 1. Create the pitch-black canvas
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # 2. Draw glowing particles to the canvas
        for p in particles:
            p.update()
            p.draw(overlay)

        # 3. Additive Blending: Layer the light perfectly onto the webcam feed
        cv2.add(frame, overlay, dst=frame)

        cv2.imshow("Elsa Upward Blast", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()