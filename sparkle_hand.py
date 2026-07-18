# Import OpenCV for webcam capture and display.
import cv2
import sys


def main():
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

    window_name = "Sparkle Hand"

    # Continuously read frames from the webcam until the user quits.
    while True:
        # Read one frame; ret is True when a frame was captured successfully.
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Flip horizontally so the feed behaves like a mirror.
        mirrored_frame = cv2.flip(frame, 1)

        # Show the mirrored frame in a window.
        cv2.imshow(window_name, mirrored_frame)

        # Wait 1 ms for a key press; exit cleanly when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
