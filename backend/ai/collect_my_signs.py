import argparse
import cv2
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture labeled webcam images for one ISL sign."
    )
    parser.add_argument("sign", help="Label for the sign, for example: hello")
    parser.add_argument("--images", type=int, default=200, help="Frames to capture (default: 200)")
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between frames (default: 0.1)")
    return parser.parse_args()


# Command-line options make repeated collection sessions reproducible without
# editing this source file for every new sign.
args = parse_args()
SIGN_WORD = args.sign.strip().lower().replace(" ", "_")
NUM_IMAGES = args.images
CAPTURE_DELAY = args.delay
DATA_PATH = os.path.join(os.path.dirname(__file__), 'dataset')

if not SIGN_WORD:
    raise SystemExit("The sign label cannot be empty.")
if NUM_IMAGES < 2:
    raise SystemExit("--images must be at least 2.")
if CAPTURE_DELAY < 0:
    raise SystemExit("--delay cannot be negative.")

# --- SCRIPT ---
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Initialize webcam
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print(f"\n--- Preparing to capture images for: '{SIGN_WORD}' ---")
print("Get in position! Starting in 5 seconds...")

# Countdown before starting
for i in range(5, 0, -1):
    print(i, end='...', flush=True)
    time.sleep(1)

print("\nSTARTING CAPTURE! Vary your hand position slightly for each frame.")

for img_num in range(NUM_IMAGES):
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    # Flip the frame horizontally (mirror view) for intuitive posing
    frame = cv2.flip(frame, 1)

    # Show instructions on the screen
    cv2.putText(frame, f'Capturing: {SIGN_WORD}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Image {img_num + 1}/{NUM_IMAGES}', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Show the frame to the user
    cv2.imshow('Sign Language Data Collection', frame)

    # Generate a unique filename using a timestamp
    image_name = os.path.join(DATA_PATH, f'{SIGN_WORD}_{int(time.time() * 1000)}.jpg')
    # Save the original (un-flipped) frame if you want, or the flipped one. We save the flipped one so what you see is what you get.
    cv2.imwrite(image_name, frame)
    
    print(f"Saved {image_name}")

    # Wait for 100ms between captures. Press 'q' to quit early.
    if cv2.waitKey(max(1, round(CAPTURE_DELAY * 1000))) & 0xFF == ord('q'):
        break

print(f"\n--- Finished capturing for '{SIGN_WORD}' ---")
cap.release()
cv2.destroyAllWindows()