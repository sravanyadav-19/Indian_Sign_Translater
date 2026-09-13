import cv2
import os
import time

# --- CONFIGURATION ---
VIDEO_SOURCE_DIR = os.path.join('..', 'public', 'signs')
DATASET_DEST_DIR = os.path.join('dataset')
FRAMES_TO_EXTRACT_PER_VIDEO = 50

# --- SCRIPT ---
if not os.path.exists(DATASET_DEST_DIR):
    os.makedirs(DATASET_DEST_DIR)

video_files = [f for f in os.listdir(VIDEO_SOURCE_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]

for video_file in video_files:
    sign_name = os.path.splitext(video_file)[0].lower().replace(" ", "")
    video_path = os.path.join(VIDEO_SOURCE_DIR, video_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < FRAMES_TO_EXTRACT_PER_VIDEO:
        print(f"Warning: Video {video_file} has fewer frames than requested.")
        frame_indices_to_capture = range(total_frames)
    else:
        frame_indices_to_capture = [int(i) for i in range(0, total_frames, total_frames // FRAMES_TO_EXTRACT_PER_VIDEO)]

    print(f"--- Processing '{video_file}' for sign '{sign_name}' ---")
    
    saved_frame_count = 0
    current_frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame_index in frame_indices_to_capture:
            image_name = f"{sign_name}_{int(time.time() * 1000)}_{saved_frame_count}.jpg"
            save_path = os.path.join(DATASET_DEST_DIR, image_name)
            
            cv2.imwrite(save_path, frame)
            print(f"Saved {save_path}")
            saved_frame_count += 1

        current_frame_index += 1
        
    cap.release()

print("\n--- Frame extraction complete. ---")
cv2.destroyAllWindows()