import cv2
import os

def save_frames_with_label(video_path, output_folder, label):
    """Split video into frames and save them with the specified label"""
    # Create a folder for the video label
    label_folder = os.path.join(output_folder, f"label_{label}")
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    
    # Open the video file
    video_cap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, frame = video_cap.read()
        if success:
            frame_filename = os.path.join(label_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save the frame
            count += 1
    
    video_cap.release()

# Define the paths and labels for each video
videos = [
    {"path": "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/frontpull_2hands_eeg_mi.mp4", "label": 0},
    {"path": "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/left_leg_eeg_mi.mp4", "label": 1},
    {"path": "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/pull_2hands_eeg_mi.mp4", "label": 2},
    {"path": "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/right_leg_eeg_mi.mp4", "label": 3},
    {"path": "/home/meow/my_data_disk_5T/mi_eeg_uacg_data/up_2hands_eeg_mi.mp4", "label": 4},
]

# Split and save frames
output_folder = "frames_output"
for video in videos:
    save_frames_with_label(video["path"], output_folder, video["label"])
