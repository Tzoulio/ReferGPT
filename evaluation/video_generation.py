import cv2
import os
import re

# Configuration
input_folder = "/media/diana/Elements/leandro/PhD/MOTKitti/CasTrack/evaluation/results/sha_key/data/refer-kitti/0005/cars-in-the-same-direction-of-ours"  # Change this to your folder path
output_video = "/media/diana/Elements/leandro/PhD/MOTKitti/CasTrack/evaluation/results/sha_key/data/refer-kitti/video_results/cars-in-the-same-direction-of-ours.mp4"
frame_rate = 30  # Adjust as needed

# Get the list of image files and sort them numerically
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
image_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))  # Sort by frame number

# Read the first frame to get dimensions
first_frame = cv2.imread(os.path.join(input_folder, image_files[0]))
height, width, layers = first_frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 output
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Process each frame
for image in image_files:
    frame = cv2.imread(os.path.join(input_folder, image))
    video.write(frame)

# Release the video writer
video.release()
print(f"Video saved as {output_video}")
