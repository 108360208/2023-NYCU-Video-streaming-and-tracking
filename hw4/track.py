import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from ultralytics import settings
import random

from collections import defaultdict
import numpy as np

def generate_unique_colors(num_colors):
    colors = set()
    while len(colors) < num_colors:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.add(color)
    return list(colors)

num_colors = 30
colors = generate_unique_colors(num_colors)

model = YOLO('yolov8x.pt')
# Open the video file
video_path = "demo_2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

cap_out = cv2.VideoWriter(
    "out.mp4",
    cv2.VideoWriter_fourcc(*"MP4V"),
    cap.get(cv2.CAP_PROP_FPS),
    (1280, 720),
)
people_count = 0
people_count_list = []
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], iou = 0.2, conf = 0.2)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results and results[0].boxes and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for i in track_ids:
                if(i > people_count):
                    people_count = i

            # print("track_ids: ", track_ids)
            # print("boxes: ", boxes)
            # Visualize the results on the frame
            annotated_frame = results[0].plot(colors = colors)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            cap_out.write(annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)
        cap_out.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cap_out.release()
cv2.destroyAllWindows()
#find the max number in people_count\
print(f"Count: {(people_count)}")