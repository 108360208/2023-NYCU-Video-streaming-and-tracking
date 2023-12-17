import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from ultralytics import settings
from deepsort_tracker import DeepSort
import random


from collections import defaultdict
import numpy as np
# Load the YOLOv8 model
# model = YOLO('yolov8x.pt')
# results = model.track(source="hard_9.mp4", show=True, tracker="bytetrack.yaml", classes=[0] , conf=0.7, iou=0.95) 
def generate_unique_colors(num_colors):
    colors = set()
    while len(colors) < num_colors:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.add(color)
    return list(colors)


num_colors = 30
colors = generate_unique_colors(num_colors)

# colors = [
#     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#     for j in range(10)
# ]
# Load the YOLOv8 model
model = YOLO('yolov8x.pt')
# Open the video file
video_path = "hard_9.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

cap_out = cv2.VideoWriter(
    "out.mp4",
    cv2.VideoWriter_fourcc(*"MP4V"),
    cap.get(cv2.CAP_PROP_FPS),
    (1280, 720),
)
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
        # for index, track_id in enumerate(track_ids):
        #     bbox = boxes[index]
        #     color = colors[int(track_id) % len(colors)]
        #     cv2.rectangle(
        #         frame,
        #         (int(bbox[0]), int(bbox[1])),
        #         (int(bbox[2]), int(bbox[3])),
        #         color,
        #         thickness=4,
        #     )

        #     display_text = "ID: " + str(track_id)
        #     (text_width, text_height), _ = cv2.getTextSize(
        #         display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        #     )

        #     cv2.rectangle(
        #         frame,
        #         (int(bbox[0]), int(bbox[1]) - 30),
        #         (
        #             int(bbox[0]) + int(text_width),
        #             int(bbox[1]),
        #         ),
        #         color,
        #         -1,
        #     )

        #     text_x = int(bbox[0]) + 5
        #     text_y = int(bbox[1]) + text_height + 5
        #     cv2.putText(
        #         frame,
        #         display_text,
        #         (int(bbox[0]), int(bbox[1]) - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.75,
        #         (255, 255, 255),
        #         2,
        #     )
        # print("track_ids: ", results[0])
        # Plot the tracks
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)

            # Draw the tracking lines
            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

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
# classes = model.names

# video_path = "hard_9.mp4"
# cap = cv2.VideoCapture(video_path)

# colors = [
#     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#     for j in range(10)
# ]

# ret, image = cap.read()

# cap_out = cv2.VideoWriter(
#     "out.mp4",
#     cv2.VideoWriter_fourcc(*"MP4V"),
#     cap.get(cv2.CAP_PROP_FPS),
#     (image.shape[1], image.shape[0]),
# )

# while ret:
#     results = model.predict(source=image, classes=[0])

#     for result in results:
#         detections = []
#         boxes = result.boxes
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2 = r[:4]
#             w, h = x2 - x1, y2 - y1
#             coordinates = list((int(x1), int(y1), int(w), int(h)))
#             conf = r[4]
#             clsId = int(r[5])
#             cls = classes[clsId]
#             if cls == "person" and conf > 0.7:
#                 detections.append((coordinates, conf, cls))

#         # print("detections: ", detections)
#         tracks = object_tracker.update_tracks(detections, frame=image)

#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             track_id = track.track_id
#             bbox = track.to_ltrb()
#             color = colors[int(track_id) % len(colors)]
#             cv2.rectangle(
#                 image,
#                 (int(bbox[0]), int(bbox[1])),
#                 (int(bbox[2]), int(bbox[3])),
#                 color,
#                 thickness=4,
#             )

#             display_text = "ID: " + str(track_id)
#             (text_width, text_height), _ = cv2.getTextSize(
#                 display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
#             )

#             cv2.rectangle(
#                 image,
#                 (int(bbox[0]), int(bbox[1]) - 30),
#                 (
#                     int(bbox[0]) + int(text_width),
#                     int(bbox[1]),
#                 ),
#                 color,
#                 -1,
#             )

#             text_x = int(bbox[0]) + 5
#             text_y = int(bbox[1]) + text_height + 5
#             cv2.putText(
#                 image,
#                 display_text,
#                 (int(bbox[0]), int(bbox[1]) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.75,
#                 (255, 255, 255),
#                 2,
#             )

#     cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
#     cv2.imshow("Result", image)
#     cap_out.write(image)
#     ret, image = cap.read()

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()