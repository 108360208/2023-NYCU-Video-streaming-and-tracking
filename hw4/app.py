from flask import Flask, render_template, Response, request
import cv2
import cv2
from ultralytics import YOLO
import torchvision.transforms as transforms
from ultralytics import settings
import random
from collections import defaultdict
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(0)

ignored_track_ids = set()

if not cap.isOpened():
    print("無法打開相機鏡頭")
    exit()

def generate_unique_colors(num_colors):
    colors = set()
    while len(colors) < num_colors:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.add(color)
    return list(colors)

def click_event(x, y, param):
    global ignored_track_ids
    
    for (x1, y1, x2, y2), track_id in param:
        if x1 < x < x2 and y1 < y < y2:
            if track_id not in ignored_track_ids:
                # print("Unignore track id: {}".format(track_id))
                ignored_track_ids.add(track_id)
                print("Ignore track id: {}".format(track_id))  
    return "Position updated successfully"
results = None 
boxes_to_display = []
ignored_track_ids = dict()
cur_box = []
def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break
        else:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], iou = 0.2, conf = 0.2)
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy
                track_ids = results[0].boxes.id.int().cpu().tolist()
                global boxes_to_display
                global ignored_track_ids
                global cur_box
                cur_box = []
                boxes_to_display = []
                for box, track_id in zip(boxes, track_ids):
                    cur_box.append((box, track_id))
                for box, track_id in zip(boxes, track_ids):
                    if (track_id not in ignored_track_ids) or (not ignored_track_ids[track_id]):
                        boxes_to_display.append((box, track_id))

                for (x1, y1, x2, y2), track_id in boxes_to_display:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 2)
                    frame = cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[track_id % len(colors)], 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                # print(type(buffer))
                frame = buffer.tobytes()
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

num_colors = 1000
colors = generate_unique_colors(num_colors)

model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/update_position', methods=['POST'])
def update_position():
    data = request.get_json()
    x = data['x']
    y = data['y']
    print(x,y)
    global boxes_to_display
    global ignored_track_ids
    global cur_box
    for (x1, y1, x2, y2), track_id in cur_box:
        if x1 < x < x2 and y1 < y < y2:
            if track_id in ignored_track_ids:
                ignored_track_ids[track_id] = not ignored_track_ids[track_id]
            else :
                ignored_track_ids[track_id] = True
            if ignored_track_ids[track_id]:
                print("Ignore track id: {}".format(track_id))  
            else:
                print("Unignore track id: {}".format(track_id))
    return 'Position updated successfully'
if __name__ == '__main__':
    app.run(debug=True)
