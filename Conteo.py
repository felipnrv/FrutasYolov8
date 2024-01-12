import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import*

# Load YOLO model
model = YOLO('D:/Felipe/Codigo/Model/best.pt')

def RGB(event, x, y, flags, param):
    """
    Mouse event callback function that prints the coordinates of the mouse cursor when it moves.

    Args:
        event (int): The type of mouse event.
        x (int): The x-coordinate of the mouse cursor.
        y (int): The y-coordinate of the mouse cursor.
        flags (int): Additional flags.
        param (object): Additional parameters.

    Returns:
        None
    """
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)

# Create a named window and set the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video capture
cap = cv2.VideoCapture(0)

# Read class names from file
my_file = open("config.yaml", "r")
data = my_file.read()
class_list = data.split("\n") 

# Initialize counters and trackers
count = 0
tracker = Tracker()
tracker1 = Tracker()
tracker2 = Tracker()
cy1 = 184
cy2 = 209
offset = 8
upcar = {}
downcar = {}
countercarup = []
countercardown = []
downbus = {}
counterbusdown = []
upbus = {}
counterbusup = []
downtruck = {}
countertruckdown = []

while True:    
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection using YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    list = []
    list1 = []
    list2 = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'Pitahaya' in c:
            list.append([x1, y1, x2, y2])
        elif 'Granadilla' in c:
            list1.append([x1, y1, x2, y2])
        elif 'Arandanos' in c:
            list2.append([x1, y1, x2, y2])

    # Update object trackers
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx3 = int(x3 + x4) // 2
        cy3 = int(y3 + y4) // 2
        cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)

    # Draw counting lines
    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (3, cy2), (1016, cy2), (0, 0, 255), 2)

    # Display frame
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()