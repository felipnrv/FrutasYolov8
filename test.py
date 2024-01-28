"""
from roboflow import Roboflow
import cv2
import numpy as np
rf = Roboflow(api_key="8049iBONyQC4bosBrcaa")
project = rf.workspace("proyecto-detec-frutas").project("frutas-nzthy")
dataset = project.version(1).download("yolov8")

data = {"name": "daniel ","age":"21","?":False}

#db.push(data)

#para poner el nombre de la llave
#db.child("users").child("Usuario").set(data)

#leer 
#felipe=db.child("users").child("Usuario").get()
#print(felipe.val())

#actualizar
#db.child("users").child("Usuario").update({"name":"pablo"})

#eliminar
#db.child("users").child("Usuario").child("age").remove()

#eliminar todo el nodo
db.child("users").child("Usuario").remove()

#m.model()
#data1=m.model()



#firebase = pyrebase.initialize_app(config)
#db = firebase.database()

#data = {"name": "daniel", "age": "21", "?": False}

# Push data to Firebase
#db.push(data)

# Get terminal output
output = subprocess.check_output(["python", "main.py"])

# Push terminal output to Firebase
db.push({"terminal_output": output.decode("utf-8")})
"""
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to get output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Function to draw bounding box and label
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load image
image = cv2.imread("image.jpeg")

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), True, crop=False)

# Set input for the network
net.setInput(blob)

# Forward pass through the network
outs = net.forward(get_output_layers(net))

# Initialize counters for each class
# Initialize class counts
class_counts = {class_name: 0 for class_name in classes}

# Define the line position
line_position = 300

# Process each output layer
for out in outs:
    # Process each detection
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # Update class count
            class_name = classes[class_id]
            class_counts[class_name] += 1

            # Draw bounding box and label
            draw_prediction(image, class_id, confidence, x, y, x + width, y + height)

            # Check if object passes the line
            if y + height > line_position:
                # Display class count at the corner of the screen
                cv2.putText(image, f"{class_name}: {class_counts[class_name]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Process each output layer
for out in outs:
    # Process each detection
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Get bounding box coordinates
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            # Update class count
            class_name = classes[class_id]
            class_counts[class_name] += 1

            # Draw bounding box and label
            draw_prediction(image, class_id, confidence, x, y, x + width, y + height)

# Display image with class counts
for i, (class_name, count) in enumerate(class_counts.items()):
    cv2.putText(image, f"{class_name}: {count}", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
