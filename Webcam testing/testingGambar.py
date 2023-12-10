import cv2
import numpy as np

confidence_thresh = 0.05
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Load image
image = cv2.imread("images.jpeg")
height, width, channels = image.shape

# Preprocess image
blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Get information about detected objects
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_thresh:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, 0.4)

# Draw bounding boxes on the image
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"
        cv2.putText(image, text, (x, y - 10), font, 0.5, color, 2)
        print(text)

# Show the resulting image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
