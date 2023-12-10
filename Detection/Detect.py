import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2
from antares_http import antares
from datetime import datetime

def get_current_day():
    # Get the current date and time
    current_datetime = datetime.now()

    # Extract and print the current day
    current_day = current_datetime.day
    return current_day

def save_number_to_file(number, filename):
    with open(filename, 'w') as file:
        file.write(str(number))
    print(f"Number {number} has been saved to {filename}.")

def get_number_from_file(filename):
    try:
        with open(filename, 'r') as file:
            saved_number = int(file.read())
        return saved_number
    except FileNotFoundError:
        print(f"File {filename} not found. No number saved yet.")
        save_number_to_file(0, 'saved_number.txt')
        save_number_to_file(get_current_day(), 'saved_date.txt')

        return None
    except ValueError:
        print(f"Error reading the number from {filename}. The file may be corrupted.")
        save_number_to_file(0, 'saved_number.txt')
        save_number_to_file(get_current_day(), 'saved_date.txt')

        return None


cred = credentials.Certificate("./key.json")
app = firebase_admin.initialize_app(cred, { 'storageBucket' : 'plate-detection-f6347.appspot.com' })

antares.setDebug(True)
antares.setAccessKey('870f1680796438f0:71e9fc8c307a7e39')

confidence_thresh = 0.1
metadataBefore = ''
personCount = 0
dishCount = 0

total = get_number_from_file('saved_number.txt')
save_number_to_file(get_current_day(), 'saved_date.txt')
dateText = get_number_from_file('saved_date.txt')


while(1):
    dateNow = get_current_day()
    dateText = get_number_from_file('saved_date.txt')
    bucket = storage.bucket()
    blob = bucket.get_blob("data/photo.jpg") #blob
    metadata = blob.metadata
    print(metadata)
    if metadata == metadataBefore:
        continue
    arr = np.frombuffer(blob.download_as_string(), np.uint8) #array of bytes
    image = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555) #actual image


    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load classes
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]

    # Load image
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

    for i in indexes:
        box = boxes[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        if classes[class_ids[i]] == 'person':
            personCount += 1
        elif classes[class_ids[i]] == 'dish':
            dishCount += 1

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

    # Show the resulting image
    print(personCount)
    print(dishCount)

    if dishCount > personCount:
        selisih = dishCount - personCount
    else:
        selisih = 0

    if dateNow >  dateText:
        save_number_to_file(dateNow, 'saved_date.txt')
        total = 0

    total += selisih
    save_number_to_file(total, 'saved_number.txt')

    myData = {
        'dishLeft' : selisih,
        'total' : total
    }
    antares.send(myData, 'desproKhalid', 'desproCam')
    cv2.imshow("Object Detection", image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    metadataBefore = metadata
    personCount = 0
    dishCount = 0
