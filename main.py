import cv2
import numpy as np


net = cv2.dnn.readNet("F:\imagedetection\yolov3.weights","F:\imagedetection\yolov3.cfg")
classes = []
with open("F:\imagedetection\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

 
img = cv2.imread("F:\imagedetection\image16.jpeg")
new_width = 800
new_height = 600
resized_img = cv2.resize(img, (new_width, new_height))


blob = cv2.dnn.blobFromImage(resized_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * new_width)
            center_y = int(detection[1] * new_height)
            w = int(detection[2] * new_width)
            h = int(detection[3] * new_height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(resized_img, label, (x, y + 30), font, 1, color, 2)


cv2.imshow("Object Detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

