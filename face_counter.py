import cv2
import numpy as np
import sys

def load_yoloface():
    net = cv2.dnn.readNet("yoloface-lib/yolov3-wider_16000.weights", "yoloface-lib/yolov3-face.cfg")
    layer_names = net.getLayerNames()
    try:
        # Attempt to flatten the array and subtract 1 for 0-based indexing
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        # If flattening doesn't work, it's likely already flat, or a different issue with the version
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


def detect_faces(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs

def get_boxes(outputs, height, width, conf_threshold=0.5):
    boxes = []
    confs = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(confidence))
    return boxes, confs

def draw_labels(boxes, img):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def count_faces(img_path):
    model, output_layers = load_yoloface()
    image = cv2.imread(img_path)
    height, width, channels = image.shape
    outputs = detect_faces(image, model, output_layers)
    boxes, confs = get_boxes(outputs, height, width)

    # Apply Non-maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.5, nms_threshold=0.4)
    final_boxes = [boxes[i] for i in indices.flatten()]

    # Draw labels on the image
    draw_labels(final_boxes, image)
    
    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the number of faces detected
    print(f"Number of faces detected: {len(final_boxes)}")

# Get the filename from the command line argument
filename = sys.argv[1]

# Call the count_faces function with the filename
count_faces(filename)
