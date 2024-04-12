import cv2
import numpy as np
import sys

def load_yolo():
    # Assuming yolov3.cfg and yolov3.weights are correctly set up
    net = cv2.dnn.readNet("photocounter-lib/yolov3.weights", "photocounter-lib/yolov3.cfg")
    layer_names = net.getLayerNames()
    try:
        output_layer_indices = net.getUnconnectedOutLayers().flatten()
    except AttributeError:
        output_layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[index - 1] for index in output_layer_indices]

    # Load all class names from coco.names
    with open("photocounter-lib/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    # print(f"Loaded {len(classes)} classes.")

    return net, classes, output_layers



def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs

def get_box_dimensions(outputs, height, width, conf_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > conf_threshold:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    # Apply Non-maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, nms_threshold)
    boxes = [boxes[i] for i in indices.flatten()]
    confs = [confs[i] for i in indices.flatten()]
    class_ids = [class_ids[i] for i in indices.flatten()]

    return boxes, confs, class_ids



def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # Ensure color is a tuple of integers
            color = tuple([int(c) for c in colors[class_ids[i]]])
            # Convert coordinates to integers
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(img, label, (int(x), int(y) - 5), font, 1, color, 1)

            
def count_people(img_path):
    model, classes, output_layers = load_yolo()
    image = cv2.imread(img_path)
    height, width, channels = image.shape
    outputs = detect_objects(image, model, output_layers)
    # Adjust these thresholds as needed
    conf_threshold = 0.3
    nms_threshold = 0.4
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width, conf_threshold, nms_threshold)


    # Define colors for different classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Draw labels on the image
    # draw_labels(boxes, confs, colors, class_ids, classes, image)
    
    # Display the image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Count and print the number of people
    people_count = sum([1 for i in class_ids if i < len(classes) and classes[i] == "person"])
    print(f"Number of people in image: {people_count}")

# Replace 'path_to_image.jpg' with the path to the image file you want to process
count_people(sys.argv[1])
