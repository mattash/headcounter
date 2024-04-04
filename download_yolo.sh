#!/bin/bash

# Set URLs for the YOLOv3 weights and cfg
YOLOV3_WEIGHTS_URL="https://pjreddie.com/media/files/yolov3.weights"
YOLOV3_CFG_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"

# Download YOLOv3 weights
wget -O yolov3.weights $YOLOV3_WEIGHTS_URL

# Download YOLOv3 cfg
wget -O yolov3.cfg $YOLOV3_CFG_URL

echo "Download completed."
