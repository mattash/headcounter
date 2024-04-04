# Head Counter

## Description
This project uses the YOLO (You Only Look Once) real-time object detection system to count the number of people in an image.

## Installation
This project requires Python 3 and the following Python libraries installed:

- OpenCV
- Numpy

To install these libraries, you can use pip:

```
bash
pip install opencv-python numpy
```

## Usage
To run this project, you can use the following command:

```
python3 photo_counter.py <path_to_image>
```

This command will count "people" using OpenCV 

```
python3 face_counter.py <path_to_image>
```

This command will count faces only

Replace <path_to_image> with the path to the image you want to process.

## TODO
Build a seperate clip for counting faces in a video.