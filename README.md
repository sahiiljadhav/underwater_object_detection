Object Detection System with Distance Measurement
Created by Sahil Jadhav
Project Overview
This project implements a real-time object detection system with distance measurement capabilities using a standard webcam. The system can detect objects within a defined restricted area and estimate their distance from the camera. It's built using Python and leverages deep learning for accurate object detection.
Features

Real-time object detection using webcam
Distance measurement for detected objects
Configurable restricted area monitoring
Camera calibration system
Visual feedback with bounding boxes and distance information
Detection confidence scoring

Prerequisites

Python 3.7 or higher
Webcam (built-in or external)

Required Libraries
bashCopypip install torch torchvision opencv-python numpy
Installation

Clone this repository or download the source files
Install the required dependencies:

bashCopypip install -r requirements.txt
Usage

Run the main program:

bashCopypython detection_system.py

Camera Calibration:

When the program starts, it will enter calibration mode
Hold an object of known width (e.g., A4 paper - 0.21 meters) at 1 meter distance
Draw a box around the object using your mouse
Press 'c' to complete calibration or 'q' to skip


Main Detection:

The system will show the webcam feed with:

Green rectangle: Restricted area
Red rectangles: Detected objects
Text overlay: Distance and confidence scores


Press 'q' to quit the program



Configuration
You can modify these parameters in the DetectionSystem class:

restricted_area: Change the monitored area dimensions
known_width: Adjust the reference object width (in meters)
Detection confidence threshold (default: 0.5)

Troubleshooting

Camera not detected:

Ensure your webcam is properly connected
Try changing the camera index in cv2.VideoCapture(0)


Inaccurate distance measurements:

Perform careful calibration with precise measurements
Ensure good lighting conditions
Keep the camera steady


Poor detection performance:

Ensure adequate lighting
Keep objects within the camera's field of view
Adjust the confidence threshold if needed



Future Improvements

Multi-object tracking
Distance measurement using stereo vision
Support for different camera types
Custom object detection training
Data logging and analysis features

License
This project is open source and available under the MIT License.
Contact
Created by Sahil Jadhav
Acknowledgments

PyTorch Team for the deep learning framework
OpenCV community for computer vision tools
Torchvision Team for pre-trained models
