# opencv-face-filters

Snapchat-like face filters for laptop camera/webcam or video. (Currently implemented filters are: glasses, mustache, and cat whiskers.)

## How it works

Faces are detected from camera or video frames using OpenCV Haar Cascades. The detected faces are passed to a convolutional neural network implemented in Keras and trained on facial keypoint data (obtained from [Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data)). The predicted facial keypoints are then used to determine the placement of a filter in the image.

## Prerequisites

* Python 3.6
* NumPy
* Keras 2.2.2
* OpenCV-Python 3.4.3

## Demo

Facial filters applied on laptop camera:

![Alt Text](https://github.com/zezzer/opencv-face-filters/blob/master/images/demo.gif)

Facial filters applied on a specified video input:

![Alt Text](https://github.com/zezzer/opencv-face-filters/blob/master/images/kids.gif)

## Usage

Run the following command to launch the app:

`python3 run.py`

Use the -v option to run facial filters on a video. 

`python3 run.py -v /path/to/video`

After the camera or video frame pops up, press the space bar to switch between filters.

Press the 'q' key to quit the application.

## Notes

Facial keypoint dataset was obtained from Kaggle: https://www.kaggle.com/c/facial-keypoints-detection/data.

Haar cascade was obtained from here: https://github.com/opencv/opencv/tree/master/data/haarcascades.