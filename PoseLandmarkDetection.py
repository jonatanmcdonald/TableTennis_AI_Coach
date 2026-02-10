import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

img = cv2.imread('./Static Image.jpg')
#img = cv2.resize(img, (350, 350))
cv2.imshow('Image', img)

base_options = python.BaseOptions(model_asset_path='./pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

image = mp.Image.create_from_file('./Static Image.jpg')
detection_result = detector.detect(image)

dir(detection_result)

img = cv2.imread('./Static Image.jpg')
h, w, _ = img.shape

len(detection_result.pose_landmarks[0])

img = cv2.imread('./Static Image.jpg')
for lmarks in detection_result.pose_landmarks[0]:
    x_cord = int(lmarks.x * w)
    y_cord = int(lmarks.y * h)
    img = cv2.circle(img, (x_cord, y_cord), 5, (0, 255, 0), -1)
#img = cv2.resize(img, (350, 350))
cv2.imshow('Image with Landmarks', img)
waitKey = cv2.waitKey(0)
cv2.destroyAllWindows()
