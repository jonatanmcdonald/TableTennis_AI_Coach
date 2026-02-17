import time 
import mediapipe as mp
from mediapipe.tasks import python

import cv2
import numpy as np

model_path = "./efficientdet_lite0.tflite"

mp_image = mp.Image.create_from_file("./MediaPipe_Practice/girl_sitting.jpg")


BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE,
    score_threshold=0.5,
)   

with ObjectDetector.create_from_options(options) as detector:
    start = time.time()
    detection_result = detector.detect(mp_image)
    end = time.time()
    print(time.time() - start)
    print(len(detection_result.detections))

for data in detection_result.detections:
    print(data)
    print("\n")

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def visualize(
        image,
        detection_result
) -> np.ndarray:

    for detection in detection_result.detections:
        #Draw bounding box
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        #Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, 
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location,
                    cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image

image_copy = np.copy(mp_image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('Recognize Object', rgb_annotated_image)
waitKey = cv2.waitKey(0)
cv2.destroyAllWindows()