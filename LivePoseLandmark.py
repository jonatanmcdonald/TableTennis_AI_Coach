from matplotlib import image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections
from mediapipe.tasks.python.vision import PoseLandmark

import numpy as np
import time
import cv2
from pprint import pprint

model_path = './pose_landmarker.task'

#Declare model variables and make name shorter for better display later
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


#Points I want to be detected
keep_indices = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE
]

keep_connections = [
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE)
    ]


#Create a pose landmarker with instance from live stream mode
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    
    
    #Draw a coordinates on image
    img = output_image.numpy_view().copy()
    h, w, _ = img.shape

    if result.pose_landmarks:
        #store deteced landmark pose in variable pose
        pose = result.pose_landmarks[0]
        
        #for idx, landmark in enumerate(pose):
            #name = PoseLandmark(idx).name
        for idx in keep_indices:
            landmark = pose[idx]
            name = PoseLandmark(idx).name
            print(
                f"{name}: "
                f"x={landmark.x:.3f}, "
                f"y={landmark.y:.3f}, "
                f"z={landmark.z:.3f}"
            )

        #draw landmarks on image
        #for landmark in pose:
        for idx in keep_indices:
            landmark = pose[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            #print(f"x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}\n")

        for connection in PoseLandmarksConnections.POSE_LANDMARKS:
            if connection.start in keep_indices and connection.end in keep_indices:
                start = pose[connection.start]
                end = pose[connection.end]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
    
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
   
    


    cv2.imshow('Live Landmarks', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        global running 
        running = False
        cap.release()
        cv2.destroyAllWindows()




#Initializing the pose landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

#capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video")
else:
    print("Video file opened successfully")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        #ret returns a boolean indicatin if the frame was read correctly
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        #convert frame to right mp image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_async(mp_image, timestamp_ms) 

    
    