import math
import numpy as np

from mediapipe.tasks.python.vision import PoseLandmark

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    #Vectors from B to A and B to C
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    #Clip to handle numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    #convert to degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def elbow_angle(pose, side="right"):
    if side.lower() == "right":
        shoulder = pose[PoseLandmark.RIGHT_SHOULDER]
        elbow = pose[PoseLandmark.RIGHT_ELBOW]
        wrist = pose[PoseLandmark.RIGHT_WRIST]
    else:
        shoulder = pose[PoseLandmark.LEFT_SHOULDER]
        elbow = pose[PoseLandmark.LEFT_ELBOW]
        wrist = pose[PoseLandmark.LEFT_WRIST]
    return calculate_angle(shoulder, elbow, wrist)

def elbow_to_body_distance(pose, side="right"):
    if side.lower() == "right":
        shoulder = pose[PoseLandmark.RIGHT_SHOULDER]
        hip = pose[PoseLandmark.RIGHT_HIP]
        elbow = pose[PoseLandmark.RIGHT_ELBOW]
    else:
        shoulder = pose[PoseLandmark.LEFT_SHOULDER]
        hip = pose[PoseLandmark.LEFT_HIP]
        elbow = pose[PoseLandmark.LEFT_ELBOW]
    
    #midpoint between shoulder and hip
    torso_midpoint_x = (shoulder.x + hip.x) / 2
    torso_midpoint_y = (shoulder.y + hip.y) / 2

    #distance from elbow to torso midpoint
    distance = math.sqrt((elbow.x - torso_midpoint_x) ** 2 + 
                         (elbow.y - torso_midpoint_y) ** 2)
    
    return distance