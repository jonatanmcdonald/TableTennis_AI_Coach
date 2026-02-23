import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision import PoseLandmark
  

#Load the trained model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=4):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
#Joints we care about
keep_indices = [
    11, 12, 14, 16, 23, 24 #LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP
]

def normalize_frame(keypoints):
    """
    Normalize keypoints:
    Center on hip midpoint
    -Scale by torso length (distance between shoulders and hips)
    """
    keypoints = np.array(keypoints).reshape(-1, 2) #(6.2)

    #Hip midpoint
    left_hip = keypoints[keep_indices.index(PoseLandmark.LEFT_HIP)]
    right_hip = keypoints[keep_indices.index(PoseLandmark.RIGHT_HIP)]
    hip_mid = (left_hip + right_hip) / 2

    #Shoulder midpoint
    left_shoulder = keypoints[keep_indices.index(PoseLandmark.LEFT_SHOULDER)]
    right_shoulder = keypoints[keep_indices.index(PoseLandmark.RIGHT_SHOULDER)]
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    #Torso length
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_length == 0:
        torso_length = 1.0 #prevent divide by zero

    #Center and scale
    keypoints_norm = (keypoints - hip_mid) / torso_length
    return keypoints_norm.flatten().tolist() #12 features    

def is_wrist_moving(buffer, threshold=0.02):
    if len(buffer) < 2:
        return False
    prev = np.array(buffer[-2])
    curr = np.array(buffer[-1])
    movement = np.linalg.norm(curr[2:4] - prev[2:4])
    return movement > threshold
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier().to(device)
model.load_state_dict(torch.load('lstm_forehand_front.pth', map_location=device))
model.eval()

#PoseLandmarker setup
model_path = './pose_landmarker.task'
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)

#Joints we care about
#keep_indices = [
#    11, 12, 14, 16, 23, 24 #LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP
#]
label_map_inv = {0: "good", 1: "elbow", 2: "low", 3: "norot"}

buffer = [] #rolling buffer of last 30 frames
frame_idx = 0

cap = cv2.VideoCapture(0) #Open camera

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = frame_idx * 33
        frame_idx += 1

        #Run pose detection
        try: 
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print("Pose detection error:", e)
            continue

        annotated_frame = frame.copy()

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            frame_keypoints = []

            for idx in keep_indices:
                landmark = pose[idx]
                frame_keypoints.extend([landmark.x, landmark.y])
                #x = int(landmark.x * frame.shape[1])
                #y = int(landmark.y * frame.shape[0])

            frame_keypoints = normalize_frame(frame_keypoints)
            buffer.append(frame_keypoints)
            if len(buffer) > 30:
                buffer.pop(0)

        #Run prediction when buffer is full
        if len(buffer) == 30 and is_wrist_moving(buffer):
            seq = torch.tensor(np.array(buffer, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(seq)
                pred_class = torch.argmax(output, dim=1).item()
            feedback = label_map_inv[pred_class]
            cv2.putText(annotated_frame, f"Feedback: {feedback}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Live Feedback:", feedback)
            
        cv2.imshow("Live Stroke Feedback", annotated_frame)
        key = cv2.waitKey(1)
        if key == 27: #ESC to exit
            break

cap.release()
cv2.destroyAllWindows()