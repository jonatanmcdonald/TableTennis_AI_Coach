import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision import PoseLandmark
  

#Load the trained model
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_layers=2, num_classes=4):
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
    PoseLandmark.LEFT_SHOULDER.value,
    PoseLandmark.RIGHT_SHOULDER.value,
    PoseLandmark.RIGHT_ELBOW.value,
    PoseLandmark.RIGHT_WRIST.value,
    PoseLandmark.LEFT_HIP.value,
    PoseLandmark.RIGHT_HIP.value
]

def normalize_frame(keypoints):
    """
    Normalize keypoints:
    Center on hip midpoint
    -Scale by torso length (distance between shoulders and hips)
    """
    keypoints = np.array(keypoints).reshape(-1, 3) #x, y, z

    #Hip midpoint
    left_hip = keypoints[keep_indices.index(PoseLandmark.LEFT_HIP.value)]
    right_hip = keypoints[keep_indices.index(PoseLandmark.RIGHT_HIP.value)]
    hip_mid = (left_hip + right_hip) / 2

    #Shoulder midpoint
    left_shoulder = keypoints[keep_indices.index(PoseLandmark.LEFT_SHOULDER.value)]
    right_shoulder = keypoints[keep_indices.index(PoseLandmark.RIGHT_SHOULDER.value)]
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    #Torso length
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_length == 0:
        torso_length = 1.0 #prevent divide by zero

    #Center and scale
    keypoints_norm = (keypoints - hip_mid) / torso_length
    return keypoints_norm.flatten().tolist() #18 features    


def get_wrist_speed(prev, curr):
    prev = np.array(prev)
    curr = np.array(curr)
    return np.linalg.norm(curr[9:12] - prev[9:12])


def is_wrist_moving(buffer, threshold=0.15, min_frames=5):
    if len(buffer) < min_frames:
        return False
    
    movement = 0.0
    valid_steps = 0
    for i in range(-min_frames, -1):
        prev = np.array(buffer[i])
        curr = np.array(buffer[i+1])
        # only count movement above a tiny noise floor
        diff = np.linalg.norm(curr[9:12] - prev[9:12])
        if diff > 0.03:  # ignore movements smaller than 0.03
            movement += diff
            valid_steps += 1

    if valid_steps == 0:
        return False

    avg_movement = movement / valid_steps
    return avg_movement > threshold


#Loads Model  
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
velocity_buffer = [] #for movement detection
swing_frames = []

collecting_swing = False
cooldown = 0

frame_idx = 0

#swing_in_progress = False  #tracks if mid swing
#movement_counter = 0  #counts consecutive frames of movement
#cooldown = 0          #frames to wait after swing before allowing new one

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
                frame_keypoints.extend([landmark.x, landmark.y, landmark.z])
                #x = int(landmark.x * frame.shape[1])
                #y = int(landmark.y * frame.shape[0])

            frame_keypoints = normalize_frame(frame_keypoints)
            
            #smoothing with simple moving average
            if buffer:
                alpha = 0.5
                prev = np.array(buffer[-1])
                curr = np.array(frame_keypoints)
                frame_keypoints = (alpha * prev + (1 - alpha) * curr).tolist()

            buffer.append(frame_keypoints)
            if len(buffer) > 30:
                buffer.pop(0)

        
        #swing detection logic
        if len(buffer) > 1:
            v = get_wrist_speed(buffer[-2], buffer[-1])
            velocity_buffer.append(v)

            if len(velocity_buffer) > 30:
                velocity_buffer.pop(0)

            #cooldown countdown
            if cooldown > 0:
                cooldown -= 1
            
            START_THRESHOLD = 0.08
            END_THRESHOLD = 0.04

            #start of swing
            if not collecting_swing and v > START_THRESHOLD and cooldown == 0:
                collecting_swing = True
                swing_frames = buffer[-min(5, len(buffer)):]
                print(f"Swing started at frame {frame_idx}, collecting frames...")

            #Collect frames
            if collecting_swing:
                swing_frames.append(buffer[-1])

            #end of swing
            if collecting_swing and v < END_THRESHOLD:
                collecting_swing = False
                print(f"Swing ended at frame {frame_idx}, Frames collected: {len(swing_frames)}")

                cooldown = 20  # set cooldown to prevent immediate new swing detection
                swing_frames_copy = swing_frames.copy() # saves frames for classification 
                swing_frames = [] # reset for next swing 

                #run model only if valid swing detected
                if len(swing_frames_copy) > 10:
                    seq = torch.tensor(np.array(swing_frames_copy, dtype=np.float32)).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(seq)
                        pred_class = torch.argmax(output, dim=1).item()

                    feedback = label_map_inv[pred_class]

                    print(f"Swing classified as: '{feedback}', Cooldown: {cooldown}")

                    cv2.putText(annotated_frame, f"Feedback: {feedback}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    print("Live Feedback:", feedback)

                    cooldown = 20

                swing_frames = []
        
        cv2.imshow("Live Stroke Feedback", annotated_frame)

        key = cv2.waitKey(1)
        if key == 27: #ESC to exit
            break


cap.release()
cv2.destroyAllWindows()