import cv2
from sympy import fps
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision import PoseLandmark

running = False

# MODEL (2D VERSION)
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


# JOINTS
keep_indices = [
    PoseLandmark.LEFT_SHOULDER.value,
    PoseLandmark.RIGHT_SHOULDER.value,
    PoseLandmark.RIGHT_ELBOW.value,
    PoseLandmark.RIGHT_WRIST.value,
    PoseLandmark.LEFT_HIP.value,
    PoseLandmark.RIGHT_HIP.value
]


# NORMALIZATION (2D)
def normalize_frame(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 2)  # x, y only

    # hips
    left_hip = keypoints[keep_indices.index(PoseLandmark.LEFT_HIP.value)]
    right_hip = keypoints[keep_indices.index(PoseLandmark.RIGHT_HIP.value)]
    hip_mid = (left_hip + right_hip) / 2

    # shoulders
    left_shoulder = keypoints[keep_indices.index(PoseLandmark.LEFT_SHOULDER.value)]
    right_shoulder = keypoints[keep_indices.index(PoseLandmark.RIGHT_SHOULDER.value)]
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_length == 0:
        torso_length = 1.0

    keypoints_norm = (keypoints - hip_mid) / torso_length

    # keep ONLY x,y
    return keypoints_norm[:, :2].flatten().tolist()  # 12 features


# SPEED (2D wrist)
def get_wrist_speed(prev, curr):
    prev = np.array(prev)
    curr = np.array(curr)
    return np.linalg.norm(curr[6:8] - prev[6:8])  # wrist x,y



# LOAD MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier().to(device)
model.load_state_dict(torch.load('./2D/lstm2D/lstm_forehand_front_2d_70.pth', map_location=device))
model.eval()



# MEDIAPIPE SETUP

model_path = './pose_landmarker.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)

label_map_inv = {0: "good", 1: "elbow", 2: "low", 3: "norot"}


# MAIN LOOP
def start_feedback(feedback_ui_callback=None):
    global running
    running = True

    buffer = []
    velocity_buffer = []
    swing_frames = []

    collecting_swing = False
    swing_classified = False
    cooldown = 0

    frame_idx = 0
    SEQ_LEN = 30
    SWING_COOLDOWN_FRAMES = 20  # minimum frames between swings
    MIN_SWING_FRAMES = 8        # minimum frames to consider a swing
    START_THRESHOLD = 0.10
    END_THRESHOLD = 0.08
    PEAK_THRESHOLD = 0.15     # wrist speed must reach this to count as a full swing

    v_peak = 0  # max wrist speed during current swing

    # --- Persistent feedback ---
    feedback_text = ""
    feedback_timer = 0
    FEEDBACK_DURATION = 30  # number of frames to keep feedback on screen (~1 second at 30 FPS)


    cap = cv2.VideoCapture(0)

    #fps = cap.get(cv2.CAP_PROP_FPS)
    #print("Camera FPS:", fps)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = frame_idx * 33
            frame_idx += 1

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print("Pose detection error:", e)
                continue

            annotated_frame = frame.copy()

            # POSE → FEATURES
            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                frame_keypoints = []
                for idx in keep_indices:
                    landmark = pose[idx]
                    frame_keypoints.extend([landmark.x, landmark.y])

                frame_keypoints = normalize_frame(frame_keypoints)

                # smoothing
                if buffer:
                    alpha = 0.6
                    prev = np.array(buffer[-1])
                    curr = np.array(frame_keypoints)
                    frame_keypoints = (alpha * prev + (1 - alpha) * curr).tolist()

                buffer.append(frame_keypoints)
                if len(buffer) > SEQ_LEN:
                    buffer.pop(0)

            # SWING DETECTION
            if len(buffer) > 1:
                v = get_wrist_speed(buffer[-2], buffer[-1])
                velocity_buffer.append(v)
                if len(velocity_buffer) > 30:
                    velocity_buffer.pop(0)

                #Show wrist speed for debugging
                #print(f"Frame {frame_idx}: Wrist speed = {v:.4f} | START_THRESHOLD = {START_THRESHOLD}, END_THRESHOLD = {END_THRESHOLD}")

                if v > START_THRESHOLD:
                    print(f"  -> Wrist speed ABOVE START_THRESHOLD, swing may start soon")

                if collecting_swing and v < END_THRESHOLD:
                    print(f"  -> Wrist speed BELOW END_THRESHOLD, swing may end soon")


                if cooldown > 0:
                    cooldown -= 1

                # START SWING
                if not collecting_swing and v > START_THRESHOLD and cooldown == 0:
                    collecting_swing = True
                    swing_classified = False
                    swing_frames = buffer[-min(5, len(buffer)):]  # start collecting recent frames
                    cooldown = SWING_COOLDOWN_FRAMES
                    v_peak = 0  # reset peak at start of swing
                    print(f"Swing started at frame {frame_idx}, collecting frames...")

                # COLLECT FRAMES
                if collecting_swing:
                    swing_frames.append(buffer[-1])
                    if v > v_peak:
                        v_peak = v  # update peak wrist speed during swing

                # END SWING
                if collecting_swing and v < END_THRESHOLD and not swing_classified and v_peak > PEAK_THRESHOLD:
                    collecting_swing = False

                    # Only classify if swing has enough frames
                    if len(swing_frames) >= MIN_SWING_FRAMES:
                        swing_classified = True
                        print(f"Swing ended at frame {frame_idx}, Frames collected: {len(swing_frames)}")

                        swing_frames_copy = swing_frames.copy()
                        swing_frames = []

                        # MODEL PREDICTION
                        if len(swing_frames_copy) > 0:
                            # Pad or trim frames to SEQ_LEN
                            if len(swing_frames_copy) >= SEQ_LEN:
                                last_frame = swing_frames_copy[-1]
                                while len(swing_frames_copy) < SEQ_LEN:
                                    swing_frames_copy.append(last_frame)
                            else:
                                swing_frames_copy = swing_frames_copy[-SEQ_LEN:]

                            seq = torch.tensor(np.array(swing_frames_copy, dtype=np.float32)).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = model(seq)
                                pred_class = torch.argmax(output, dim=1).item()

                            feedback = label_map_inv[pred_class]

                            # --- Update persistent feedback ---
                            feedback_text = feedback
                            feedback_timer = FEEDBACK_DURATION

                            # UI callback
                            if feedback_ui_callback:
                                import tkinter as tk
                                tk._default_root.after(0, feedback_ui_callback, feedback)

                            print(f"Swing classified as: '{feedback}'")
                            
                            print("Live Feedback:", feedback)

                        # Reset cooldown to avoid immediate new swing
                        cooldown = SWING_COOLDOWN_FRAMES


            # --- Draw persistent feedback ---
            if feedback_timer > 0:
                cv2.putText(
                    annotated_frame,
                    f"Feedback: {feedback_text}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                feedback_timer -= 1

            cv2.imshow("Live Stroke Feedback (2D)", annotated_frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def stop_feedback():
    global running
    running = False


if __name__ == "__main__":
    start_feedback()