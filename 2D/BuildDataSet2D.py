import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmark,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)

# Paths
INPUT_ROOT = "Videos/Strokes"
OUTPUT_ROOT = "dataset2D/forehand_front2D"

# Map input folders to class names
label_map = {
    "Good": "good2D",
    "Elbow": "elbow2D",
    "Low": "low2D",
    "NoRot": "norot2D"
}

# Same joints
keep_indices = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
]

# Model setup
model_path = './pose_landmarker.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)

def normalize_frame(keypoints):
    """
    EXACT same logic as 3D version, just 2D
    """
    keypoints = np.array(keypoints).reshape(-1, 2)  # (x, y)

    # Hip midpoint
    left_hip = keypoints[keep_indices.index(PoseLandmark.LEFT_HIP)]
    right_hip = keypoints[keep_indices.index(PoseLandmark.RIGHT_HIP)]
    hip_mid = (left_hip + right_hip) / 2

    # Shoulder midpoint
    left_shoulder = keypoints[keep_indices.index(PoseLandmark.LEFT_SHOULDER)]
    right_shoulder = keypoints[keep_indices.index(PoseLandmark.RIGHT_SHOULDER)]
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    # Torso length
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_length == 0:
        torso_length = 1.0

    # Normalize
    keypoints_norm = (keypoints - hip_mid) / torso_length

    return keypoints_norm.flatten().tolist()  # 12 features


# EXACT SAME STRUCTURE AS YOUR 3D VERSION
with PoseLandmarker.create_from_options(options) as landmarker:

    for input_label in label_map:
        input_folder = os.path.join(INPUT_ROOT, input_label)
        output_folder = os.path.join(OUTPUT_ROOT, label_map[input_label])
        os.makedirs(output_folder, exist_ok=True)

        for subfile in sorted(os.listdir(input_folder), key=lambda x: int(''.join(filter(str.isdigit, x)))):
            video_path = os.path.join(input_folder, subfile)

            if not video_path.lower().endswith((".mp4", ".mov")):
                continue

            print(f"Processing: {video_path}")

            # IMPORTANT: reset landmarker per video (same as your working 3D)
            with PoseLandmarker.create_from_options(options) as landmarker:

                cap = cv2.VideoCapture(video_path)
                stroke_data = []

                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                if not fps or fps != fps:
                    fps = 30

                frame_interval = int(1000 / fps)
                prev_timestamp = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    timestamp_ms = int(prev_timestamp)
                    prev_timestamp += frame_interval

                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if result.pose_landmarks:
                        pose = result.pose_landmarks[0]
                        frame_keypoints = []

                        for idx in keep_indices:
                            landmark = pose[idx]
                            frame_keypoints.extend([landmark.x, landmark.y])  # 🔥 ONLY x,y

                        # Normalize
                        frame_keypoints = normalize_frame(frame_keypoints)
                        stroke_data.append(frame_keypoints)

                cap.release()

                if len(stroke_data) == 0:
                    print(f"No keypoints detected in {video_path}")
                    continue

                # Pad/trim EXACT same way
                stroke_array = np.array(stroke_data)

                if len(stroke_array) > 30:
                    stroke_array = stroke_array[:30]
                elif len(stroke_array) < 30:
                    pad = np.zeros((30 - len(stroke_array), stroke_array.shape[1]))
                    stroke_array = np.vstack((stroke_array, pad))

                save_name = os.path.splitext(subfile)[0] + ".npy"
                np.save(os.path.join(output_folder, save_name), stroke_array)

                print(f"Saved: {save_name}")

cv2.destroyAllWindows()