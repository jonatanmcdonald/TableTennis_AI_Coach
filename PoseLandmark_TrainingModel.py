import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmark,
    PoseLandmarksConnections,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)
from mediapipe.tasks.python import BaseOptions


video_path = "./Videos/Senura_Side.mp4"
output_video_path = "dataset/good/output_video.mp4"

# keep only landmarks we care about
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

model_path = './pose_landmarker.task'

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    
)

with PoseLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video")
        exit()

    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        exit()

    

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((frame_idx / fps) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            pose = result.pose_landmarks[0]

            # Draw landmarks
            for idx in keep_indices:
                landmark = pose[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)


            # Draw connections
            for connection in PoseLandmarksConnections.POSE_LANDMARKS:
                if connection.start in keep_indices and connection.end in keep_indices:
                    start = pose[connection.start]
                    end = pose[connection.end]
                    x1, y1 = int(start.x * width), int(start.y * height)
                    x2, y2 = int(end.x * width), int(end.y * height)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Write frame to output video
        out.write(frame)


        preview = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))
        # Show frame live (optional)
        cv2.imshow("Pose Preview", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_video_path}")
