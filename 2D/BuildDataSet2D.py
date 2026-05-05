import numpy as np
import matplotlib.pyplot as plt
import os

# SETTINGS
data_folder = "dataset2D/forehand_front2D/good2D" #<---- MUST MANUALLY SWITCH FOLDER

# Keypoints
keypoints = ['Left Hip', 'Right Hip', 'Left Shoulder', 'Right Shoulder', 'Right Wrist', 'Right Elbow']

# Choose coordinate to visualize: "x" or "y"
COORD_MODE = "y"

# LOAD DATA
npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
all_data = {}

for file_name in npy_files:
    data = np.load(os.path.join(data_folder, file_name))
    if data.shape != (30, len(keypoints)*2):
        print(f"Skipping {file_name}, wrong shape: {data.shape}")
        continue

    # Remove starting offset (center motion)
    data = data - data[0]
    all_data[file_name] = data

# PLOT MOTION
for kp_idx, kp_name in enumerate(keypoints):
    plt.figure(figsize=(12, 6))

    for file_name, data in all_data.items():
        if COORD_MODE == "x":
            vals = data[:, kp_idx*2]
        elif COORD_MODE == "y":
            vals = data[:, kp_idx*2 + 1]
        else:
            raise ValueError("COORD_MODE must be 'x' or 'y'")

        plt.plot(vals, label=file_name)

    plt.title(f"{kp_name} Motion ({COORD_MODE.upper()})")
    plt.xlabel("Timestep")
    plt.ylabel("Centered Coordinate Value")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    plt.subplots_adjust(right=0.75)
    plt.show()