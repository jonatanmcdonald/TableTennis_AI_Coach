import numpy as np
import matplotlib.pyplot as plt
import os

# Folder for one category
data_folder = "dataset2D/forehand_front2D/good2D"  # Change to "elbow2D", "low2D", "norot2D"

# List all .npy files
npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]

# Keypoints
keypoints = ['Left Hip', 'Right Hip', 'Left Shoulder', 'Right Shoulder', 'Right Wrist', 'Right Elbow']

# Loop over each keypoint
for kp_idx, kp_name in enumerate(keypoints):
    plt.figure(figsize=(12, 6))
    
    for file_name in npy_files:
        data = np.load(os.path.join(data_folder, file_name))
        timesteps, features = data.shape
        num_keypoints = len(keypoints)
        
        # Check if dimensions match (expecting 12 for 2D: 6 keypoints × 2 coords)
        if features != num_keypoints * 2:
            print(f"Warning: {file_name} has {features} features, expected {num_keypoints*2}")
            continue
        
        # Extract x and y for this keypoint
        x_vals = data[:, kp_idx*2]      # x coordinate
        y_vals = data[:, kp_idx*2 + 1]  # y coordinate
        
        # Plot x and y separately
        plt.plot(x_vals, label=f"{file_name} x", linestyle='solid')
        plt.plot(y_vals, label=f"{file_name} y", linestyle='none')
        
    
    plt.title(f"{kp_name} Coordinates for Category: {os.path.basename(data_folder)}")
    plt.xlabel("Timestep")
    plt.ylabel("Coordinate Value")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    plt.subplots_adjust(right=0.75)
    plt.show()