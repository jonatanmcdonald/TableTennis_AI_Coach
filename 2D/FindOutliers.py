import numpy as np
import os

# Change this to whichever class you want to inspect
data_folder = "dataset2D/forehand_front2D/elbow2D"  # Change to "good2D", "low2D", "norot2D"

# Load files
files = [f for f in os.listdir(data_folder) if f.endswith(".npy")]

strokes = []
file_names = []

for f in files:
    data = np.load(os.path.join(data_folder, f))
    
    # Ensure correct shape
    if data.shape != (30, 12):
        print(f"Skipping {f}, wrong shape: {data.shape}")
        continue
    
    strokes.append(data.flatten())
    file_names.append(f)

strokes = np.array(strokes)

# Compute mean stroke
mean_stroke = np.mean(strokes, axis=0)

# Compute distances
distances = np.linalg.norm(strokes - mean_stroke, axis=1)

# Normalize (z-score)
distances = (distances - np.mean(distances)) / np.std(distances)

# Sort from worst → best
sorted_idx = np.argsort(distances)[::-1]

print("\n=== OUTLIER SCORES (highest = most suspicious) ===\n")

for i in sorted_idx:
    print(f"{file_names[i]:30s}  score: {distances[i]:.2f}")