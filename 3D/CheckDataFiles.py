import os
import numpy as np

DATASET_PATH = "dataset/forehand_front"

for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)
    
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            
            non_zero_frames = np.sum(np.any(data != 0, axis=1))
            zero_frames = data.shape[0] - non_zero_frames

            print(f"{file}: real={non_zero_frames}, padded={zero_frames}")
            
            #print(f"{file}: min={data.min():.2f}, max={data.max():.2f}")

            #if np.allclose(data[0], data[-1]):
            #   print(f"⚠️ No movement in {file}")

            #if np.all(data == 0):
            #    print(f"❌ All zeros in {file}")

            #if np.isnan(data).any():
            #    print(f"❌ NaNs in {file}")
            
            #if np.isinf(data).any():
            #    print(f"❌ Inf values in {file}")
            #if data.shape != (30, 18):
            #    print(f"❌ Bad shape in {file}: {data.shape}")
            #else:
            #    print(f"✅ {file}: {data.shape}")
            