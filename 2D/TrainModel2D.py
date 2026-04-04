import torch
from torch.utils.data import Dataset
import numpy as np
import os

class Forehand_Front_2D_Dataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: root folder containing subfolders
        """
        self.samples = []
        self.labels = []
        # Map folder names to numeric labels
        self.label_map = {
            "good2D": 0,
            "elbow2D": 1,
            "low2D": 2,
            "norot2D": 3
        }

        for label_name, label_idx in self.label_map.items():
            folder = os.path.join(root_dir, label_name)
            for file in os.listdir(folder):
                if file.endswith('.npy'):
                    self.samples.append(os.path.join(folder, file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load .npy file
        data_2d = np.load(self.samples[idx]).astype(np.float32)  # shape: (timesteps, 12)
        
       
        label = self.labels[idx]
        # Convert to PyTorch tensors
        return torch.tensor(data_2d), torch.tensor(label, dtype=torch.long)