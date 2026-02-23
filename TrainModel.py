import torch
from torch.utils.data import Dataset
import numpy as np
import os

class Forehand_Front_Dataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dor: root folder containing subfolders
        """
        self.samples = []
        self.labels = []
        #Map folder names to numeric labels
        self.label_map = {
            "good": 0,
            "elbow": 1,
            "low": 2,
            "norot":3
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
        #Load .npy file
        data = np.load(self.samples[idx]).astype(np.float32) #shape: (30, 12)
        label = self.labels[idx]
        #Convert to Pytorch tensors
        return torch.tensor(data), torch.tensor(label, dtype=torch.long)
    