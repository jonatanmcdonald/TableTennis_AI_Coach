import torch
from torch.utils.data import DataLoader
from TrainModel2D import Forehand_Front_2D_Dataset  # <- Use 2D dataset

# Load the dataset
dataset = Forehand_Front_2D_Dataset(root_dir='dataset2D/forehand_front2D')

# Create a DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test
for X, y in loader:
    print("X shape:", X.shape)  # [batch_size, frames, features] -> features = 12 now
    print("y:", y)              # tensor
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
X, y = X.to(device), y.to(device)