import torch
from torch.utils.data import DataLoader
from TrainModel import Forehand_Front_Dataset

#Load the dataset
dataset = Forehand_Front_Dataset(root_dir='dataset/forehand_front')

#Create a DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

#Test
for X, y in loader:
    print("X shape:", X.shape) #[batch_size, frames, features] 
    print("y:", y)             #tensor
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
X, y = X.to(device), y.to(device)