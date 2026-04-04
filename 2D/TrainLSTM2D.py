import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from TrainModel2D import Forehand_Front_2D_Dataset  # <- Use the 2D dataset

# Define LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=4):  # input_size=12 for 2D
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # output: [batch, seq, features]
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, features]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]              # Take last timestep
        out = self.fc(out)               # [batch, num_classes]
        return out

# Load Dataset and DataLoader
dataset = Forehand_Front_2D_Dataset(root_dir='dataset2D/forehand_front2D')

all_labels = [dataset[i][1] for i in range(len(dataset))]
all_labels = torch.tensor(all_labels).numpy()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(split.split(torch.zeros(len(all_labels)), all_labels))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

print("Train label counts:", Counter([all_labels[i] for i in train_idx]))
print("Validation label counts:", Counter([all_labels[i] for i in val_idx]))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    val_labels_in_batch = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            val_labels_in_batch.extend([int(label) for label in y.cpu()])

            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)

            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    label_counts = Counter(val_labels_in_batch)
    print("Validation label counts:", label_counts)

    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Train Acc: {epoch_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), "2D/lstm2D/lstm_forehand_front_2d_40.pth")
    print("Model saved as lstm_forehand_front_2d_40.pth")