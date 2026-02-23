import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TrainModel import Forehand_Front_Dataset

#Define LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=4):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True #output [batch, seq, features]
        )

        #Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, features]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) #out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]             #Take last timestep
        out = self.fc(out)              #[batch, num_classes]
        return out
    
#Load Dataset and DataLoader
dataset = Forehand_Front_Dataset(root_dir='dataset/forehand_front')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

#Initialize model, loss, optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training loop
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X,y = X.to(device), y.to(device)

        #Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Statistics
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")


    #Save the model
    torch.save(model.state_dict(), "lstm_forehand_front.pth")
    print("Model saved as lstm_forehand_front.pth")