import torch
import torch.nn as nn
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        assert len(eeg_data) == len(labels), f"Mismatch: {len(eeg_data)} EEG samples vs {len(labels)} labels"
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.eeg_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            # self.eeg_data[idx].clone().detach(),  # Recommended copy method
            # self.labels[idx].clone().detach()
        )
# -----------------------------
# Neural Network Models
# -----------------------------
# 1 convolutional layer

class Rayanet(nn.Module):
    def __init__(self, in_channels=1):  # Change in_channels if your images have a different number of channels
        super(Rayanet, self).__init__()
        self.features = nn.Sequential(
            # Input: (in_channels) x 32 x 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  
            nn.Dropout(0.1)
        )
        
        # Classifier: flatten the features and pass through fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(16*16*32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x  # raw logits

# 3 convolutional layer
class EEGNet2(nn.Module):
    def __init__(self, in_channels=3):  
        super(EEGNet2, self).__init__()

        self.features = nn.Sequential(
            # Block 1: Output size = 32 x 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16 x 16
            nn.Dropout(0.2),

            # Block 2: Output size = 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 8 x 8
            nn.Dropout(0.3),

            # Block 3: Output size = 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=2, stride=2),  # Output: 4 x 4
            nn.Dropout(0.4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension if needed (for grayscale input)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x  # Raw logits

# -----------------------------
# Training and Evaluation
# -----------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to class 0/1
        correct += (predictions == labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to class 0/1
            correct += (predictions == labels).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)