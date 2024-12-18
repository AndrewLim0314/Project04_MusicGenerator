import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MusicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Match saved fc1
        self.fc2 = nn.Linear(128, 64)        # Match saved fc2
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))    # First layer
        x = self.dropout(F.relu(self.fc2(x)))  # Second layer with dropout
        output = self.fc3(x)       # Output layer
        return output



