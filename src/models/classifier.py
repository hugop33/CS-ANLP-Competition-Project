import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim=12):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x