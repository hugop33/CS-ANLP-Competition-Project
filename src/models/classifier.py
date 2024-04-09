import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_features, *hidden_dims, output_dim=12):
        super(Classifier, self).__init__()
        dims = [input_features] + list(hidden_dims)
        self.fc = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])
        self.fout = nn.Linear(dims[-1], output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            x = self.relus[i](x)
        x = self.softmax(x)
        return x