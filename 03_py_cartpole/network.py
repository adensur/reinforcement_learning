import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIMS = 4
OUTPUT_DIMS = 2


class Network(nn.Module):
    def __init__(self, hidden_size=32, layers=2):
        super(Network, self).__init__()
        self.layers = []
        self.input_layer = nn.Linear(INPUT_DIMS, hidden_size)
        for i in range(layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.final_layer = nn.Linear(hidden_size, OUTPUT_DIMS)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.final_layer(x)
