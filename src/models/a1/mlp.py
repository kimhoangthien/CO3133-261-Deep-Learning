from torch import nn
from math import prod

class MLP(nn.Module):
    def __init__(self, input_shape, hidden_size, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(input_shape), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
            )
    
    def forward(self, inputs):
        return self.layers(inputs)