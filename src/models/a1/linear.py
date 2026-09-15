from math import prod
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(prod(input_shape), num_classes))

    def forward(self, inputs):
        return self.layers(inputs)
