"""Linear baseline classifier for Assignment 1 Fashion-MNIST experiments.

The model flattens ``[batch, channels, height, width]`` image tensors, applies
one affine transformation, and returns ``[batch, classes]`` raw logits.
"""

from math import prod
from typing import Sequence

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """Classify flattened images with one affine transformation.

    The model returns raw logits. Softmax is deliberately omitted because the
    training pipeline uses :class:`torch.nn.CrossEntropyLoss`, which applies
    log-softmax internally in a numerically stable way.

    Args:
        input_shape: Shape of one input sample, excluding the batch dimension.
        num_classes: Number of target classes.
    """

    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        super().__init__()

        shape = tuple(input_shape)
        if not shape:
            raise ValueError("input_shape must contain at least one dimension")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
            for dimension in shape
        ):
            raise ValueError("input_shape dimensions must be positive integers")
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 2
        ):
            raise ValueError("num_classes must be an integer greater than one")

        self.input_shape = shape
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(prod(shape), num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for a batch of images."""
        return self.layers(inputs)
