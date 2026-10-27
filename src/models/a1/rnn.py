"""Recurrent Fashion-MNIST classifier over image rows.

An image shaped ``[B, 1, 28, 28]`` becomes a row sequence ``[B, 28, 28]``:
each timestep contains one row and the RNN input size is the row width.
"""

from math import isfinite
from typing import Sequence

import torch
from torch import nn


class RNNClassifier(nn.Module):
    """Classify images using the final hidden state of an LSTM or GRU."""

    def __init__(
        self,
        input_shape: Sequence[int],
        num_classes: int,
        recurrent_type: str = "lstm",
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        shape = tuple(input_shape)
        if (
            len(shape) != 3
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 1
                for dimension in shape
            )
            or shape[0] != 1
        ):
            raise ValueError(
                "input_shape must be [1, height, width] with positive dimensions"
            )
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 2
        ):
            raise ValueError("num_classes must be an integer greater than one")
        if (
            isinstance(input_size, bool)
            or not isinstance(input_size, int)
            or input_size < 1
            or shape[2] != input_size
        ):
            raise ValueError("input_size must be a positive integer matching image row width")
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size < 1
        ):
            raise ValueError("hidden_size must be a positive integer")
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers < 1
        ):
            raise ValueError("num_layers must be a positive integer")
        if (
            not isinstance(recurrent_type, str)
            or recurrent_type.lower() not in ("lstm", "gru")
        ):
            raise ValueError("recurrent_type must be 'lstm' or 'gru'")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if num_layers == 1 and dropout != 0:
            raise ValueError("dropout must be zero when num_layers is one")

        self.input_shape = shape
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.recurrent_type = recurrent_type.lower()

        recurrent_class = nn.LSTM if self.recurrent_type == "lstm" else nn.GRU
        self.recurrent = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw class logits from the final recurrent layer's hidden state."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != self.input_shape:
            expected_shape = ", ".join(map(str, self.input_shape))
            raise ValueError(f"inputs must have shape [batch, {expected_shape}]")
        if inputs.shape[0] == 0:
            raise ValueError("inputs must contain at least one image")

        # Indexing the single channel preserves the batch axis, including for B=1.
        sequence = inputs[:, 0, :, :]
        if isinstance(self.recurrent, nn.LSTM):
            _, (hidden_state, _) = self.recurrent(sequence)
        else:
            _, hidden_state = self.recurrent(sequence)

        return self.classifier(hidden_state[-1])
