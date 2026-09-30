"""M2 interface skeleton owned by Kim; intentionally not registered yet.

Input: [B, 1, 28, 28]. Conceptual row sequence: [B, 28, 28].
One timestep is one image row; sequence length = 28, input size = 28.
Output: [B, num_classes] raw logits (10 classes for Fashion-MNIST).
"""

from torch import nn


class RNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # TODO(A1-M2, Kim): Select explicit constructor/config arguments and validate input_shape, num_classes and recurrent settings.
        # TODO(A1-M2, Kim): Construct one LSTM OR GRU and document hidden size, layer count and any dropout constraints.
        # TODO(A1-M2, Kim): Define the final/aggregated hidden representation and construct its classification head.
        raise NotImplementedError("Kim's A1 M2 recurrent classifier is pending")

    def forward(self, inputs):
        # TODO(A1-M2, Kim): Validate [B, 1, 28, 28] images and convert to [B, 28, 28] row sequences; document timestep meaning.
        # TODO(A1-M2, Kim): Run the recurrence and extract the documented final/aggregated hidden representation.
        # TODO(A1-M2, Kim): Return [B, num_classes] raw logits and verify forward/backward with the shared trainer and CrossEntropyLoss.
        # NOTE(A1): Sequence conversion belongs here; shared loaders and trainer still supply image batches. Do not apply softmax.
        raise NotImplementedError("Kim's A1 M2 recurrent forward pass is pending")
