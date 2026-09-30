"""M2 interface skeleton owned by Antoine; intentionally not registered yet.

Conceptual flow: image -> tokens -> projection -> positional encoding
-> Transformer encoder -> aggregation -> classifier -> logits.
Input: [B, 1, 28, 28]. Output: [B, num_classes] raw logits.
"""

from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # TODO(A1-M2, Antoine): Select explicit constructor/config arguments; validate image dimensions, num_classes and attention dimension/head compatibility.
        # TODO(A1-M2, Antoine): Define row, column or patch tokens, sequence length and embedding dimension; construct token projection.
        # TODO(A1-M2, Antoine): Add positional encoding and a Transformer encoder/self-attention stack with documented attention input/output shapes.
        # TODO(A1-M2, Antoine): Choose sequence aggregation and construct a classifier for its resulting representation.
        # NOTE(A1): Implement the mandatory classifier within the project; do not substitute a pretrained Vision Transformer.
        raise NotImplementedError("Antoine's A1 M2 Transformer classifier is pending")

    def forward(self, inputs):
        # TODO(A1-M2, Antoine): Validate [B, 1, 28, 28] input and convert images to the documented token sequence with checked shapes.
        # TODO(A1-M2, Antoine): Apply projection, positions, encoder and aggregation; return [B, num_classes] raw logits through the head.
        # TODO(A1-M2, Antoine): Verify forward/backward compatibility with shared image loaders, generic trainer and CrossEntropyLoss without softmax.
        raise NotImplementedError("Antoine's A1 M2 Transformer forward pass is pending")
