"""M2 interface skeleton owned by Khoa; intentionally not registered yet.

Input: [B, 1, 28, 28]. Output: [B, num_classes] raw logits.
"""

from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # NOTE(A1): The mandatory CNN must be group-designed, not merely a pretrained backbone; keep it appropriate for Fashion-MNIST.
        # TODO(A1-M2, Khoa): Select explicit constructor/config arguments and validate input_shape, configurable num_classes and channel settings.
        # TODO(A1-M2, Khoa): Design convolution blocks and activation choice; document feature-map dimensions at each block.
        # TODO(A1-M2, Khoa): Add pooling or justified spatial reduction and calculate the resulting feature-map dimensions.
        # TODO(A1-M2, Khoa): Choose flattening or global aggregation and construct the classification head, documenting any dropout.
        raise NotImplementedError("Khoa's A1 M2 CNN classifier is pending")

    def forward(self, inputs):
        # TODO(A1-M2, Khoa): Validate image shapes and pass images through the planned blocks, aggregation and head.
        # TODO(A1-M2, Khoa): Return [B, num_classes] raw logits and verify compatibility with generic training and CrossEntropyLoss.
        # NOTE(A1): Shared loaders provide [B, 1, 28, 28] images; the shared trainer owns loss and optimization. Do not apply softmax.
        raise NotImplementedError("Khoa's A1 M2 CNN forward pass is pending")
