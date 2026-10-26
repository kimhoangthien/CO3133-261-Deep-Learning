"""M2 interface skeleton owned by Khoa; intentionally not registered yet.

Input: [B, 1, 28, 28]. Output: [B, num_classes] raw logits.
"""

from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, base_channels=32, fc_size=128, dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=base_channels, kernel_size=3, stride=1, padding=1) # [B, 1, 28, 28] -> [B, 32, 28, 28]
        self.relu1 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # [B, 32, 28, 28] -> [B, 32, 14, 14]
        
        self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1) # [B, 32, 14, 14] -> [B, 64, 14, 14]
        self.relu2 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # [B, 64, 14, 14] -> [B, 64, 7, 7]
        
        self.flatten = nn.Flatten()
        
        flattened_size = base_channels * 2 * 7 * 7
        
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, num_classes)
        )
        
        

    def forward(self, inputs):
        
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)     # [B, 3136]
        logits = self.classifier(x) # [B, num_classes]
        
        return logits
