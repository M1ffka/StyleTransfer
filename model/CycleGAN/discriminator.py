import torch
from torch import nn

class Discriminator(nn.Module):
    """
        *Discriminator Class*

        Implements a discriminator that evaluates the authenticity of an input image.
        Consists of an `upfeature` layer, followed by a series of 3 contracting blocks,
        and concludes with a final convolutional layer.

        *Attributes*:
        - **input_channels** (*int*): Number of channels expected in the input.
        - **hidden_channels** (*int*, optional): Number of hidden channels (default: 64).

        *Methods*:
        - **forward(x)**: Performs a forward pass through the discriminator.
          Parameters:
            - *x* (*torch.Tensor*): Input image as a tensor with shape (batch size, channels, height, width).
          Returns:
            - *torch.Tensor*: Output tensor indicating the authenticity evaluation.
    """
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn