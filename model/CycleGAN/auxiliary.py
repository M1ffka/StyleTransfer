import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
        Implements a Residual Block which performs two convolutions
        and instance normalization, then adds the input to the output.

        Attributes:
            input_channels (int): Number of channels expected in the input tensor.

        Methods:
            forward(x):
                Completes a forward pass, applying convolutions and
                activations, returning the transformed tensor with
                added residual.
    """
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class ContractingBlock(nn.Module):
    """
        Implements a Contracting Block with a convolution followed
        by a max pooling operation and an optional instance normalization.

        Attributes:
            input_channels (int): Number of channels expected in the input tensor.
            use_bn (bool): Whether to use instance normalization.
            kernel_size (int): Kernel size for the convolution.
            activation (str): Activation function to use ('relu' or 'leaky_relu').

        Methods:
            forward(x):
                Completes a forward pass, applying convolution,
                instance normalization (optional), and activation,
                returning the transformed tensor.
    """
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):
    """
       Implements an Expanding Block that uses a convolutional transpose operation to upsample
       an input tensor, with an optional instance normalization.

       Args:
           input_channels (int): Number of channels expected from a given input.
           use_bn (bool, optional): If set to True, instance normalization is applied after
                                    the convolution transpose operation. Defaults to True.

       Methods:
           forward(x):
               Performs a forward pass, applying transpose convolution, optional instance normalization,
               and ReLU activation, and returns the transformed tensor.
    """
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    """
        Implements a Feature Map Block that is typically used as the final layer of a Generator.
        It maps the output to the desired number of output channels.

        Args:
            input_channels (int): Number of channels expected from a given input.
            output_channels (int): Number of channels to be produced in the output.

        Methods:
            forward(x):
                Performs a forward pass, applying a convolution to map the input tensor
                to the desired number of output channels, and returns the transformed tensor.
    """
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        return x