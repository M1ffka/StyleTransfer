import torch
from torch import nn

class Generator(nn.Module):
  """
      *Generator Class*

      Implements a generator to transform an input image into an image from another class.
      Consists of a series of 2 contracting blocks, 9 residual blocks, and 2 expanding blocks,
      with an `upfeature` layer at the start and a `downfeature` layer at the end.

      *Attributes*:
      - **input_channels** (*int*): Number of channels expected in the input.
      - **output_channels** (*int*): Number of channels expected in the output.
      - **hidden_channels** (*int*, optional): Number of hidden channels (default: 64).

      *Methods*:
      - **forward(x)**: Performs a forward pass through the generator.
        Parameters:
          - *x* (*torch.Tensor*): Input image as a tensor with shape (batch size, channels, height, width).
        Returns:
          - *torch.Tensor*: Transformed image.
  """
  def __init__(self, input_channels, output_channels, hidden_channels = 64):
    super(Generator, self).__init__()
    self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
    self.contract1 = ContractingBlock(hidden_channels)
    self.contract2 = ContractingBlock(hidden_channels * 2)
    res_mult = 4
    self.res0 = ResidualBlock(hidden_channels * res_mult)
    self.res1 = ResidualBlock(hidden_channels * res_mult)
    self.res2 = ResidualBlock(hidden_channels * res_mult)
    self.res3 = ResidualBlock(hidden_channels * res_mult)
    self.res4 = ResidualBlock(hidden_channels * res_mult)
    self.res5 = ResidualBlock(hidden_channels * res_mult)
    self.res6 = ResidualBlock(hidden_channels * res_mult)
    self.res7 = ResidualBlock(hidden_channels * res_mult)
    self.res8 = ResidualBlock(hidden_channels * res_mult)
    self.expand2 = ExpandingBlock(hidden_channels * 4)
    self.expand3 = ExpandingBlock(hidden_channels * 2)
    self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
    self.tanh = torch.nn.Tanh()
  def forward(self, x):
    x0 = self.upfeature(x)
    x1 = self.contract1(x0)
    x2 = self.contract2(x1)
    x3 = self.res0(x2)
    x4 = self.res1(x3)
    x5 = self.res2(x4)
    x6 = self.res3(x5)
    x7 = self.res4(x6)
    x8 = self.res5(x7)
    x9 = self.res6(x8)
    x10 = self.res7(x9)
    x11 = self.res8(x10)
    x12 = self.expand2(x11)
    x13 = self.expand3(x12)
    xn = self.downfeature(x13)
    return self.tanh(xn)