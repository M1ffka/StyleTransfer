import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
        Displays images from a tensor.

        Parameters:
        image_tensor (torch.Tensor): A tensor containing the images.
        num_images (int): Number of images to display (default is 25).
        size (tuple): Dimensions of the images (default is (1, 28, 28)).
    """
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()