import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import kagglehub
import torchvision
from torchvision import transforms
import torch

class ImageDataset(Dataset):

    def __init__(self, root, transform=None, mode="train"):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, f'{mode}A') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, f'{mode}B') + '/*.*'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))

        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)

        if index == len(self) - 1:
            self.new_perm()

        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


path = kagglehub.dataset_download("balraj98/monet2photo")

load_shape = 128
target_shape = 128

transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset = ImageDataset(path, transform=transform)