import torch
from torch import nn

adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

n_epochs = 40
dim_A = 3
dim_B = 3
display_step = 1000
batch_size = 1
lr = 0.0002
load_shape = 128
target_shape = 128
device = 'cuda'