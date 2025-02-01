import torch
from torch import nn

checkpoint_path = 'YOUR_PATH'
checkpoint = torch.load(checkpoint_path)

gen_AB.load_state_dict(pre_dict['gen_AB'])
gen_BA.load_state_dict(pre_dict['gen_BA'])
gen_opt.load_state_dict(pre_dict['gen_opt'])
disc_A.load_state_dict(pre_dict['disc_A'])
disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
disc_B.load_state_dict(pre_dict['disc_B'])
disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])