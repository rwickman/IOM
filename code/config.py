import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inv_scale_factor = 1000
inv_locs_scale_factor = 100