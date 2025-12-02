import numpy as np
import torch

from .models.network_dncnn import DnCNN


def get_model(weights_path: str, n_channels: int, nb: int, device: torch.device):
    model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    # model = DnCNN(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    model.load_state_dict(torch.load(weights_path), strict=True)
    model.eval()
    model.to(device)

    print(f"Successfully loaded {np.sum([p.numel() for p in model.parameters()]):,} parameters from {weights_path}")
    return model
