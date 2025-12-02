import yaml
import torch
import numpy as np

from .restormer import Restormer


def get_model(opt_path: str, device: torch.device):
    opt = yaml.load(open(opt_path, mode='r'), Loader=yaml.Loader)
    opt['network_g'].pop('type')

    model = Restormer(**opt['network_g'])
    weights_path = opt['path']['pretrain_network_g']
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()

    print(f"Successfully loaded {np.sum([p.numel() for p in model.parameters()]):,} parameters from {weights_path}")
    return model
