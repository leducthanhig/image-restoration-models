import os

import numpy as np
import torch

from .aug import get_normalize
from .models.networks import get_generator
from .config import config


def normalize(x: np.ndarray):
    x, _ = get_normalize()(x, x)
    return x


def pad(x: torch.Tensor):
    h, w = x.shape[-2:]
    block_size = 32
    min_height = (h // block_size + 1) * block_size
    min_width = (w // block_size + 1) * block_size

    x = torch.nn.functional.pad(x, (0, min_width - w, 0, min_height - h), 'constant', 0)

    return x


def postprocess(x: torch.Tensor):
    return (x + 1) / 2.0


def get_model(weights_path: str, device: torch.device):
    model_name = os.path.basename(weights_path).split('.')[0]
    config['model']['g_name'] = model_name

    model = get_generator(config['model'])
    model.load_state_dict(torch.load(weights_path)['model'])
    model.to(device)
    model.train(True)

    print(f"Successfully loaded {np.sum([p.numel() for p in model.parameters()]):,} parameters from {weights_path}")
    return model.module
