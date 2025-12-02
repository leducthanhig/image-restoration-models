import numpy as np
import torch

from .rednet import REDNet


def get_model(weights_path: str, device: torch.device):
    """Load REDNet model with converted weights.

    Args:
        param: Model parameter (sigma for denoising, kernel type for deblurring)
        task: Task name ('denoising', 'deblurring', 'inpainting')
        model_root: Root directory containing model weights
        device: Device to load model on

    Returns:
        Loaded REDNet model in eval mode
    """
    model = REDNet()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Successfully loaded {np.sum([p.numel() for p in model.parameters()]):,} parameters from {weights_path}")
    return model
