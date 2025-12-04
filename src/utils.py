import cv2
import numpy as np
from torch.nn import Module

from deblurganv2 import Predictor


def get_model_total_parameters(model: Module | Predictor) -> int:
    if isinstance(model, Predictor):
        # DeblurGANv2 Predictor
        return sum(p.numel() for p in model.model.parameters())
    else:
        # Standard PyTorch model
        return sum(p.numel() for p in model.parameters())


def add_gaussian_noise(img: np.ndarray, sigma=15):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.

    np.random.seed(seed=0)  # for reproducibility
    img += np.random.normal(0, sigma/255., img.shape)
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)


def imread_uint8(file_path: str, n_channels=3):
    if n_channels == 1:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img, dtype=np.uint8)
        return np.expand_dims(img, axis=2)

    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(img, dtype=np.uint8)


def imread_uint16(file_path: str):
    img = cv2.imread(file_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(img, dtype=np.uint16)


def imwrite_uint(file_path: str, img: np.ndarray):
    cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
