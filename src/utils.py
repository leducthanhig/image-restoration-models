import os
import time

import cv2
import numpy as np
import torch
from torch.nn import Module
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from deblurganv2 import Predictor
from restormer import Restormer
from mair.realDenoising.basicsr.models.archs.mairunet_arch import MaIRUNet
from configs import ROOT_RESULTS_DIR


def get_model_total_parameters(model: Module | Predictor) -> int:
    if isinstance(model, Predictor):
        # DeblurGANv2 Predictor
        return sum(p.numel() for p in model.model.parameters())
    else:
        # Standard PyTorch model
        return sum(p.numel() for p in model.parameters())


def add_gaussian_noise(img: np.ndarray, sigma: int | float = 15):
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


def find_max_patch_size(
    model: torch.nn.Module | Predictor,
    device: torch.device,
    channels: int = 3,
    max_side: int = 2048,
    step: int = 16,
):
    """
    Heuristically find the maximum square patch size that a model (PyTorch nn.Module)
    can process without CUDA OOM using binary search.
    - max_side: upper bound for the search (pixels).
    - step: quantization step for result (e.g., 16).
    Returns an int patch_size (<= max_side).
    """
    if not torch.cuda.is_available():
        return None

    # Limit candidates to max_side
    max_side = int(max_side)
    lo, hi = step, max_side

    @torch.no_grad()
    def try_forward(sz):
        torch.cuda.empty_cache()
        # small dummy input
        x = np.random.randint(0, 255, (sz, sz, channels), dtype=np.uint8)
        try:
            _ = run_model_inference(model, x, device)
            # synchronize to catch async OOMs
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'CUDA out of memory' in str(e):
                torch.cuda.empty_cache()
                return False
            # propagate other errors
            raise

    # Binary search
    best = step
    while lo <= hi:
        mid = ((lo + hi) // (2 * step)) * step
        if mid < step:
            mid = step
        try:
            ok = try_forward(mid)
        except Exception as e:
            # if model raises other exceptions (e.g. wrong input shape), stop trying.
            print(f"Exception during try_forward with size {mid}: {e}")
            return None
        if ok:
            best = mid
            lo = mid + step
        else:
            hi = mid - step

    return best


def get_result_save_dir(test_name: str, dataset_name: str, model_name: str) -> str:
    """Get the directory path for saving results."""
    dir_path = os.path.join(ROOT_RESULTS_DIR, test_name, dataset_name, model_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_result_image(pred: np.ndarray, test_name: str, dataset_name: str, model_name: str, img_name: str) -> str:
    """Save prediction image to results directory."""
    dir_path = get_result_save_dir(test_name, dataset_name, model_name)
    file_path = os.path.join(dir_path, img_name)
    imwrite_uint(file_path, pred)
    return file_path


def calculate_metrics(pred: np.ndarray, target: np.ndarray, data_range: int | float | None = None):
    """Calculate PSNR and SSIM metrics between prediction and target."""
    # Determine data range based on dtype
    if data_range is None:
        if pred.dtype == np.uint8:
            data_range = 255
        elif pred.dtype == np.uint16:
            data_range = 65535
        else:
            data_range = 1.0

    # Calculate PSNR
    psnr_value = psnr(target, pred, data_range=data_range)

    # Calculate SSIM
    if pred.ndim == 3 and pred.shape[2] == 3:  # Color image
        ssim_value = ssim(target, pred, data_range=data_range, channel_axis=2)
    elif pred.ndim == 3 and pred.shape[2] == 1:  # Grayscale with channel dim
        ssim_value = ssim(target[:, :, 0], pred[:, :, 0], data_range=data_range)
    else:  # Grayscale without channel dim
        ssim_value = ssim(target, pred, data_range=data_range)

    return psnr_value, ssim_value


def run_model_inference(
    model: Module | Predictor,
    input_img: np.ndarray,
    device: torch.device,
    patch_size: int | None = None,
    patch_overlap: int = 32,
    need_degradation=False,
    noise_level: int | float | None = None,
):
    """Run inference based on model type. Returns (prediction, inference_time_ms)."""
    start_time = time.time()

    with torch.no_grad():
        # Clear GPU cache (if available)
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.empty_cache()

        h, w = input_img.shape[:2]
        if patch_size:
            patch_size = min(patch_size, h, w)
            stride = patch_size - patch_overlap
            h_idx_list = list(range(0, h-patch_size, stride)) + [h-patch_size]
            w_idx_list = list(range(0, w-patch_size, stride)) + [w-patch_size]
        else:
            patch_size = max(h, w)
            h_idx_list = [0]
            w_idx_list = [0]

        output_img = np.zeros(shape=(h, w, min(3, input_img.shape[2])), dtype=input_img.dtype)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                input_patch = input_img[h_idx:h_idx+patch_size, w_idx:w_idx+patch_size, :]

                if isinstance(model, Predictor):
                    # DeblurGANv2 uses its own Predictor class. It returns uint8 [0,255]
                    pred: np.ndarray = model(input_patch)
                else:
                    # Standard PyTorch models

                    # Normalize to [0,1]
                    if input_patch.dtype == np.uint8:
                        img_normed = input_patch.astype(np.float32) / 255.0
                    elif input_patch.dtype == np.uint16:
                        img_normed = input_patch.astype(np.float32) / 65535.0

                    # Add noise if required
                    if need_degradation and noise_level is not None:
                        img_normed = add_gaussian_noise(img_normed, noise_level)

                    # Convert to tensor: (H, W, C) -> (1, C, H, W)
                    input_tensor = torch.from_numpy(img_normed.transpose(2, 0, 1)).unsqueeze(0).to(device)

                    if isinstance(model, Restormer) or isinstance(model, MaIRUNet):
                        # Padding in case images are not multiples of 8
                        h,w = input_tensor.shape[2], input_tensor.shape[3]
                        factor = 8
                        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                        padh = H-h if h%factor!=0 else 0
                        padw = W-w if w%factor!=0 else 0
                        input_tensor = torch.nn.functional.pad(input_tensor, (0,padw,0,padh), 'reflect')
                        output_tensor: torch.Tensor = model(input_tensor)[:, :, :h, :w]
                    else:
                        output_tensor: torch.Tensor = model(input_tensor)

                    # Convert back to numpy: (1, C, H, W) -> (H, W, C)
                    pred = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    # Rescale to uint
                    if input_patch.dtype == np.uint8:
                        pred = np.clip(pred * 255.0, 0, 255).round().astype(np.uint8)
                    elif input_patch.dtype == np.uint16:
                        pred = np.clip(pred * 65535.0, 0, 65535).round().astype(np.uint16)

                output_img[h_idx:h_idx+patch_size, w_idx:w_idx+patch_size, :] = pred

    inference_time_ms = (time.time() - start_time) * 1000
    return output_img, inference_time_ms
