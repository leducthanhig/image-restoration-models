import os
import time
from typing import Callable

import cv2
import numpy as np
import torch
from torch.nn import Module
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from configs import ROOT_RESULTS_DIR


def get_model_total_parameters(model: Module) -> int:
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
    model: torch.nn.Module,
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


def normalize(img: np.ndarray):
    if img.dtype == np.uint8:
        img_normed = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img_normed = img.astype(np.float32) / 65535.0

    return img_normed.astype(np.float32)


def pad(x: torch.Tensor, downscale_factor: int = 8):
    h, w = x.shape[-2:]
    H = ((h+downscale_factor)//downscale_factor)*downscale_factor
    W = ((w+downscale_factor)//downscale_factor)*downscale_factor
    padh = H-h if h%downscale_factor!=0 else 0
    padw = W-w if w%downscale_factor!=0 else 0
    x = torch.nn.functional.pad(x, (0, padw, 0, padh), 'reflect')
    return x


def get_gaussian_weights(height: int, width: int, n_channels=3, sigma_scale=0.125):
    """
    Generates a 2D Gaussian mask.

    Args:
        height: Height of the mask.
        width: Width of the mask.
        n_channels: Number of channels.
        sigma_scale: Controls the 'spread' of the bell curve.
                     Lower = sharper peak (uses only strict center).
                     Higher = flatter (averages more of the edge).
    """
    # 1. Create a grid of coordinates
    y_coords = np.arange(height)
    x_coords = np.arange(width)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')

    # 2. Calculate center coordinates
    center_y = height / 2.0
    center_x = width / 2.0

    # 3. Calculate Variance (Sigma^2) based on patch size
    # We scale sigma relative to the patch size so it works for 64x64 or 512x512
    sigma_y = height * sigma_scale
    sigma_x = width * sigma_scale

    # 4. The 2D Gaussian Formula
    # g(x,y) = exp( - ( (x-cx)^2/2sx^2 + (y-cy)^2/2sy^2 ) )
    gaussian = np.exp(
        -((y_grid - center_y)**2 / (2 * sigma_y**2) +
          (x_grid - center_x)**2 / (2 * sigma_x**2))
    )

    # 5. Expand to match image channels (H, W, C)
    gaussian = np.repeat(gaussian[:, :, np.newaxis], n_channels, axis=2)

    return gaussian.astype(np.float32)


def run_model_inference(
    model: Module,
    input_img: np.ndarray,
    device: torch.device,
    normalize: Callable[[np.ndarray], np.ndarray] = normalize,
    patch_size: int | None = None,
    patch_overlap: int = 32,
    need_degradation=False,
    noise_level: int | float | None = None,
    pad: Callable[[torch.Tensor], torch.Tensor] | None = None,
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    progress_bar=None,
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

        # Normalize
        img_normed = normalize(input_img)

        # Process in patches if required
        # Modified from: https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/main_test_swinir.py#L262-L282
        h, w = img_normed.shape[:2]
        if patch_size:
            patch_size = min(patch_size, max(h, w))
            stride = max(patch_size - patch_overlap, 1)
            h_idx_list = list(range(0, h-patch_size, stride)) + [max(h-patch_size, 0)]
            w_idx_list = list(range(0, w-patch_size, stride)) + [max(w-patch_size, 0)]
        else:
            patch_size = max(h, w)
            h_idx_list = [0]
            w_idx_list = [0]

        output_img = np.zeros(shape=(h, w, min(3, img_normed.shape[2])), dtype=np.float32)
        weight_map = np.zeros(shape=(h, w, min(3, img_normed.shape[2])), dtype=np.float32)

        # Pre-calculate the window mask once
        window_mask = get_gaussian_weights(patch_size, patch_size, min(3, img_normed.shape[2]))

        if progress_bar is not None:
            progress_bar = progress_bar.tqdm(None, desc="Processing patches", total=len(h_idx_list) * len(w_idx_list))

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                input_patch = img_normed[h_idx:h_idx+patch_size, w_idx:w_idx+patch_size, :].copy()

                # Add noise if required
                if need_degradation and noise_level is not None:
                    input_patch = add_gaussian_noise(input_patch, noise_level)

                # Convert to tensor: (H, W, C) -> (1, C, H, W)
                input_tensor = torch.from_numpy(input_patch.transpose(2, 0, 1)).unsqueeze(0).to(device)

                if pad is not None:
                    h_patch, w_patch = input_tensor.shape[-2:]
                    input_tensor = pad(input_tensor)
                    output_tensor: torch.Tensor = model(input_tensor)[:, :, :h_patch, :w_patch]
                else:
                    output_tensor: torch.Tensor = model(input_tensor)

                if postprocess is not None:
                    output_tensor = postprocess(output_tensor)

                # Convert back to numpy: (1, C, H, W) -> (H, W, C)
                pred = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                # If the patch at the edge is smaller than patch_size,
                # we need to crop the window_mask to match.
                curr_h, curr_w = pred.shape[:2]
                current_window = window_mask[:curr_h, :curr_w]

                # Accumulate output and weights with weights
                output_img[h_idx:h_idx+curr_h, w_idx:w_idx+curr_w, :] += pred * current_window
                weight_map[h_idx:h_idx+curr_h, w_idx:w_idx+curr_w, :] += current_window

                if progress_bar is not None:
                    progress_bar.update()

        # Average overlapping regions with weights
        output_img /= np.maximum(weight_map, 1e-8)

        # Convert back to original dtype
        if input_img.dtype == np.uint16:
            output_img = np.clip(output_img * 65535.0, 0, 65535).round().astype(np.uint16)
        else:
            output_img = np.clip(output_img * 255.0, 0, 255).round().astype(np.uint8)

    inference_time_ms = (time.time() - start_time) * 1000
    return output_img, inference_time_ms
