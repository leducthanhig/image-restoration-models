import os
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import Module
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import datasets
import deblurganv2
import dncnn
import mair
import rednet
import restormer
from utils import (add_gaussian_noise,
                   get_model_total_parameters,
                   imwrite_uint)
from configs import ROOT_WEIGHTS_DIR, ROOT_RESULTS_DIR


# Global results storage
results_table = []


def get_result_save_dir(test_name: str, dataset_name: str, model_name: str) -> str:
    """Get the directory path for saving results."""
    dir_path = os.path.join(ROOT_RESULTS_DIR, test_name, dataset_name, model_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_result_image(pred: np.ndarray, test_name: str, dataset_name: str, model_name: str, img_idx: int) -> str:
    """Save prediction image to results directory."""
    dir_path = get_result_save_dir(test_name, dataset_name, model_name)
    file_path = os.path.join(dir_path, f'{img_idx:04d}.png')
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


def run_model_inference(model: Module | deblurganv2.Predictor, input_img: np.ndarray, device: torch.device, need_degradation=False, noise_level: float | None = None) -> tuple[np.ndarray, float]:
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

        if isinstance(model, deblurganv2.Predictor):
            # DeblurGANv2 uses its own Predictor class. It returns uint8 [0,255]
            pred: np.ndarray = model(input_img)
        else:
            # Standard PyTorch models

            # Normalize to [0,1]
            if input_img.dtype == np.uint8:
                img_normed = input_img.astype(np.float32) / 255.0
            elif input_img.dtype == np.uint16:
                img_normed = input_img.astype(np.float32) / 65535.0

            # Add noise if required
            if need_degradation and noise_level is not None:
                img_normed = add_gaussian_noise(img_normed, noise_level)

            # Convert to tensor: (H, W, C) -> (1, C, H, W), gt
            input_tensor = torch.from_numpy(img_normed.transpose(2, 0, 1)).unsqueeze(0).to(device)

            if isinstance(model, restormer.Restormer):
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
            if input_img.dtype == np.uint8:
                pred = np.clip(pred * 255.0, 0, 255).round().astype(np.uint8)
            elif input_img.dtype == np.uint16:
                pred = np.clip(pred * 65535.0, 0, 65535).round().astype(np.uint16)

    inference_time_ms = (time.time() - start_time) * 1000
    return pred, inference_time_ms


def test_gaussian_denoising_gray_nonblind():
    """Test Gaussian denoising on grayscale images (non-blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets_list = ['Set12', 'BSD68', 'Urban100']
    sigmas = [15, 25, 50]

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Gray, Non-blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            # Load dataset
            loader = datasets.gaussian_noise_dataset_loader(dataset_name, sigma=sigma, n_channels=1)

            # Test REDNet (sigma=50 only)
            if sigma == 50:
                print(f"\nTesting REDNet on {dataset_name} (sigma={sigma})...")
                model = rednet.get_model(f'{ROOT_WEIGHTS_DIR}/REDNet/50.pt', device=device)
                model_params = get_model_total_parameters(model)
                test_name = 'Gaussian_Denoising_Gray_NonBlind'

                psnr_list, ssim_list, time_list = [], [], []
                img_idx = 0
                for clean_img in tqdm(loader, desc="REDNet"):
                    pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, dataset_name, 'REDNet', img_idx)
                    img_idx += 1

                results_table.append({
                    'Task': 'Gaussian Denoising',
                    'Type': 'Gray Non-blind',
                    'Dataset': dataset_name,
                    'Sigma': sigma,
                    'Model': 'REDNet',
                    'Model_Params': model_params,
                    'PSNR': np.mean(psnr_list),
                    'SSIM': np.mean(ssim_list),
                    'Std_PSNR': np.std(psnr_list),
                    'Std_SSIM': np.std(ssim_list),
                    'Avg_Time_ms': np.mean(time_list),
                    'Std_Time_ms': np.std(time_list)
                })

            # Test DnCNN
            print(f"\nTesting DnCNN on {dataset_name} (sigma={sigma})...")
            model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_{sigma}.pth', n_channels=1, nb=17, device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Gray_NonBlind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="DnCNN"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'DnCNN', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'DnCNN',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })

            # Test Restormer
            print(f"\nTesting Restormer on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model(f'src/restormer/options/GaussianGrayDenoising_RestormerSigma{sigma}.yml', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Gray_NonBlind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="Restormer"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })


def test_gaussian_denoising_gray_blind():
    """Test Gaussian denoising on grayscale images (blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets_list = ['Set12', 'BSD68', 'Urban100']
    sigmas = [15, 25, 50]

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Gray, Blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            # Load dataset
            loader = datasets.gaussian_noise_dataset_loader(dataset_name, sigma=sigma, n_channels=1)

            # Test DnCNN
            print(f"\nTesting DnCNN (Blind) on {dataset_name} (sigma={sigma})...")
            model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_gray_blind.pth', n_channels=1, nb=20, device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Gray_Blind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="DnCNN Blind"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'DnCNN', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'DnCNN',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })

            # Test Restormer
            print(f"\nTesting Restormer (Blind) on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model('src/restormer/options/GaussianGrayDenoising_Restormer.yml', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Gray_Blind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="Restormer Blind"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })


def test_gaussian_denoising_color_nonblind():
    """Test Gaussian denoising on color images (non-blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets_list = ['CBSD68', 'Kodak', 'McMaster', 'Urban100']
    sigmas = [15, 25, 50]

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Color, Non-blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            # Load dataset
            loader = datasets.gaussian_noise_dataset_loader(dataset_name, sigma=sigma, n_channels=3)

            # Test Restormer
            print(f"\nTesting Restormer on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model(f'src/restormer/options/GaussianColorDenoising_RestormerSigma{sigma}.yml', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Color_NonBlind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="Restormer"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })

            # Test MaIR
            print(f"\nTesting MaIR on {dataset_name} (sigma={sigma})...")
            model = mair.get_model(f'src/mair/options/test_MaIR_CDN_s{sigma}.yml')
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Color_NonBlind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="MaIR"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'MaIR', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'MaIR',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })


def test_gaussian_denoising_color_blind():
    """Test Gaussian denoising on color images (blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets_list = ['CBSD68', 'Kodak', 'McMaster', 'Urban100']
    sigmas = [15, 25, 50]

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Color, Blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            # Load dataset
            loader = datasets.gaussian_noise_dataset_loader(dataset_name, sigma=sigma, n_channels=3)

            # Test DnCNN
            print(f"\nTesting DnCNN (Blind) on {dataset_name} (sigma={sigma})...")
            model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_color_blind.pth', n_channels=3, nb=20, device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Color_Blind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="DnCNN Blind"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'DnCNN', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'DnCNN',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })

            # Test Restormer
            print(f"\nTesting Restormer (Blind) on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model('src/restormer/options/GaussianColorDenoising_Restormer.yml', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Gaussian_Denoising_Color_Blind'

            psnr_list, ssim_list, time_list = [], [], []
            img_idx = 0
            for clean_img in tqdm(loader, desc="Restormer Blind"):
                pred, inference_time = run_model_inference(model, clean_img, device, need_degradation=True, noise_level=sigma)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
                img_idx += 1

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'Model_Params': model_params,
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list),
                'Avg_Time_ms': np.mean(time_list),
                'Std_Time_ms': np.std(time_list)
            })


def test_real_noise_denoising():
    """Test real noise denoising on SIDD dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'SIDD'

    print(f"\n{'='*80}")
    print(f"Testing Real Noise Denoising - {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    loader = datasets.real_noise_dataset_loader(dataset_name)

    # Test Restormer
    print(f"\nTesting Restormer on {dataset_name}...")
    model = restormer.get_model('src/restormer/options/RealDenoising_Restormer.yml', device=device)
    model_params = get_model_total_parameters(model)
    test_name = 'Real_Noise_Denoising'

    psnr_list, ssim_list, time_list = [], [], []
    img_idx = 0
    for noisy_img, clean_img in tqdm(loader, desc="Restormer"):
        pred, inference_time = run_model_inference(model, noisy_img, device)
        p, s = calculate_metrics(pred, clean_img)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(inference_time)
        save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
        img_idx += 1

    results_table.append({
        'Task': 'Real Noise Denoising',
        'Type': 'Real',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'Restormer',
        'Model_Params': model_params,
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list),
        'Avg_Time_ms': np.mean(time_list),
        'Std_Time_ms': np.std(time_list)
    })

    # Test MaIR
    print(f"\nTesting MaIR on {dataset_name}...")
    model = mair.get_model('src/mair/realDenoising/options/test_MaIR_RealDN.yml')
    model_params = get_model_total_parameters(model)
    test_name = 'Real_Noise_Denoising'

    psnr_list, ssim_list, time_list = [], [], []
    img_idx = 0
    for noisy_img, clean_img in tqdm(loader, desc="MaIR"):
        pred, inference_time = run_model_inference(model, noisy_img, device)
        p, s = calculate_metrics(pred, clean_img)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(inference_time)
        save_result_image(pred, test_name, dataset_name, 'MaIR', img_idx)
        img_idx += 1

    results_table.append({
        'Task': 'Real Noise Denoising',
        'Type': 'Real',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'MaIR',
        'Model_Params': model_params,
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list),
        'Avg_Time_ms': np.mean(time_list),
        'Std_Time_ms': np.std(time_list)
    })


def test_defocus_blur_deblurring():
    """Test defocus blur deblurring on DPDD dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'DPDD'

    print(f"\n{'='*80}")
    print(f"Testing Defocus Blur Deblurring - {dataset_name}")
    print(f"{'='*80}")

    # Test Restormer (Single-image)
    print(f"\nTesting Restormer (Single-image) on {dataset_name}...")
    loader = datasets.defocus_blur_dataset_loader(dataset_name, dual_pixel=False)
    model = restormer.get_model('src/restormer/options/DefocusDeblur_Single_8bit_Restormer.yml', device=device)
    model_params = get_model_total_parameters(model)
    test_name = 'Defocus_Deblurring_Single'

    psnr_list, ssim_list, time_list = [], [], []
    img_idx = 0
    for input_img, target_img, _, _ in tqdm(loader, desc="Restormer Single"):
        pred, inference_time = run_model_inference(model, input_img, device)
        p, s = calculate_metrics(pred, target_img)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(inference_time)
        save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
        img_idx += 1

    results_table.append({
        'Task': 'Defocus Deblurring',
        'Type': 'Single-image',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'Restormer',
        'Model_Params': model_params,
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list),
        'Avg_Time_ms': np.mean(time_list),
        'Std_Time_ms': np.std(time_list)
    })

    # Test Restormer (Dual-pixel)
    print(f"\nTesting Restormer (Dual-pixel) on {dataset_name}...")
    loader = datasets.defocus_blur_dataset_loader(dataset_name, dual_pixel=True)
    model = restormer.get_model('src/restormer/options/DefocusDeblur_DualPixel_16bit_Restormer.yml', device=device)
    model_params = get_model_total_parameters(model)
    test_name = 'Defocus_Deblurring_Dual'

    psnr_list, ssim_list, time_list = [], [], []
    img_idx = 0
    for input_img, target_img, _, _ in tqdm(loader, desc="Restormer Dual"):
        pred, inference_time = run_model_inference(model, input_img, device)
        p, s = calculate_metrics(pred, target_img)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(inference_time)
        save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
        img_idx += 1

    results_table.append({
        'Task': 'Defocus Deblurring',
        'Type': 'Dual-pixel',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'Restormer',
        'Model_Params': model_params,
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list),
        'Avg_Time_ms': np.mean(time_list),
        'Std_Time_ms': np.std(time_list)
    })


def test_motion_blur_deblurring():
    """Test motion blur deblurring on multiple datasets: GoPro, HIDE, RealBlur_J, RealBlur_R."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets_list = ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']

    for dataset_name in datasets_list:
        print(f"\n{'='*80}")
        print(f"Testing Motion Blur Deblurring - {dataset_name}")
        print(f"{'='*80}")

        # Load dataset
        loader = datasets.motion_blur_dataset_loader(dataset_name)

        # Test DeblurGANv2 (fpn_inception)
        print(f"\nTesting DeblurGANv2 (fpn_inception) on {dataset_name}...")
        model = deblurganv2.Predictor(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_inception.h5', model_name='fpn_inception', device=device)
        model_params = get_model_total_parameters(model)
        test_name = 'Motion_Deblurring'

        psnr_list, ssim_list, time_list = [], [], []
        img_idx = 0
        for input_img, target_img in tqdm(loader, desc=f"DeblurGANv2 Inception | {dataset_name}"):
            pred, inference_time = run_model_inference(model, input_img, device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, 'DeblurGANv2_fpn_inception', img_idx)
            img_idx += 1

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'DeblurGANv2 (fpn_inception)',
            'Model_Params': model_params,
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list),
            'Avg_Time_ms': np.mean(time_list),
            'Std_Time_ms': np.std(time_list)
        })

        # Test DeblurGANv2 (fpn_mobilenet)
        print(f"\nTesting DeblurGANv2 (fpn_mobilenet) on {dataset_name}...")
        model = deblurganv2.Predictor(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_mobilenet.h5', model_name='fpn_mobilenet', device=device)
        model_params = get_model_total_parameters(model)
        test_name = 'Motion_Deblurring'

        psnr_list, ssim_list, time_list = [], [], []
        img_idx = 0
        for input_img, target_img in tqdm(loader, desc=f"DeblurGANv2 MobileNet | {dataset_name}"):
            pred, inference_time = run_model_inference(model, input_img, device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, 'DeblurGANv2_fpn_mobilenet', img_idx)
            img_idx += 1

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'DeblurGANv2 (fpn_mobilenet)',
            'Model_Params': model_params,
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list),
            'Avg_Time_ms': np.mean(time_list),
            'Std_Time_ms': np.std(time_list)
        })

        # Test Restormer
        print(f"\nTesting Restormer on {dataset_name}...")
        model = restormer.get_model('src/restormer/options/Deblurring_Restormer.yml', device=device)
        model_params = get_model_total_parameters(model)
        test_name = 'Motion_Deblurring'

        psnr_list, ssim_list, time_list = [], [], []
        img_idx = 0
        for input_img, target_img in tqdm(loader, desc=f"Restormer | {dataset_name}"):
            pred, inference_time = run_model_inference(model, input_img, device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, 'Restormer', img_idx)
            img_idx += 1

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'Restormer',
            'Model_Params': model_params,
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list),
            'Avg_Time_ms': np.mean(time_list),
            'Std_Time_ms': np.std(time_list)
        })

        # Test MaIR
        print(f"\nTesting MaIR on {dataset_name}...")
        model = mair.get_model('src/mair/realDenoising/options/test_MaIR_MotionDeblur.yml')
        model_params = get_model_total_parameters(model)
        test_name = 'Motion_Deblurring'

        psnr_list, ssim_list, time_list = [], [], []
        img_idx = 0
        for input_img, target_img in tqdm(loader, desc=f"MaIR | {dataset_name}"):
            pred, inference_time = run_model_inference(model, input_img, device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, 'MaIR', img_idx)
            img_idx += 1

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'MaIR',
            'Model_Params': model_params,
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list),
            'Avg_Time_ms': np.mean(time_list),
            'Std_Time_ms': np.std(time_list)
        })


def save_results(output_path=os.path.join(ROOT_RESULTS_DIR, 'results_summary.csv')):
    """Save results table to CSV file."""
    df = pd.DataFrame(results_table)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print(f"{'='*80}")
    print("\nResults Summary:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    # Denoising - Gaussian Noise
    test_gaussian_denoising_gray_nonblind()
    test_gaussian_denoising_gray_blind()
    test_gaussian_denoising_color_nonblind()
    test_gaussian_denoising_color_blind()

    # Denoising - Real Noise
    test_real_noise_denoising()

    # Deblurring - Defocus Blur
    test_defocus_blur_deblurring()

    # Deblurring - Motion Blur
    test_motion_blur_deblurring()

    # Save all results
    save_results()
