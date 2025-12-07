import os
import sys
from typing import Literal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import data_loaders
import deblurganv2
import dncnn
import mair
import rednet
import restormer
from utils import (get_model_total_parameters,
                   pad,
                   run_model_inference,
                   calculate_metrics,
                   save_result_image)
from configs import ROOT_WEIGHTS_DIR, ROOT_RESULTS_DIR, PATCH_CONFIG


# Global results storage
results_table = []


def test_gaussian_denoising_gray_nonblind(
    datasets_list: list[Literal['Set12', 'BSD68', 'Urban100']] = ['Set12', 'BSD68', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['REDNet', 'DnCNN', 'Restormer']] = ['REDNet', 'DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on grayscale images (non-blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Gray, Non-blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")
            test_name = 'Gaussian_Denoising_Gray_Nonblind'

            # Load dataset
            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=1)

            # Test REDNet
            weights_path = f'{ROOT_WEIGHTS_DIR}/REDNet/{sigma}.pt'
            if 'REDNet' in models and os.path.exists(weights_path):
                print(f"\nTesting REDNet on {dataset_name} (sigma={sigma})...")
                model = rednet.get_model(weights_path, device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="REDNet"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['REDNet'])
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'REDNet', img_name)

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
            weights_path = f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_{sigma}.pth'
            if 'DnCNN' in models and os.path.exists(weights_path):
                print(f"\nTesting DnCNN on {dataset_name} (sigma={sigma})...")
                model = dncnn.get_model(weights_path, n_channels=1, nb=17, device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="DnCNN"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['DnCNN'])
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'DnCNN', img_name)

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
            config_path = f'src/restormer/options/GaussianGrayDenoising_RestormerSigma{sigma}.yml'
            if 'Restormer' in models and os.path.exists(config_path):
                print(f"\nTesting Restormer on {dataset_name} (sigma={sigma})...")
                model = restormer.get_model(config_path, device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="Restormer"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['Restormer'][0],
                                                               pad=pad)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'Restormer', img_name)

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


def test_gaussian_denoising_gray_blind(
    datasets_list: list[Literal['Set12', 'BSD68', 'Urban100']] = ['Set12', 'BSD68', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['DnCNN', 'Restormer']] = ['DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on grayscale images (blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Gray, Blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")
            test_name = 'Gaussian_Denoising_Gray_Blind'

            # Load dataset
            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=1)

            # Test DnCNN
            if 'DnCNN' in models:
                print(f"\nTesting DnCNN (Blind) on {dataset_name} (sigma={sigma})...")
                model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_gray_blind.pth', n_channels=1, nb=20, device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="DnCNN Blind"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['DnCNN'])
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'DnCNN', img_name)

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
            if 'Restormer' in models:
                print(f"\nTesting Restormer (Blind) on {dataset_name} (sigma={sigma})...")
                model = restormer.get_model('src/restormer/options/GaussianGrayDenoising_Restormer.yml', device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="Restormer Blind"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['Restormer'][0],
                                                               pad=pad)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'Restormer', img_name)

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


def test_gaussian_denoising_color_nonblind(
    datasets_list: list[Literal['CBSD68', 'Kodak', 'McMaster', 'Urban100']] = ['CBSD68', 'Kodak', 'McMaster', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['Restormer', 'MaIR']] = ['Restormer', 'MaIR'],
):
    """Test Gaussian denoising on color images (non-blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Color, Non-blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")
            test_name = 'Gaussian_Denoising_Color_Nonblind'

            # Load dataset
            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=3)

            # Test Restormer
            config_path = f'src/restormer/options/GaussianColorDenoising_RestormerSigma{sigma}.yml'
            if 'Restormer' in models and os.path.exists(config_path):
                print(f"\nTesting Restormer on {dataset_name} (sigma={sigma})...")
                model = restormer.get_model(config_path, device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="Restormer"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['Restormer'][0],
                                                               pad=pad)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'Restormer', img_name)

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
            config_path = f'src/mair/options/test_MaIR_CDN_s{sigma}.yml'
            if 'MaIR' in models and os.path.exists(config_path):
                print(f"\nTesting MaIR on {dataset_name} (sigma={sigma})...")
                model = mair.get_model(config_path)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="MaIR"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['MaIR'][0],
                                                               pad=pad)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'MaIR', img_name)

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


def test_gaussian_denoising_color_blind(
    datasets_list: list[Literal['CBSD68', 'Kodak', 'McMaster', 'Urban100']] = ['CBSD68', 'Kodak', 'McMaster', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['DnCNN', 'Restormer']] = ['DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on color images (blind)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Color, Blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")
            test_name = 'Gaussian_Denoising_Color_Blind'

            # Load dataset
            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=3)

            # Test DnCNN
            if 'DnCNN' in models:
                print(f"\nTesting DnCNN (Blind) on {dataset_name} (sigma={sigma})...")
                model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_color_blind.pth', n_channels=3, nb=20, device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="DnCNN Blind"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['DnCNN'])
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'DnCNN', img_name)

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
            if 'Restormer' in models:
                print(f"\nTesting Restormer (Blind) on {dataset_name} (sigma={sigma})...")
                model = restormer.get_model('src/restormer/options/GaussianColorDenoising_Restormer.yml', device=device)
                model_params = get_model_total_parameters(model)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc="Restormer Blind"):
                    pred, inference_time = run_model_inference(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **PATCH_CONFIG['Restormer'][0],
                                                               pad=pad)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", 'Restormer', img_name)

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


def test_real_noise_denoising(models: list[Literal['Restormer', 'MaIR']] = ['Restormer', 'MaIR']):
    """Test real noise denoising on SIDD dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'SIDD'

    print(f"\n{'='*80}")
    print(f"Testing Real Noise Denoising - {dataset_name}")
    print(f"{'='*80}")

    # Load dataset
    loader = data_loaders.real_noise_dataset_loader(dataset_name)

    # Test Restormer
    if 'Restormer' in models:
        print(f"\nTesting Restormer on {dataset_name}...")
        model = restormer.get_model('src/restormer/options/RealDenoising_Restormer.yml', device=device)
        model_params = get_model_total_parameters(model)
        test_name = 'Real_Noise_Denoising'

        psnr_list, ssim_list, time_list = [], [], []
        img_idx = 0
        for noisy_img, clean_img in tqdm(loader, desc="Restormer"):
            pred, inference_time = run_model_inference(model, noisy_img, device, **PATCH_CONFIG['Restormer'][0], pad=pad)
            p, s = calculate_metrics(pred, clean_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, 'Restormer', f'{img_idx:04d}.png')
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
    if 'MaIR' in models:
        print(f"\nTesting MaIR on {dataset_name}...")
        model = mair.get_model('src/mair/realDenoising/options/test_MaIR_RealDN.yml')
        model_params = get_model_total_parameters(model)
        test_name = 'Real_Noise_Denoising'

        psnr_list, ssim_list, time_list = [], [], []
        img_idx = 0
        for noisy_img, clean_img in tqdm(loader, desc="MaIR"):
            pred, inference_time = run_model_inference(model, noisy_img, device, **PATCH_CONFIG['MaIR'][1], pad=pad)
            p, s = calculate_metrics(pred, clean_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, 'MaIR', f'{img_idx:04d}.png')
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
    loader = data_loaders.defocus_blur_dataset_loader(dataset_name, dual_pixel=False)
    model = restormer.get_model('src/restormer/options/DefocusDeblur_Single_8bit_Restormer.yml', device=device)
    model_params = get_model_total_parameters(model)
    test_name = 'Defocus_Deblurring_Single'

    psnr_list, ssim_list, time_list = [], [], []
    for input_img, target_img, img_name in tqdm(loader, desc="Restormer Single"):
        pred, inference_time = run_model_inference(model, input_img, device, **PATCH_CONFIG['Restormer'][1], pad=pad)
        p, s = calculate_metrics(pred, target_img)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(inference_time)
        save_result_image(pred, test_name, dataset_name, 'Restormer', img_name)

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
    loader = data_loaders.defocus_blur_dataset_loader(dataset_name, dual_pixel=True)
    model = restormer.get_model('src/restormer/options/DefocusDeblur_DualPixel_16bit_Restormer.yml', device=device)
    model_params = get_model_total_parameters(model)
    test_name = 'Defocus_Deblurring_Dual'

    psnr_list, ssim_list, time_list = [], [], []
    for input_img, target_img, img_name in tqdm(loader, desc="Restormer Dual"):
        pred, inference_time = run_model_inference(model, input_img, device, **PATCH_CONFIG['Restormer'][1], pad=pad)
        p, s = calculate_metrics(pred, target_img)
        psnr_list.append(p)
        ssim_list.append(s)
        time_list.append(inference_time)
        save_result_image(pred, test_name, dataset_name, 'Restormer', img_name)

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


def test_motion_blur_deblurring(
    datasets_list: list[Literal['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']] = ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R'],
    models: list[Literal['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR']] = ['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR'],
):
    """Test motion blur deblurring on multiple datasets: GoPro, HIDE, RealBlur_J, RealBlur_R."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in datasets_list:
        print(f"\n{'='*80}")
        print(f"Testing Motion Blur Deblurring - {dataset_name}")
        print(f"{'='*80}")

        # Load dataset
        loader = data_loaders.motion_blur_dataset_loader(dataset_name)

        # Test DeblurGANv2 (fpn_inception)
        if 'DeblurGANv2 (Inception)' in models:
            print(f"\nTesting DeblurGANv2 (fpn_inception) on {dataset_name}...")
            model = deblurganv2.get_model(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_inception.h5', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Motion_Deblurring'

            psnr_list, ssim_list, time_list = [], [], []
            for input_img, target_img, img_name in tqdm(loader, desc=f"DeblurGANv2 Inception | {dataset_name}"):
                pred, inference_time = run_model_inference(model,
                                                           input_img,
                                                           device,
                                                           normalize=deblurganv2.normalize,
                                                           **PATCH_CONFIG['DeblurGANv2'][0],
                                                           pad=deblurganv2.pad,
                                                           postprocess=deblurganv2.postprocess)
                p, s = calculate_metrics(pred, target_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'DeblurGANv2_fpn_inception', img_name)

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
        if 'DeblurGANv2 (MobileNet)' in models:
            print(f"\nTesting DeblurGANv2 (fpn_mobilenet) on {dataset_name}...")
            model = deblurganv2.get_model(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_mobilenet.h5', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Motion_Deblurring'

            psnr_list, ssim_list, time_list = [], [], []
            for input_img, target_img, img_name in tqdm(loader, desc=f"DeblurGANv2 MobileNet | {dataset_name}"):
                pred, inference_time = run_model_inference(model,
                                                           input_img,
                                                           device,
                                                           normalize=deblurganv2.normalize,
                                                           **PATCH_CONFIG['DeblurGANv2'][1],
                                                           pad=deblurganv2.pad,
                                                           postprocess=deblurganv2.postprocess)
                p, s = calculate_metrics(pred, target_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'DeblurGANv2_fpn_mobilenet', img_name)

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
        if 'Restormer' in models:
            print(f"\nTesting Restormer on {dataset_name}...")
            model = restormer.get_model('src/restormer/options/Deblurring_Restormer.yml', device=device)
            model_params = get_model_total_parameters(model)
            test_name = 'Motion_Deblurring'

            psnr_list, ssim_list, time_list = [], [], []
            for input_img, target_img, img_name in tqdm(loader, desc=f"Restormer | {dataset_name}"):
                pred, inference_time = run_model_inference(model, input_img, device, **PATCH_CONFIG['Restormer'][1], pad=pad)
                p, s = calculate_metrics(pred, target_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'Restormer', img_name)

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
        if 'MaIR' in models:
            print(f"\nTesting MaIR on {dataset_name}...")
            model = mair.get_model('src/mair/realDenoising/options/test_MaIR_MotionDeblur.yml')
            model_params = get_model_total_parameters(model)
            test_name = 'Motion_Deblurring'

            psnr_list, ssim_list, time_list = [], [], []
            for input_img, target_img, img_name in tqdm(loader, desc=f"MaIR | {dataset_name}"):
                pred, inference_time = run_model_inference(model, input_img, device, **PATCH_CONFIG['MaIR'][1], pad=pad)
                p, s = calculate_metrics(pred, target_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                save_result_image(pred, test_name, dataset_name, 'MaIR', img_name)

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


def save_results(out_dir: str = ROOT_RESULTS_DIR, file_name: str = 'results_summary.csv'):
    """Save results table to CSV file."""
    output_path=os.path.join(out_dir, file_name)
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
