import os
import sys
from typing import Literal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import data_loaders
from utils import (get_model_total_parameters,
                   get_model_instance,
                   get_patch_config,
                   get_model_prediction,
                   calculate_metrics,
                   save_result_image)
from configs import ROOT_RESULTS_DIR


# Global results storage
results_table = []


def test_gaussian_denoising_gray_nonblind(
    device: torch.device,
    datasets_list: list[Literal['Set12', 'BSD68', 'Urban100']] = ['Set12', 'BSD68', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['REDNet', 'DnCNN', 'Restormer']] = ['REDNet', 'DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on grayscale images (non-blind)."""
    test_name = 'Gaussian_Denoising_Gray_Nonblind'
    task = 'denoising'
    subtask = 'gaussian'

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Gray, Non-blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=1)

            for model_name in models:
                print(f"\nTesting {model_name} on {dataset_name} (sigma={sigma})...")
                try:
                    model = get_model_instance(task, subtask, model_name, device, gray=True, sigma=sigma)
                except FileNotFoundError:
                    print(f"Model weights for {model_name} not found. Skipping this model.")
                    continue
                model_params = get_model_total_parameters(model)
                patch_config = get_patch_config(task, subtask, model_name)
                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc=model_name):
                    pred, inference_time = get_model_prediction(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **patch_config)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", model_name, img_name)

                results_table.append({
                    'Task': 'Denoising',
                    'Type': 'Gray Non-blind Gaussian Noise',
                    'Dataset': dataset_name,
                    'Sigma': sigma,
                    'Model': model_name,
                    'Model_Params': model_params,
                    'PSNR': np.mean(psnr_list),
                    'SSIM': np.mean(ssim_list),
                    'Std_PSNR': np.std(psnr_list),
                    'Std_SSIM': np.std(ssim_list),
                    'Avg_Time_ms': np.mean(time_list),
                    'Std_Time_ms': np.std(time_list)
                })


def test_gaussian_denoising_gray_blind(
    device: torch.device,
    datasets_list: list[Literal['Set12', 'BSD68', 'Urban100']] = ['Set12', 'BSD68', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['DnCNN', 'Restormer']] = ['DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on grayscale images (blind)."""
    test_name = 'Gaussian_Denoising_Gray_Blind'
    task = 'denoising'
    subtask = 'gaussian'

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Gray, Blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=1)

            for model_name in models:
                print(f"\nTesting {model_name} (Blind) on {dataset_name} (sigma={sigma})...")
                try:
                    model = get_model_instance(task, subtask, model_name, device, gray=True)
                except FileNotFoundError:
                    print(f"Model weights for {model_name} not found. Skipping this model.")
                    continue
                model_params = get_model_total_parameters(model)
                patch_config = get_patch_config(task, subtask, model_name)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc=f"{model_name} Blind"):
                    pred, inference_time = get_model_prediction(model,
                                                                clean_img,
                                                                device,
                                                                need_degradation=True,
                                                                noise_level=sigma,
                                                                **patch_config)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", model_name, img_name)

                results_table.append({
                    'Task': 'Denoising',
                    'Type': 'Gray Blind Gaussian Noise',
                    'Dataset': dataset_name,
                    'Sigma': sigma,
                    'Model': model_name,
                    'Model_Params': model_params,
                    'PSNR': np.mean(psnr_list),
                    'SSIM': np.mean(ssim_list),
                    'Std_PSNR': np.std(psnr_list),
                    'Std_SSIM': np.std(ssim_list),
                    'Avg_Time_ms': np.mean(time_list),
                    'Std_Time_ms': np.std(time_list)
                })


def test_gaussian_denoising_color_nonblind(
    device: torch.device,
    datasets_list: list[Literal['CBSD68', 'Kodak', 'McMaster', 'Urban100']] = ['CBSD68', 'Kodak', 'McMaster', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['Restormer', 'MaIR']] = ['Restormer', 'MaIR'],
):
    """Test Gaussian denoising on color images (non-blind)."""
    test_name = 'Gaussian_Denoising_Color_Nonblind'
    task = 'denoising'
    subtask = 'gaussian'

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Color, Non-blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=3)

            for model_name in models:
                print(f"\nTesting {model_name} on {dataset_name} (sigma={sigma})...")
                try:
                    model = get_model_instance(task, subtask, model_name, device, sigma=sigma)
                except FileNotFoundError:
                    print(f"Model weights for {model_name} not found. Skipping this model.")
                    continue
                model_params = get_model_total_parameters(model)
                patch_config = get_patch_config(task, subtask, model_name)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc=model_name):
                    pred, inference_time = get_model_prediction(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **patch_config)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", model_name, img_name)

                results_table.append({
                    'Task': 'Denoising',
                    'Type': 'Color Non-blind Gaussian Noise',
                    'Dataset': dataset_name,
                    'Sigma': sigma,
                    'Model': model_name,
                    'Model_Params': model_params,
                    'PSNR': np.mean(psnr_list),
                    'SSIM': np.mean(ssim_list),
                    'Std_PSNR': np.std(psnr_list),
                    'Std_SSIM': np.std(ssim_list),
                    'Avg_Time_ms': np.mean(time_list),
                    'Std_Time_ms': np.std(time_list)
                })


def test_gaussian_denoising_color_blind(
    device: torch.device,
    datasets_list: list[Literal['CBSD68', 'Kodak', 'McMaster', 'Urban100']] = ['CBSD68', 'Kodak', 'McMaster', 'Urban100'],
    sigmas: list[int | float] = [15, 25, 50],
    models: list[Literal['DnCNN', 'Restormer']] = ['DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on color images (blind)."""
    test_name = 'Gaussian_Denoising_Color_Blind'
    task = 'denoising'
    subtask = 'gaussian'

    for dataset_name in datasets_list:
        for sigma in sigmas:
            print(f"\n{'='*80}")
            print(f"Testing Gaussian Denoising (Color, Blind) - {dataset_name}, sigma={sigma}")
            print(f"{'='*80}")

            loader = data_loaders.gaussian_noise_dataset_loader(dataset_name, n_channels=3)

            for model_name in models:
                print(f"\nTesting {model_name} (Blind) on {dataset_name} (sigma={sigma})...")
                try:
                    model = get_model_instance(task, subtask, model_name, device)
                except FileNotFoundError:
                    print(f"Model weights for {model_name} not found. Skipping this model.")
                    continue
                model_params = get_model_total_parameters(model)
                patch_config = get_patch_config(task, subtask, model_name)

                psnr_list, ssim_list, time_list = [], [], []
                for clean_img, img_name in tqdm(loader, desc=f"{model_name} Blind"):
                    pred, inference_time = get_model_prediction(model,
                                                               clean_img,
                                                               device,
                                                               need_degradation=True,
                                                               noise_level=sigma,
                                                               **patch_config)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)
                    time_list.append(inference_time)
                    save_result_image(pred, test_name, f"{dataset_name}_Sig{sigma}", model_name, img_name)

                results_table.append({
                    'Task': 'Denoising',
                    'Type': 'Color Blind Gaussian Noise',
                    'Dataset': dataset_name,
                    'Sigma': sigma,
                    'Model': model_name,
                    'Model_Params': model_params,
                    'PSNR': np.mean(psnr_list),
                    'SSIM': np.mean(ssim_list),
                    'Std_PSNR': np.std(psnr_list),
                    'Std_SSIM': np.std(ssim_list),
                    'Avg_Time_ms': np.mean(time_list),
                    'Std_Time_ms': np.std(time_list)
                })


def test_real_noise_denoising(
    device: torch.device,
    models: list[Literal['Restormer', 'MaIR']] = ['Restormer', 'MaIR']
):
    """Test real noise denoising on SIDD dataset."""
    test_name = 'Real_Noise_Denoising'
    task = 'denoising'
    subtask = 'real'
    dataset_name = 'SIDD'

    print(f"\n{'='*80}")
    print(f"Testing Real Noise Denoising - {dataset_name}")
    print(f"{'='*80}")

    loader = data_loaders.real_noise_dataset_loader(dataset_name)

    for model_name in models:
        print(f"\nTesting {model_name} on {dataset_name}...")
        try:
            model = get_model_instance(task, subtask, model_name, device)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        model_params = get_model_total_parameters(model)
        patch_config = get_patch_config(task, subtask, model_name)

        psnr_list, ssim_list, time_list = [], [], []
        for idx, (noisy_img, clean_img) in enumerate(tqdm(loader, desc=model_name)):
            pred, inference_time = get_model_prediction(model, noisy_img, device, **patch_config)
            p, s = calculate_metrics(pred, clean_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            save_result_image(pred, test_name, dataset_name, model_name, f'{idx:04d}.png')

        results_table.append({
            'Task': 'Denoising',
            'Type': 'Real Noise',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': model_name,
            'Model_Params': model_params,
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list),
            'Avg_Time_ms': np.mean(time_list),
            'Std_Time_ms': np.std(time_list)
        })


def test_defocus_blur_deblurring(
    device: torch.device,
    models: list[Literal['Restormer', 'Restormer (Dual-pixel)']] = ['Restormer', 'Restormer (Dual-pixel)']
):
    """Test defocus blur deblurring on DPDD dataset."""
    test_name = 'Defocus_Deblurring'
    task = 'deblurring'
    subtask = 'defocus'
    dataset_name = 'DPDD'

    print(f"\n{'='*80}")
    print(f"Testing Defocus Blur Deblurring - {dataset_name}")
    print(f"{'='*80}")

    for model_name in models:
        print(f"\nTesting {model_name} on {dataset_name}...")
        loader = data_loaders.defocus_blur_dataset_loader(dataset_name, dual_pixel='Dual-pixel' in model_name)
        try:
            model = get_model_instance(task, subtask, model_name, device)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        model_params = get_model_total_parameters(model)
        patch_config = get_patch_config(task, subtask, model_name)

        psnr_list, ssim_list, time_list = [], [], []
        for input_img, target_img, img_name in tqdm(loader, desc=model_name):
            pred, inference_time = get_model_prediction(model, input_img, device, **patch_config)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)
            time_list.append(inference_time)
            model_save_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
            save_result_image(pred, test_name, dataset_name, model_save_name, img_name)

        results_table.append({
            'Task': 'Deblurring',
            'Type': 'Defocus',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': model_name,
            'Model_Params': model_params,
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list),
            'Avg_Time_ms': np.mean(time_list),
            'Std_Time_ms': np.std(time_list)
        })


def test_motion_blur_deblurring(
    device: torch.device,
    datasets_list: list[Literal['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']] = ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R'],
    models: list[Literal['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR']] = ['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR'],
):
    """Test motion blur deblurring on multiple datasets: GoPro, HIDE, RealBlur_J, RealBlur_R."""
    test_name = 'Motion_Deblurring'
    task = 'deblurring'
    subtask = 'motion'

    for dataset_name in datasets_list:
        print(f"\n{'='*80}")
        print(f"Testing Motion Blur Deblurring - {dataset_name}")
        print(f"{'='*80}")

        loader = data_loaders.motion_blur_dataset_loader(dataset_name)

        for model_name in models:
            print(f"\nTesting {model_name} on {dataset_name}...")
            try:
                model = get_model_instance(task, subtask, model_name, device)
            except FileNotFoundError:
                print(f"Model weights for {model_name} not found. Skipping this model.")
                continue
            model_params = get_model_total_parameters(model)
            patch_config = get_patch_config(task, subtask, model_name)

            psnr_list, ssim_list, time_list = [], [], []
            for input_img, target_img, img_name in tqdm(loader, desc=model_name):
                pred, inference_time = get_model_prediction(model, input_img, device, **patch_config)
                p, s = calculate_metrics(pred, target_img)
                psnr_list.append(p)
                ssim_list.append(s)
                time_list.append(inference_time)
                model_save_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
                save_result_image(pred, test_name, dataset_name, model_save_name, img_name)

            results_table.append({
                'Task': 'Deblurring',
                'Type': 'Motion',
                'Dataset': dataset_name,
                'Sigma': 'N/A',
                'Model': model_name,
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Denoising - Gaussian Noise
    test_gaussian_denoising_gray_nonblind(device)
    test_gaussian_denoising_gray_blind(device)
    test_gaussian_denoising_color_nonblind(device)
    test_gaussian_denoising_color_blind(device)

    # Denoising - Real Noise
    test_real_noise_denoising(device)

    # Deblurring - Defocus Blur
    test_defocus_blur_deblurring(device)

    # Deblurring - Motion Blur
    test_motion_blur_deblurring(device)

    # Save all results
    save_results()
