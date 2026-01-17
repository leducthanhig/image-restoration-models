import os, sys
sys.path.append(os.path.abspath('src'))
from typing import Literal

import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    imread_uint8,
    imread_uint16,
    imwrite_uint,
    get_model_instance,
    get_patch_config,
    get_model_prediction,
)


def test_gaussian_denoising_gray_nonblind(
    device: torch.device,
    dir_path: str = 'demo',
    models: list[Literal['REDNet', 'DnCNN', 'Restormer']] = ['REDNet', 'DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on grayscale image (non-blind)."""
    print(f"\n{'='*80}")
    print(f"Testing Gaussian Denoising (Gray, Non-blind)")
    print(f"{'='*80}")

    task = 'denoising'
    subtask = 'gaussian'
    sigma = 50

    file_name_prefix = 'denoising_gaussian_gray_nonblind'
    clean_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_target.png'), 1)
    noisy_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_noisy.bmp'), 1)

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,3,1); plt.title('Noisy Image'); plt.axis('off')
    plt.imshow(noisy_img, cmap='gray')
    plt.subplot(2,3,2); plt.title('Clean Image'); plt.axis('off')
    plt.imshow(clean_img, cmap='gray')

    for i, model_name in enumerate(models):
        print(f"\nTesting {model_name}...")
        try:
            model = get_model_instance(task, subtask, model_name, device, gray=True, sigma=sigma)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, noisy_img, device, noise_level=sigma, **patch_config)

        imwrite_uint(os.path.join(dir_path, f"{file_name_prefix}_result_{model_name}.png"), pred)

        plt.subplot(2,3,i+4); plt.title(model_name); plt.axis('off')
        plt.imshow(pred, cmap='gray')

    plt.show()


def test_gaussian_denoising_gray_blind(
    device: torch.device,
    dir_path: str = 'demo',
    models: list[Literal['DnCNN', 'Restormer']] = ['DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on grayscale image (blind)."""
    print(f"\n{'='*80}")
    print(f"Testing Gaussian Denoising (Gray, Blind)")
    print(f"{'='*80}")

    task = 'denoising'
    subtask = 'gaussian'
    sigma = 25

    file_name_prefix = 'denoising_gaussian_gray_blind'
    clean_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_target.png'), 1)
    noisy_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_noisy.bmp'), 1)

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,2,1); plt.title('Noisy Image'); plt.axis('off')
    plt.imshow(noisy_img, cmap='gray')
    plt.subplot(2,2,2); plt.title('Clean Image'); plt.axis('off')
    plt.imshow(clean_img, cmap='gray')

    for i, model_name in enumerate(models):
        print(f"\nTesting {model_name} (Blind)...")
        try:
            model = get_model_instance(task, subtask, model_name, device, gray=True)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, noisy_img, device, noise_level=sigma, **patch_config)

        imwrite_uint(os.path.join(dir_path, f"{file_name_prefix}_result_{model_name}.png"), pred)

        plt.subplot(2,2,i+3); plt.title(model_name); plt.axis('off')
        plt.imshow(pred, cmap='gray')

    plt.show()


def test_gaussian_denoising_color_nonblind(
    device: torch.device,
    dir_path: str = 'demo',
    models: list[Literal['Restormer', 'MaIR']] = ['Restormer', 'MaIR'],
):
    """Test Gaussian denoising on color image (non-blind)."""
    print(f"\n{'='*80}")
    print(f"Testing Gaussian Denoising (Color, Non-blind)")
    print(f"{'='*80}")

    task = 'denoising'
    subtask = 'gaussian'
    sigma = 25

    file_name_prefix = 'denoising_gaussian_color_nonblind'
    clean_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_target.png'))
    noisy_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_noisy.bmp'))

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,2,1); plt.title('Noisy Image'); plt.axis('off')
    plt.imshow(noisy_img)
    plt.subplot(2,2,2); plt.title('Clean Image'); plt.axis('off')
    plt.imshow(clean_img)

    for i, model_name in enumerate(models):
        print(f"\nTesting {model_name}...")
        try:
            model = get_model_instance(task, subtask, model_name, device, sigma=sigma)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, noisy_img, device, noise_level=sigma, **patch_config)

        imwrite_uint(os.path.join(dir_path, f"{file_name_prefix}_result_{model_name}.png"), pred)

        plt.subplot(2,2,i+3); plt.title(model_name); plt.axis('off')
        plt.imshow(pred)

    plt.show()


def test_gaussian_denoising_color_blind(
    device: torch.device,
    dir_path: str = 'demo',
    models: list[Literal['DnCNN', 'Restormer']] = ['DnCNN', 'Restormer'],
):
    """Test Gaussian denoising on color image (blind)."""
    print(f"\n{'='*80}")
    print(f"Testing Gaussian Denoising (Color, Blind)")
    print(f"{'='*80}")

    task = 'denoising'
    subtask = 'gaussian'
    sigma = 25

    file_name_prefix = 'denoising_gaussian_color_blind'
    clean_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_target.png'))
    noisy_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_noisy.bmp'))

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,2,1); plt.title('Noisy Image'); plt.axis('off')
    plt.imshow(noisy_img)
    plt.subplot(2,2,2); plt.title('Clean Image'); plt.axis('off')
    plt.imshow(clean_img)

    for i, model_name in enumerate(models):
        print(f"\nTesting {model_name} (Blind)...")
        try:
            model = get_model_instance(task, subtask, model_name, device)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, noisy_img, device, noise_level=sigma, **patch_config)

        imwrite_uint(os.path.join(dir_path, f"{file_name_prefix}_result_{model_name}.png"), pred)

        plt.subplot(2,2,i+3); plt.title(model_name); plt.axis('off')
        plt.imshow(pred)

    plt.show()


def test_real_noise_denoising(
    device: torch.device,
    dir_path: str = 'demo',
    models: list[Literal['Restormer', 'MaIR']] = ['Restormer', 'MaIR']
):
    """Test real noise denoising."""
    print(f"\n{'='*80}")
    print(f"Testing Real Noise Denoising")
    print(f"{'='*80}")

    task = 'denoising'
    subtask = 'real'

    file_name_prefix = 'denoising_real'
    clean_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_target.bmp'))
    noisy_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_noisy.bmp'))

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,2,1); plt.title('Noisy Image'); plt.axis('off')
    plt.imshow(noisy_img)
    plt.subplot(2,2,2); plt.title('Clean Image'); plt.axis('off')
    plt.imshow(clean_img)

    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            model = get_model_instance(task, subtask, model_name, device)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, noisy_img, device, **patch_config)

        imwrite_uint(os.path.join(dir_path, f"{file_name_prefix}_result_{model_name}.png"), pred)

        plt.subplot(2,2,models.index(model_name)+3); plt.title(model_name); plt.axis('off')
        plt.imshow(pred)

    plt.show()


def test_defocus_blur_deblurring(
    device: torch.device,
    models: list[Literal['Restormer', 'Restormer (Dual-pixel)']] = ['Restormer', 'Restormer (Dual-pixel)']
):
    """Test defocus blur deblurring."""
    print(f"\n{'='*80}")
    print(f"Testing Defocus Blur Deblurring")
    print(f"{'='*80}")

    task = 'deblurring'
    subtask = 'defocus'

    file_name_prefix = 'deblurring_defocus'
    inputL_img = imread_uint16(os.path.join('demo', f'{file_name_prefix}_inputL.png'))
    inputR_img = imread_uint16(os.path.join('demo', f'{file_name_prefix}_inputR.png'))
    inputLR_img = np.concatenate([inputL_img, inputR_img], axis=2)
    inputC_img = imread_uint8(os.path.join('demo', f'{file_name_prefix}_input.png'))
    target_img = imread_uint8(os.path.join('demo', f'{file_name_prefix}_target.png'))

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,2,1); plt.title('Input (C)'); plt.axis('off')
    plt.imshow(inputC_img)
    plt.subplot(2,2,2); plt.title('Target'); plt.axis('off')
    plt.imshow(target_img)

    input_imgs = []
    models = sorted(models)
    if models[0] == 'Restormer':
        input_imgs.append(inputC_img)
    if models[-1] == 'Restormer (Dual-pixel)':
        input_imgs.append(inputLR_img)

    for i, (model_name, input_img) in enumerate(zip(models, input_imgs)):
        print(f"\nTesting {model_name}...")
        try:
            model = get_model_instance(task, subtask, model_name, device)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, input_img, device, **patch_config)

        model_name_clean = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        imwrite_uint(os.path.join('demo', f"{file_name_prefix}_result_{model_name_clean}.png"), pred)

        plt.subplot(2,2,i+3); plt.title(model_name); plt.axis('off')
        plt.imshow(pred if pred.dtype == np.uint8 else (pred / 65535.0).clip(0, 1))

    plt.show()


def test_motion_blur_deblurring(
    device: torch.device,
    dir_path: str = 'demo',
    models: list[Literal['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR']] = ['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR'],
):
    """Test motion blur deblurring."""
    print(f"\n{'='*80}")
    print(f"Testing Motion Blur Deblurring")
    print(f"{'='*80}")

    task = 'deblurring'
    subtask = 'motion'

    file_name_prefix = 'deblurring_motion'
    input_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_input.png'))
    target_img = imread_uint8(os.path.join(dir_path, f'{file_name_prefix}_target.png'))

    plt.figure(figsize=(10,5), dpi=200)
    plt.subplot(2,3,1); plt.title('Input'); plt.axis('off')
    plt.imshow(input_img)
    plt.subplot(2,3,2); plt.title('Target'); plt.axis('off')
    plt.imshow(target_img)

    for i, model_name in enumerate(models):
        print(f"\nTesting {model_name}...")
        try:
            model = get_model_instance(task, subtask, model_name, device)
        except FileNotFoundError:
            print(f"Model weights for {model_name} not found. Skipping this model.")
            continue
        patch_config = get_patch_config(task, subtask, model_name)
        pred, _ = get_model_prediction(model, input_img, device, **patch_config)

        model_name_clean = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        imwrite_uint(os.path.join(dir_path, f"{file_name_prefix}_result_{model_name_clean}.png"), pred)

        plt.subplot(2,3,i+3); plt.title(model_name); plt.axis('off')
        plt.imshow(pred)

    plt.show()


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
