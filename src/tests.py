import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import datasets
import deblurganv2
import dncnn
import mair
import rednet
import restormer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Global results storage
results_table = []


def calculate_metrics(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0):
    """Calculate PSNR and SSIM metrics between prediction and target."""
    # Ensure images are in valid range
    pred = np.clip(pred, 0, data_range)
    target = np.clip(target, 0, data_range)

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


def run_model_inference(model, input_img: np.ndarray, model_type: str, device: torch.device):
    """Run inference based on model type."""
    with torch.no_grad():
        # Clear GPU cache (if available)
        if torch.cuda.is_available():
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.empty_cache()

        if model_type == 'deblurganv2':
            # DeblurGANv2 uses its own Predictor class. It returns uint8 [0,255]. Convert to float32 [0,1].
            pred: np.ndarray = model(input_img)
            # Convert uint8 (0-255) to float range 0-1 if needed
            if hasattr(pred, 'dtype') and (pred.dtype == np.uint8 or pred.max() > 1.5):
                pred = (pred.astype(np.float32) / 255.0)
        else:
            # Standard PyTorch models
            # Convert to tensor: (H, W, C) -> (1, C, H, W)
            input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

            if model_type == 'restormer':
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

    return pred


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
                model = rednet.get_model('../weights/REDNet/50.pt', device=device)

                psnr_list, ssim_list = [], []
                for noisy_img, clean_img in tqdm(loader, desc="REDNet"):
                    pred = run_model_inference(model, noisy_img, 'rednet', device)
                    p, s = calculate_metrics(pred, clean_img)
                    psnr_list.append(p)
                    ssim_list.append(s)

                results_table.append({
                    'Task': 'Gaussian Denoising',
                    'Type': 'Gray Non-blind',
                    'Dataset': dataset_name,
                    'Sigma': sigma,
                    'Model': 'REDNet',
                    'PSNR': np.mean(psnr_list),
                    'SSIM': np.mean(ssim_list),
                    'Std_PSNR': np.std(psnr_list),
                    'Std_SSIM': np.std(ssim_list)
                })

            # Test DnCNN
            print(f"\nTesting DnCNN on {dataset_name} (sigma={sigma})...")
            model = dncnn.get_model(f'../weights/DnCNN/dncnn_{sigma}.pth', n_channels=1, nb=17, device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="DnCNN"):
                pred = run_model_inference(model, noisy_img, 'dncnn', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'DnCNN',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
            })

            # Test Restormer
            print(f"\nTesting Restormer on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model(f'restormer/options/GaussianGrayDenoising_RestormerSigma{sigma}.yml', device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="Restormer"):
                pred = run_model_inference(model, noisy_img, 'restormer', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
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
            model = dncnn.get_model('../weights/DnCNN/dncnn_gray_blind.pth', n_channels=1, nb=20, device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="DnCNN Blind"):
                pred = run_model_inference(model, noisy_img, 'dncnn', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'DnCNN',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
            })

            # Test Restormer
            print(f"\nTesting Restormer (Blind) on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model('restormer/options/GaussianGrayDenoising_Restormer.yml', device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="Restormer Blind"):
                pred = run_model_inference(model, noisy_img, 'restormer', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Gray Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
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
            model = restormer.get_model(f'restormer/options/GaussianColorDenoising_RestormerSigma{sigma}.yml', device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="Restormer"):
                pred = run_model_inference(model, noisy_img, 'restormer', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
            })

            # Test MaIR
            print(f"\nTesting MaIR on {dataset_name} (sigma={sigma})...")
            model = mair.get_model(f'mair/options/test_MaIR_CDN_s{sigma}.yml', '..')

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="MaIR"):
                pred = run_model_inference(model, noisy_img, 'mair', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Non-blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'MaIR',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
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
            model = dncnn.get_model('../weights/DnCNN/dncnn_color_blind.pth', n_channels=3, nb=20, device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="DnCNN Blind"):
                pred = run_model_inference(model, noisy_img, 'dncnn', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'DnCNN',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
            })

            # Test Restormer
            print(f"\nTesting Restormer (Blind) on {dataset_name} (sigma={sigma})...")
            model = restormer.get_model('restormer/options/GaussianColorDenoising_Restormer.yml', device=device)

            psnr_list, ssim_list = [], []
            for noisy_img, clean_img in tqdm(loader, desc="Restormer Blind"):
                pred = run_model_inference(model, noisy_img, 'restormer', device)
                p, s = calculate_metrics(pred, clean_img)
                psnr_list.append(p)
                ssim_list.append(s)

            results_table.append({
                'Task': 'Gaussian Denoising',
                'Type': 'Color Blind',
                'Dataset': dataset_name,
                'Sigma': sigma,
                'Model': 'Restormer',
                'PSNR': np.mean(psnr_list),
                'SSIM': np.mean(ssim_list),
                'Std_PSNR': np.std(psnr_list),
                'Std_SSIM': np.std(ssim_list)
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
    model = restormer.get_model('restormer/options/RealDenoising_Restormer.yml', device=device)

    psnr_list, ssim_list = [], []
    for noisy_img, clean_img in tqdm(loader, desc="Restormer"):
        pred = run_model_inference(model, noisy_img, 'restormer', device)
        p, s = calculate_metrics(pred, clean_img)
        psnr_list.append(p)
        ssim_list.append(s)

    results_table.append({
        'Task': 'Real Noise Denoising',
        'Type': 'Real',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'Restormer',
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list)
    })

    # Test MaIR
    print(f"\nTesting MaIR on {dataset_name}...")
    model = mair.get_model('mair/realDenoising/options/test_MaIR_RealDN.yml', '..')

    psnr_list, ssim_list = [], []
    for noisy_img, clean_img in tqdm(loader, desc="MaIR"):
        pred = run_model_inference(model, noisy_img, 'mair', device)
        p, s = calculate_metrics(pred, clean_img)
        psnr_list.append(p)
        ssim_list.append(s)

    results_table.append({
        'Task': 'Real Noise Denoising',
        'Type': 'Real',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'MaIR',
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list)
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
    model = restormer.get_model('restormer/options/DefocusDeblur_Single_8bit_Restormer.yml', device=device)

    psnr_list, ssim_list = [], []
    for input_img, target_img, _, _ in tqdm(loader, desc="Restormer Single"):
        pred = run_model_inference(model, input_img, 'restormer', device)
        p, s = calculate_metrics(pred, target_img)
        psnr_list.append(p)
        ssim_list.append(s)

    results_table.append({
        'Task': 'Defocus Deblurring',
        'Type': 'Single-image',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'Restormer',
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list)
    })

    # Test Restormer (Dual-pixel)
    print(f"\nTesting Restormer (Dual-pixel) on {dataset_name}...")
    loader = datasets.defocus_blur_dataset_loader(dataset_name, dual_pixel=True)
    model = restormer.get_model('restormer/options/DefocusDeblur_DualPixel_16bit_Restormer.yml', device=device)

    psnr_list, ssim_list = [], []
    for input_img, target_img, _, _ in tqdm(loader, desc="Restormer Dual"):
        pred = run_model_inference(model, input_img, 'restormer', device)
        p, s = calculate_metrics(pred, target_img)
        psnr_list.append(p)
        ssim_list.append(s)

    results_table.append({
        'Task': 'Defocus Deblurring',
        'Type': 'Dual-pixel',
        'Dataset': dataset_name,
        'Sigma': 'N/A',
        'Model': 'Restormer',
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
        'Std_PSNR': np.std(psnr_list),
        'Std_SSIM': np.std(ssim_list)
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
        model = deblurganv2.Predictor('../weights/DeblurGANv2/fpn_inception.h5', model_name='fpn_inception', device=device)

        psnr_list, ssim_list = [], []
        for input_img, target_img in tqdm(loader, desc=f"DeblurGANv2 Inception | {dataset_name}"):
            pred = run_model_inference(model, input_img, 'deblurganv2', device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'DeblurGANv2 (fpn_inception)',
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list)
        })

        # Test DeblurGANv2 (fpn_mobilenet)
        print(f"\nTesting DeblurGANv2 (fpn_mobilenet) on {dataset_name}...")
        model = deblurganv2.Predictor('../weights/DeblurGANv2/fpn_mobilenet.h5', model_name='fpn_mobilenet', device=device)

        psnr_list, ssim_list = [], []
        for input_img, target_img in tqdm(loader, desc=f"DeblurGANv2 MobileNet | {dataset_name}"):
            pred = run_model_inference(model, input_img, 'deblurganv2', device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'DeblurGANv2 (fpn_mobilenet)',
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list)
        })

        # Test Restormer
        print(f"\nTesting Restormer on {dataset_name}...")
        model = restormer.get_model('restormer/options/Deblurring_Restormer.yml', device=device)

        psnr_list, ssim_list = [], []
        for input_img, target_img in tqdm(loader, desc=f"Restormer | {dataset_name}"):
            pred = run_model_inference(model, input_img, 'restormer', device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'Restormer',
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list)
        })

        # Test MaIR
        print(f"\nTesting MaIR on {dataset_name}...")
        model = mair.get_model('mair/realDenoising/options/test_MaIR_MotionDeblur.yml', '..')

        psnr_list, ssim_list = [], []
        for input_img, target_img in tqdm(loader, desc=f"MaIR | {dataset_name}"):
            pred = run_model_inference(model, input_img, 'mair', device)
            p, s = calculate_metrics(pred, target_img)
            psnr_list.append(p)
            ssim_list.append(s)

        results_table.append({
            'Task': 'Motion Deblurring',
            'Type': 'Motion',
            'Dataset': dataset_name,
            'Sigma': 'N/A',
            'Model': 'MaIR',
            'PSNR': np.mean(psnr_list),
            'SSIM': np.mean(ssim_list),
            'Std_PSNR': np.std(psnr_list),
            'Std_SSIM': np.std(ssim_list)
        })


def save_results(output_path='../results_table.csv'):
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
