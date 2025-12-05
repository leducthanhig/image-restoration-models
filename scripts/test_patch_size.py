import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch

import deblurganv2
import dncnn
import mair
import rednet
import restormer
from utils import find_max_patch_size
from configs import ROOT_WEIGHTS_DIR


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = rednet.get_model(f'{ROOT_WEIGHTS_DIR}/REDNet/50.pt', device=device)
    print(f"REDNet: {find_max_patch_size(model, device=device, channels=1)}")
    del model

    model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_50.pth', n_channels=1, nb=17, device=device)
    print(f"DnCNN 1: {find_max_patch_size(model, device=device, channels=1)}")
    del model
    model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_gray_blind.pth', n_channels=1, nb=20, device=device)
    print(f"DnCNN 2: {find_max_patch_size(model, device=device, channels=1)}")
    del model
    model = dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_color_blind.pth', n_channels=3, nb=20, device=device)
    print(f"DnCNN 3: {find_max_patch_size(model, device=device, channels=3)}")
    del model

    model = deblurganv2.get_model(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_mobilenet.h5', device=device).model
    print(f"DeblurGANv2 fpn_mobilenet: {find_max_patch_size(model, device=device, channels=3)}")
    del model
    model = deblurganv2.get_model(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_inception.h5', device=device).model
    print(f"DeblurGANv2 fpn_inception: {find_max_patch_size(model, device=device, channels=3)}")
    del model

    model = restormer.get_model('src/restormer/options/GaussianGrayDenoising_Restormer.yml', device=device)
    print(f"Restormer Gaussian Gray Denoising: {find_max_patch_size(model, device=device, channels=1)}")
    del model
    model = restormer.get_model('src/restormer/options/GaussianColorDenoising_Restormer.yml', device=device)
    print(f"Restormer Gaussian Color Denoising: {find_max_patch_size(model, device=device, channels=3)}")
    del model
    model = restormer.get_model('src/restormer/options/DefocusDeblur_Single_8bit_Restormer.yml', device=device)
    print(f"Restormer Defocus Deblur Single 8bit: {find_max_patch_size(model, device=device, channels=3)}")
    del model
    model = restormer.get_model('src/restormer/options/DefocusDeblur_DualPixel_16bit_Restormer.yml', device=device)
    print(f"Restormer Defocus Deblur Dual Pixel 16bit: {find_max_patch_size(model, device=device, channels=6)}")
    del model

    model = mair.get_model('src/mair/options/test_MaIR_CDN_s50.yml')
    print(f"MaIR CDN s50: {find_max_patch_size(model, device=device, channels=3)}")
    del model
    model = mair.get_model('src/mair/realDenoising/options/test_MaIR_RealDN.yml')
    print(f"MaIR RealDN: {find_max_patch_size(model, device=device, channels=3)}")
    del model
