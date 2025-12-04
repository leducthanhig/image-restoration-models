# Introduction

This repository contains implementations and pre-trained models for various image restoration tasks including denoising and deblurring. The models have been tested on multiple datasets and configurations to evaluate their performance.

- Denoising Models: REDNet, DnCNN, Restormer, MaIR
- Deblurring Models: DeblurGANv2, Restormer, MaIR

# Pre-requisites

- Python 3.11+
- GNU Make (optional, for downloading weights and datasets)
- CUDA-capable GPU (optional, for faster inference)

# Set up

## Installation

```bash
pip install -r requirements.txt
```

## Download Pre-trained Weights

```bash
# ~11.6 GB
make download-weights
```

## Download Datasets

```bash
# ~2 GB
make download-datasets
```

# Run Tests

```bash
python src/tests.py
```

# Test Configurations

These tests cover a variety of image restoration tasks including denoising and deblurring using different datasets and models.
After running the tests, the results will be saved in `results/` directory and a summary CSV file `results/results_summary.csv`.

## Denoising

### Gaussian Noise

* Gray Image
    * Non-blind
        * Sigmas: 15, 25, 50
        * Datasets: Set12, BSD68, Urban100
        * Models: REDNet (sig=50 only), DnCNN, Restormer
    * Blind
        * Sigmas: 15, 25, 50
        * Datasets: Set12, BSD68, Urban100
        * Models: DnCNN, Restormer
* Color Image
    * Non-blind
        * Sigmas: 15, 25, 50
        * Datasets: CBSD68, Kodak, McMaster, Urban100
        * Models: Restormer, MaIR
    * Blind
        * Sigmas: 15, 25, 50
        * Datasets: CBSD68, Kodak, McMaster, Urban100
        * Models: DnCNN, Restormer

### Real Noise

* Datasets: SIDD
* Models: Restormer, MaIR

## Deblurring

### Defocus Blur

* Datasets: DPDD
* Models: Restormer (single-image, dual-pixel)

### Motion Blur

* Datasets: GoPro, HIDE, RealBlur_J, RealBlur_R
* Models: DeblurGANv2 (fpn_inception, fpn_mobilenet), Restormer, MaIR

# Credits

- The original implementations of the models used in this repository are credited to their respective authors:
    - [REDNet](https://bitbucket.org/chhshen/image-denoising)
    - [DnCNN](https://github.com/cszn/DnCNN)
    - [DeblurGANv2](https://github.com/KupynOrest/DeblurGANv2)
    - [Restormer](https://github.com/swz30/Restormer)
    - [MaIR](https://github.com/XLearning-SCU/2025-CVPR-MaIR)

- The tool for converting Caffe models (used in REDNet) to PyTorch is credited to [caffemodel2pytorch](https://github.com/dwml/caffemodel2pytorch).

- The datasets used for testing are provided by Restormer repository.
