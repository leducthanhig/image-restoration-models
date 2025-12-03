# Test Configurations

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
