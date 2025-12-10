echo "Downloading pre-trained weights..."

echo "Downloading REDNet weights..."
mkdir weights/REDNet
wget https://bitbucket.org/chhshen/image-denoising/raw/master/model/denoising/50.caffemodel
python caffemodel2pytorch/caffemodel2pytorch.py 50.caffemodel -o weights/REDNet/50.pt
rm 50.caffemodel

echo "Downloading DnCNN weights..."
mkdir weights/DnCNN
for noise in 15 25 50; do
    wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_${noise}.pth -P weights/DnCNN/
done
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth -P weights/DnCNN/
wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth -P weights/DnCNN/

echo "Downloading DeblurGANv2 weights..."
mkdir weights/DeblurGANv2
wget 'https://drive.usercontent.google.com/download?id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR&export=download&confirm=t' \
    -O weights/DeblurGANv2/fpn_inception.h5
wget 'https://drive.usercontent.google.com/download?id=1JhnT4BBeKBBSLqTo6UsJ13HeBXevarrU&export=download&confirm=t' \
    -O weights/DeblurGANv2/fpn_mobilenet.h5

echo "Downloading Restormer weights..."
mkdir weights/Restormer
mkdir weights/Restormer/denoising
wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth \
    -P weights/Restormer/denoising/
wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_blind.pth \
    -P weights/Restormer/denoising/
for noise in 15 25 50; do
    wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_sigma${noise}.pth \
        -P weights/Restormer/denoising/
    wget https://github.com/swz30/Restormer/releases/download/v1.0/gaussian_gray_denoising_sigma${noise}.pth \
        -P weights/Restormer/denoising/
done
wget https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth \
    -P weights/Restormer/denoising/
mkdir weights/Restormer/deblurring
wget https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth \
    -P weights/Restormer/deblurring/
wget https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth \
    -P weights/Restormer/deblurring/
wget https://github.com/swz30/Restormer/releases/download/v1.0/dual_pixel_defocus_deblurring.pth \
    -P weights/Restormer/deblurring/

echo "Downloading MaIR weights..."
mkdir weights/MaIR
mkdir weights/MaIR/denoising
wget 'https://drive.usercontent.google.com/download?id=1XUDCSK1Cs492mopqQrDVLNCC2stO1paA&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_CDN_s15.pth
wget 'https://drive.usercontent.google.com/download?id=1jIDSzksBracVnyiVSkwFNEX--JOP1H1i&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_CDN_s25.pth
wget 'https://drive.usercontent.google.com/download?id=1YdhrrPfEZ70JVuJgFdTmSLtFuu2giFdb&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_CDN_s50.pth
wget 'https://drive.usercontent.google.com/download?id=1M8pDYp_-Yl46pMFqv_tnImJ8w1z6h7bH&export=download&confirm=t' \
    -O weights/MaIR/denoising/MaIR_RealDN.pth
mkdir weights/MaIR/deblurring
wget 'https://drive.usercontent.google.com/download?id=1bdYWJ0FXYknQuJQg77KrwII2jJHlX-3k&export=download&confirm=t' \
    -O weights/MaIR/deblurring/MaIR_MotionDeblur.pth

echo "Finished downloading weights."
