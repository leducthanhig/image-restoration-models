echo "Downloading datasets. This may take a while..."

echo "Downloading DPDD dataset..."
wget 'https://drive.usercontent.google.com/download?id=1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw&export=download&confirm=t' -O test.zip
echo "Extracting DPDD dataset..."
unzip -qd datasets/deblurring/defocus test.zip
rm test.zip
mkdir datasets/deblurring/defocus/test/DPDD
find datasets/deblurring/defocus/test -mindepth 1 -maxdepth 1 ! -name DPDD -exec mv {} datasets/deblurring/defocus/test/DPDD \;

echo "Downloading GoPro dataset..."
wget 'https://drive.usercontent.google.com/download?id=1k6DTSHu4saUgrGTYkkZXTptILyG9RRll&export=download&confirm=t' -O test.zip
echo "Extracting GoPro dataset..."
unzip -qd datasets/deblurring/motion test.zip
rm test.zip
echo "Downloading HIDE dataset..."
wget 'https://drive.usercontent.google.com/download?id=1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A&export=download&confirm=t' -O test.zip
echo "Extracting HIDE dataset..."
unzip -qd datasets/deblurring/motion test.zip
rm test.zip
echo "Downloading RealBlur-J dataset..."
wget 'https://drive.usercontent.google.com/download?id=1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS&export=download&confirm=t' -O test.zip
echo "Extracting RealBlur-J dataset..."
unzip -qd datasets/deblurring/motion test.zip
rm test.zip
echo "Downloading RealBlur-R dataset..."
wget 'https://drive.usercontent.google.com/download?id=1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW&export=download&confirm=t' -O test.zip
echo "Extracting RealBlur-R dataset..."
unzip -qd datasets/deblurring/motion test.zip
rm test.zip

echo "Downloading Gaussian Denoising datasets..."
wget 'https://drive.usercontent.google.com/download?id=1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0&export=download&confirm=t' -O test.zip
echo "Extracting Gaussian Denoising datasets..."
unzip -qd datasets/denoising/gaussian test.zip
rm test.zip

echo "Downloading SIDD dataset..."
wget 'https://drive.usercontent.google.com/download?id=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ&export=download&confirm=t' -O test.zip
echo "Extracting SIDD dataset..."
unzip -qd datasets/denoising/real test.zip
rm test.zip

echo "Finished downloading datasets."
