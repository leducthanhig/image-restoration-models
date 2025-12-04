download-datasets:
	wget 'https://drive.usercontent.google.com/download?id=1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/deblurring/defocus
	rm test.zip

	wget 'https://drive.usercontent.google.com/download?id=1k6DTSHu4saUgrGTYkkZXTptILyG9RRll&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/deblurring/motion
	rm test.zip
	wget 'https://drive.usercontent.google.com/download?id=1XRomKYJF1H92g1EuD06pCQe4o6HlwB7A&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/deblurring/motion
	rm test.zip
	wget 'https://drive.usercontent.google.com/download?id=1glgeWXCy7Y0qWDc0MXBTUlZYJf8984hS&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/deblurring/motion
	rm test.zip
	wget 'https://drive.usercontent.google.com/download?id=1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/deblurring/motion
	rm test.zip

	wget 'https://drive.usercontent.google.com/download?id=1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/denoising/gaussian
	rm test.zip

	wget 'https://drive.usercontent.google.com/download?id=11vfqV-lqousZTuAit1Qkqghiv_taY0KZ&export=download&confirm=t' -O test.zip
	unzip test.zip -d datasets/denoising/real
	rm test.zip

download-weights:
	wget 'https://drive.usercontent.google.com/download?id=1kBV36HYqwFxacpVvfNCNJrqf0bPUjtis&export=download&confirm=t' -O weights.tar
	tar -xf weights.tar
	rm weights.tar
