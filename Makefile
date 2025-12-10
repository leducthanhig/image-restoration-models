install-packages:
	pip install uv
	uv pip install -r requirements.txt \
		torch==2.7 torchvision --extra-index-url https://download.pytorch.org/whl/cu126 \
		https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

download-datasets:
	sh scripts/download_datasets.sh

download-weights:
	sh scripts/download_weights.sh
