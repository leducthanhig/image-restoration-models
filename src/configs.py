ROOT_DATASET_DIR = 'datasets'
ROOT_WEIGHTS_DIR = 'weights'
ROOT_RESULTS_DIR = 'results'

# Default maximum patch sizes for different models
# These values are tested on GTX 1650 Mobile with 4GB VRAM
PATCH_SIZE = {
    'REDNet': 448,
    'DnCNN': 2048,
    'DeblurGANv2': 2048,
    'Restormer': 544,
    'MaIR': [352, 384],
}
