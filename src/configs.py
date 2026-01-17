ROOT_DATASET_DIR = 'datasets'
ROOT_WEIGHTS_DIR = 'weights'
ROOT_RESULTS_DIR = 'results'

PATCH_CONFIG = {
    'REDNet': {
        'patch_size': 128,
        'patch_overlap': 32
    },
    'DnCNN': {
        'patch_size': 256,
        'patch_overlap': 48
    },
    'DeblurGANv2': [
        {
            'patch_size': 768,
            'patch_overlap': 128
        },
        {
            'patch_size': 2048,
            'patch_overlap': 384
        }
    ],
    'Restormer': [
        {
            'patch_size': 256,
            'patch_overlap': 48
        },
        {
            'patch_size': 512,
            'patch_overlap': 96
        }
    ],
    'MaIR': [
        {
            'patch_size': 128,
            'patch_overlap': 32
        },
        {
            'patch_size': 384,
            'patch_overlap': 128
        }
    ]
}
