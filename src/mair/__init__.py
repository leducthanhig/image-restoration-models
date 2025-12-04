import os.path as osp
import random

import yaml
import numpy as np
import torch

from mair.basicsr.utils.misc import set_random_seed
from mair.basicsr.utils.dist_util import get_dist_info
from mair.basicsr.utils.options import ordered_yaml
from mair.basicsr.models import build_model
from mair.basicsr.models.sr_model import SRModel
from mair.realDenoising.basicsr.models import create_model
from mair.realDenoising.basicsr.models.image_restoration_model import ImageCleanModel


def read_opt(opt_path: str):
    # parse yml to dict
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # distributed settings
    opt['dist'] = False
    # print('Disable distributed.', flush=True)
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['is_train'] = False

    gpus = torch.cuda.device_count()
    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = gpus
    else:
        opt['num_gpu'] = min(opt['num_gpu'], gpus)

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        # if dataset.get('dataroot_gt') is not None:
        #     dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        # if dataset.get('dataroot_lq') is not None:
        #     dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    return opt


def get_model(opt_path: str):
    # parse options, set distributed setting, set ramdom seed
    opt = read_opt(opt_path)
    # create model
    if opt['model_type'] == 'ImageCleanModel':
        model: ImageCleanModel = create_model(opt)
    else:
        model: SRModel = build_model(opt)
    bare_model = model.net_g

    weights_path = opt['path']['pretrain_network_g']
    print(f"Successfully loaded {np.sum([p.numel() for p in bare_model.parameters()]):,} parameters from {weights_path}")
    return bare_model
