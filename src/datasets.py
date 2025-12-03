import os
from typing import Literal
from glob import glob

import numpy as np
import scipy.io as sio
from natsort import natsorted

from utils import load_img, load_img16, load_gray_img, add_gaussian_noise


class DataLoader:
    """Wrap a generator factory with a __len__ so it can be used by progress bars.

    Example:
        gen = DataLoader(my_gen_factory, length)
        for item in gen:  # iteration
            ...
        len(gen)  # returns the configured length
    """

    def __init__(self, gen_factory, length: int):
        self._gen_factory = gen_factory
        self._length = int(length)

    def __iter__(self):
        return self._gen_factory()

    def __len__(self):
        return self._length


def defocus_blur_dataset_loader(name='DPDD', dual_pixel=False):
    dir_path = os.path.join('..', 'data', 'deblurring', 'defocus', name)
    inputC_dir = os.path.join(dir_path, 'inputC')
    inputL_dir = os.path.join(dir_path, 'inputL')
    inputR_dir = os.path.join(dir_path, 'inputR')
    target_dir = os.path.join(dir_path, 'target')

    inputC_files = natsorted(glob(os.path.join(inputC_dir, '*.*')))
    inputL_files = natsorted(glob(os.path.join(inputL_dir, '*.*')))
    inputR_files = natsorted(glob(os.path.join(inputR_dir, '*.*')))
    target_files = natsorted(glob(os.path.join(target_dir, '*.*')))

    indoor_labels = np.load(os.path.join(dir_path, 'indoor_labels.npy'))
    outdoor_labels = np.load(os.path.join(dir_path, 'outdoor_labels.npy'))

    length = len(target_files)

    def gen():
        for i in range(length):
            inputC_file = inputC_files[i]
            inputL_file = inputL_files[i]
            inputR_file = inputR_files[i]
            target_file = target_files[i]
            indoor_label = indoor_labels[i]
            outdoor_label = outdoor_labels[i]

            if dual_pixel:
                inputL_img = load_img16(inputL_file)
                inputR_img = load_img16(inputR_file)
                input_img = np.concatenate([inputL_img, inputR_img], axis=2)
                target_img = load_img16(target_file)
            else:
                input_img = load_img(inputC_file)
                target_img = load_img(target_file)

            yield input_img, target_img, indoor_label, outdoor_label

    return DataLoader(gen, length)


def motion_blur_dataset_loader(name: Literal['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R'] = 'GoPro'):
    dir_path = os.path.join('..', 'data', 'deblurring', 'motion', name)
    input_dir = os.path.join(dir_path, 'input')
    target_dir = os.path.join(dir_path, 'target')

    input_files = natsorted(glob(os.path.join(input_dir, '*.*')))
    target_files = natsorted(glob(os.path.join(target_dir, '*.*')))

    length = len(target_files)

    def gen():
        for i in range(length):
            input_file = input_files[i]
            target_file = target_files[i]
            input_img = load_img(input_file)
            target_img = load_img(target_file)

            yield input_img, target_img

    return DataLoader(gen, length)


def gaussian_noise_dataset_loader(name: Literal['Set12', 'BSD68', 'CBSD68', 'Kodak', 'McMaster', 'Urban100'] = 'BSD68', sigma=15, n_channels=1):
    dir_path = os.path.join('..', 'data', 'denoising', 'gaussian', name)
    files = natsorted(glob(os.path.join(dir_path, '*.*')))

    length = len(files)

    def gen():
        for file in files:
            if n_channels == 1:
                img = load_gray_img(file)
            else:
                img = load_img(file)
            noisy_img = add_gaussian_noise(img, sigma=sigma)

            yield noisy_img, img

    return DataLoader(gen, length)


def real_noise_dataset_loader(name='SIDD'):
    dir_path = os.path.join('..', 'data', 'denoising', 'real', name)
    noisy_mat = sio.loadmat(os.path.join(dir_path, 'ValidationNoisyBlocksSrgb.mat'))
    noisy_images = np.asarray(noisy_mat['ValidationNoisyBlocksSrgb'], dtype=np.float32) / 255.
    gt_mat = sio.loadmat(os.path.join(dir_path, 'ValidationGtBlocksSrgb.mat'))
    gt_images = np.asarray(gt_mat['ValidationGtBlocksSrgb'], dtype=np.float32) / 255.

    # handle both (N, H, W, C) and (N, M, H, W, C) shapes from SIDD
    if noisy_images.ndim == 4:
        length = noisy_images.shape[0]

        def gen():
            for i in range(length):
                yield noisy_images[i], gt_images[i]
    else:
        N, M = noisy_images.shape[0], noisy_images.shape[1]
        length = N * M

        def gen():
            for i in range(N):
                for j in range(M):
                    yield noisy_images[i, j], gt_images[i, j]

    return DataLoader(gen, length)
