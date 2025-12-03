import cv2
import numpy as np


def add_gaussian_noise(img: np.ndarray, sigma=15):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.

    np.random.seed(seed=0)  # for reproducibility
    img += np.random.normal(0, sigma/255., img.shape)
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)


def load_img(file_path: str):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(img, dtype=np.float32) / 255.


def load_img16(file_path: str):
    img = cv2.imread(file_path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.asarray(img, dtype=np.float32) / 65535.


def load_gray_img(file_path: str):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = np.asarray(img, dtype=np.float32) / 255.
    return np.expand_dims(img, axis=2)
