import numpy as np

def add_gaussian_noise(img: np.ndarray, sigma=15):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.

    np.random.seed(seed=0)  # for reproducibility
    img += np.random.normal(0, sigma/255., img.shape)
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)
