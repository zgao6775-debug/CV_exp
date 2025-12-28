import numpy as np

def to_grayscale(img):
    if img.ndim == 2:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)

def convolve2d(img, kernel):
    img = img.astype(np.float64)
    k = np.flipud(np.fliplr(kernel.astype(np.float64)))
    h, w = img.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)
    return out

def sobel_given(img):
    k = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    g = to_grayscale(img)
    return convolve2d(g, k)

def sobel_magnitude(img):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    ky = kx.T
    g = to_grayscale(img)
    gx = convolve2d(g, kx)
    gy = convolve2d(g, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, mag

def normalize_to_uint8(arr):
    arr = arr.astype(np.float64)
    m, M = arr.min(), arr.max()
    if M - m < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - m) / (M - m)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)
