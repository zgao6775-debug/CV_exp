import numpy as np

def channel_histogram(channel, bins=256):
    counts = np.zeros(bins, dtype=np.int64)
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            v = int(channel[i, j])
            if v < 0:
                v = 0
            if v >= bins:
                v = bins - 1
            counts[v] += 1
    return counts

def rgb_histograms(img, bins=256):
    if img.ndim == 2:
        c = channel_histogram(img, bins)
        return c, c, c
    r = channel_histogram(img[..., 0], bins)
    g = channel_histogram(img[..., 1], bins)
    b = channel_histogram(img[..., 2], bins)
    return r, g, b

def plot_rgb_histograms(r, g, b, save_path):
    try:
        import matplotlib.pyplot as plt
        x = np.arange(len(r))
        plt.figure(figsize=(8, 4))
        plt.plot(x, r, color="r", label="R")
        plt.plot(x, g, color="g", label="G")
        plt.plot(x, b, color="b", label="B")
        plt.xlim(0, len(r) - 1)
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return True
    except Exception:
        return False
