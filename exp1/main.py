import os
import numpy as np
from PIL import Image
from .sobel import sobel_given, sobel_magnitude, normalize_to_uint8, to_grayscale
from .histogram import rgb_histograms, plot_rgb_histograms
from .glcm import glcm_features

def read_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_image(arr, path):
    Image.fromarray(arr).save(path)

def process_one(img_path, out_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = read_image(img_path)
    given = sobel_given(img)
    gx, gy, mag = sobel_magnitude(img)
    save_image(normalize_to_uint8(given), os.path.join(out_dir, f"{name}_sobel_given.png"))
    save_image(normalize_to_uint8(gx), os.path.join(out_dir, f"{name}_sobel_gx.png"))
    save_image(normalize_to_uint8(gy), os.path.join(out_dir, f"{name}_sobel_gy.png"))
    save_image(normalize_to_uint8(mag), os.path.join(out_dir, f"{name}_sobel_mag.png"))
    r, g, b = rgb_histograms(img)
    plot_rgb_histograms(r, g, b, os.path.join(out_dir, f"{name}_rgb_hist.png"))
    gray = to_grayscale(img)
    feats = glcm_features(gray, levels=32)
    np.save(os.path.join(out_dir, f"{name}_texture.npy"), feats)
    return {
        "hist_bins": len(r),
        "texture": feats,
        "outputs": [
            f"{name}_sobel_given.png",
            f"{name}_sobel_gx.png",
            f"{name}_sobel_gy.png",
            f"{name}_sobel_mag.png",
            f"{name}_rgb_hist.png",
            f"{name}_texture.npy",
        ],
    }

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    out_dir = os.path.join(base, "output")
    os.makedirs(out_dir, exist_ok=True)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    results = []
    for p in files:
        results.append(process_one(p, out_dir))
    txt = os.path.join(out_dir, "summary.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for r in results:
            f.write(str(r) + "\n")

if __name__ == "__main__":
    main()
