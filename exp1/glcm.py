import numpy as np

def quantize(img, levels):
    img = img.astype(np.float64)
    m, M = img.min(), img.max()
    if M - m < 1e-12:
        return np.zeros_like(img, dtype=np.int32)
    norm = (img - m) / (M - m)
    q = (norm * (levels - 1)).astype(np.int32)
    return q

def glcm_matrix(img_q, distance, angle, levels):
    h, w = img_q.shape
    dx = int(round(np.cos(angle))) * distance
    dy = int(round(np.sin(angle))) * distance
    mat = np.zeros((levels, levels), dtype=np.float64)
    for y in range(h):
        ny = y + dy
        if ny < 0 or ny >= h:
            continue
        for x in range(w):
            nx = x + dx
            if nx < 0 or nx >= w:
                continue
            i = img_q[y, x]
            j = img_q[ny, nx]
            mat[i, j] += 1.0
    s = mat.sum()
    if s > 0:
        mat /= s
    return mat

def glcm_features_from_matrix(mat):
    levels = mat.shape[0]
    i = np.arange(levels)
    j = np.arange(levels)
    I, J = np.meshgrid(i, j, indexing="ij")
    contrast = np.sum((I - J) ** 2 * mat)
    energy = np.sum(mat ** 2)
    homogeneity = np.sum(mat / (1.0 + np.abs(I - J)))
    entropy = -np.sum(np.where(mat > 0, mat * np.log(mat + 1e-12), 0.0))
    return {
        "contrast": float(contrast),
        "energy": float(energy),
        "homogeneity": float(homogeneity),
        "entropy": float(entropy),
    }

def glcm_features(img_gray, levels=32, distances=(1,), angles=(0.0, np.pi/4, np.pi/2, 3*np.pi/4)):
    q = quantize(img_gray, levels)
    feats = []
    for d in distances:
        for a in angles:
            m = glcm_matrix(q, d, a, levels)
            feats.append(glcm_features_from_matrix(m))
    keys = list(feats[0].keys())
    agg = {}
    for k in keys:
        agg[k] = float(np.mean([f[k] for f in feats]))
    return agg
