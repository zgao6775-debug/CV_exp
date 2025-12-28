import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


def get_edge_img(
    color_img,
    gaussian_ksize=5,
    gaussian_sigmax=1,
    canny_threshold1=330,
    canny_threshold2=380,
):
    gaussian = cv2.GaussianBlur(
        color_img,
        (gaussian_ksize, gaussian_ksize),
        gaussian_sigmax,
    )
    gray_img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
    edges_img = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)
    return edges_img


def roi_mask(gray_img, pts):
    mask = np.zeros_like(gray_img)
    mask = cv2.fillPoly(mask, pts=[np.array(pts)], color=255)
    img_mask = cv2.bitwise_and(gray_img, mask)
    return img_mask


def get_lines(edge_img, left_line_prev, right_line_prev):
    def calculate_slope(line):
        x_1, y_1, x_2, y_2 = line[0]
        change_x = x_2 - x_1
        change_y = y_2 - y_1
        return change_y / change_x if change_x != 0 else 0

    def reject_abnormal_lines(lines, threshold=0.3):
        slopes = [calculate_slope(line) for line in lines]
        while len(lines) > 0:
            mean = np.mean(slopes)
            diff = [abs(s - mean) for s in slopes]
            idx = np.argmax(diff)
            if diff[idx] > threshold:
                slopes.pop(idx)
                lines.pop(idx)
            else:
                break
        return lines

    def ransac_fit(lines):
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        X = x_coords.reshape(-1, 1)
        y = y_coords.reshape(-1, 1)
        ransac = RANSACRegressor(residual_threshold=10, max_trials=200)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        x_inliers = x_coords[inlier_mask]
        y_inliers = y_coords[inlier_mask]
        poly = np.polyfit(x_inliers, y_inliers, deg=1)
        point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
        point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
        return np.array([point_min, point_max], dtype=int)

    lines = cv2.HoughLinesP(
        edge_img,
        1,
        np.pi / 180,
        15,
        minLineLength=35,
        maxLineGap=20,
    )
    if lines is None:
        left_line = left_line_prev
        right_line = right_line_prev
    else:
        left_lines = [line for line in lines if calculate_slope(line) > 0]
        right_lines = [line for line in lines if calculate_slope(line) < 0]
        left_lines = reject_abnormal_lines(left_lines)
        right_lines = reject_abnormal_lines(right_lines)

        if len(left_lines) == 0:
            left_line = left_line_prev
        else:
            left_line = ransac_fit(left_lines)

        if len(right_lines) == 0:
            right_line = right_line_prev
        else:
            right_line = ransac_fit(right_lines)

    return left_line, right_line


def draw_lines(img, lines):
    left_line, right_line = lines
    cv2.line(
        img,
        tuple(left_line[0]),
        tuple(left_line[1]),
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.line(
        img,
        tuple(right_line[0]),
        tuple(right_line[1]),
        color=(0, 0, 255),
        thickness=3,
    )

    y_coords = [
        left_line[0][1],
        left_line[1][1],
        right_line[0][1],
        right_line[1][1],
    ]
    y_min, y_max = min(y_coords), max(y_coords)
    slope_left = (left_line[1][1] - left_line[0][1]) / (left_line[1][0] - left_line[0][0])
    intercept_left = left_line[0][1] - slope_left * left_line[0][0]
    slope_right = (right_line[1][1] - right_line[0][1]) / (right_line[1][0] - right_line[0][0])
    intercept_right = right_line[0][1] - slope_right * right_line[0][0]

    if slope_left != 0:
        x_min_left = (y_min - intercept_left) / slope_left
        x_max_left = (y_max - intercept_left) / slope_left
    else:
        x_min_left = x_max_left = 0
    if slope_right != 0:
        x_min_right = (y_min - intercept_right) / slope_right
        x_max_right = (y_max - intercept_right) / slope_right
    else:
        x_min_right = x_max_right = 0

    cv2.line(
        img,
        (int(x_min_left), y_min),
        (int(x_max_left), y_max),
        color=(0, 255, 0),
        thickness=1,
    )
    cv2.line(
        img,
        (int(x_min_right), y_min),
        (int(x_max_right), y_max),
        color=(0, 255, 0),
        thickness=1,
    )


def default_roi_pts(width, height):
    return [
        (int(width * 0.08), int(height * 0.95)),
        (int(width * 0.45), int(height * 0.60)),
        (int(width * 0.55), int(height * 0.60)),
        (int(width * 0.92), int(height * 0.95)),
    ]


def auto_canny_thresholds(bgr_img, sigma=0.33):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    v = float(np.median(gray))
    low = int(max(0.0, (1.0 - sigma) * v))
    high = int(min(255.0, (1.0 + sigma) * v))
    if high <= low:
        high = min(255, low + 1)
    return low, high


def iter_image_paths(data_dir):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    for pat in patterns:
        for p in glob.glob(str(Path(data_dir) / pat)):
            yield p


def process_one_image(path, output_dir, save_debug=False):
    img = cv2.imread(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    pts = default_roi_pts(w, h)
    left_prev = np.array([pts[0], pts[1]])
    right_prev = np.array([pts[3], pts[2]])

    t1, t2 = auto_canny_thresholds(img)
    edges = get_edge_img(img, canny_threshold1=t1, canny_threshold2=t2)
    roi_edges = roi_mask(edges, pts)
    left_line, right_line = get_lines(roi_edges, left_prev, right_prev)

    out = img.copy()
    draw_lines(out, (left_line, right_line))

    stem = Path(path).stem
    out_path = Path(output_dir) / f"{stem}_lane.png"
    cv2.imwrite(str(out_path), out)

    cv2.imwrite(str(Path(output_dir) / f"{stem}_edges.png"), edges)
    if save_debug:
        cv2.imwrite(str(Path(output_dir) / f"{stem}_roi_edges.png"), roi_edges)

    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Lane detection on images in a folder.")
    parser.add_argument("--data-dir", default="data", help="Input image folder.")
    parser.add_argument("--output-dir", default="output_images", help="Output folder.")
    parser.add_argument("--save-debug", action="store_true", help="Save debug edge images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = list(iter_image_paths(args.data_dir))
    if len(paths) == 0:
        raise SystemExit(f"No images found in: {args.data_dir}")

    saved = []
    for p in paths:
        out_path = process_one_image(p, args.output_dir, save_debug=args.save_debug)
        if out_path is not None:
            saved.append(out_path)

    print("saved", len(saved), "images")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()

