import os
import cv2
import numpy as np
import pandas as pd


VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]


def find_gt(img_path, gt_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]

    for ext in VALID_EXTENSIONS:
        gt_path = os.path.join(gt_dir, stem + ext)

        if os.path.exists(gt_path):
            return gt_path

    return None


def otsu_cloud_mask(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    R = img_rgb[:, :, 0]
    B = img_rgb[:, :, 2]

    ratio = R / (B + 1e-6)
    ratio = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX)
    ratio = ratio.astype(np.uint8)

    ratio = cv2.GaussianBlur(ratio, (5, 5), 0)

    _, mask = cv2.threshold(
        ratio,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask



def cloud_percentage(mask):
    return (mask > 0).mean() * 100