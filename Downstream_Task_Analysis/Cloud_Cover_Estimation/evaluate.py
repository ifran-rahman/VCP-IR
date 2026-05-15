import os
import cv2
import pandas as pd
from glob import glob
import numpy as np

from src.cloud_metrics import (
    find_gt,
    otsu_cloud_mask,
    cloud_percentage
)


GT_DIR = "/home/ifran/Projects_UBUNTU/Prompt-Aware-Restoration/i2i/DownstreamAnalysis/swimseg/GTmaps"

DATASETS = {
    "clean": "/home/ifran/Projects_UBUNTU/Prompt-Aware-Restoration/i2i/DownstreamAnalysis/swimseg/clean",
    "degraded": "/home/ifran/Projects_UBUNTU/Prompt-Aware-Restoration/i2i/DownstreamAnalysis/swimseg/degraded",
    "restored": "/home/ifran/Projects_UBUNTU/Prompt-Aware-Restoration/i2i/DownstreamAnalysis/swimseg/restored"
}


for name, image_dir in DATASETS.items():

    results = []

    image_paths = sorted(glob(os.path.join(image_dir, "*")))

    for img_path in image_paths:

        gt_path = find_gt(img_path, GT_DIR)

        if gt_path is None:
            continue

        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, 0)

        pred_mask = otsu_cloud_mask(img)

        pred_cloud = cloud_percentage(pred_mask)
        gt_cloud = cloud_percentage(gt)

        results.append({
            "image": os.path.basename(img_path),
            "predicted_cloud": pred_cloud,
            "ground_truth_cloud": gt_cloud,
            "absolute_error": abs(pred_cloud - gt_cloud)
        })

    df = pd.DataFrame(results)

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{name}_results.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

    rmse = np.sqrt((df["absolute_error"] ** 2).mean())
    mae = df["absolute_error"].mean()

    print(f"\n{name.upper()} RESULTS")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE : {mae:.4f}")

