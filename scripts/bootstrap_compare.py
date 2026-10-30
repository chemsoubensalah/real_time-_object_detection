"""How big is the noise? Bootstrap confidence interval for the AP@0.5 difference between two models.

Each model runs once on the test images. Then we draw B resamples of the images WITH replacement (some images
twice, some not at all), compute AP@0.5 for both models on the same resample, and look at the spread of the
difference. If the 95% interval contains 0, the difference can be luck of which images were chosen.

Usage:
    python scripts/bootstrap_compare.py --a models/v2_yolo11n_12cls.pt --b models/v3_yolo11n_12cls.pt --out results/v3/bootstrap_v2_v3.json
"""
import argparse
import glob
import json
import os
import random

import cv2
import torch
from torchmetrics.detection import MeanAveragePrecision

import detect_realtime as dr
from evaluate_v2 import CLASSES, read_gt

SETS = {"coco": "data/v2_test_coco", "drone": "data/v2_test_drone"}


def predictions(weights, size, folder):
    model = dr.load_model(weights, 0.001, "cpu", "../yolov5")
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "images", "*"))):
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        det = dr.detect(model, frame, size)
        gb, gl = read_gt(os.path.join(folder, "labels", os.path.splitext(os.path.basename(path))[0] + ".txt"), w, h)
        out.append(({"boxes": det[:, :4], "scores": det[:, 4], "labels": det[:, 5].long()}, {"boxes": gb, "labels": gl}))
    return out


def ap50(items, drone_only):
    m = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=drone_only)
    m.update([p for p, _ in items], [t for _, t in items])
    r = m.compute()
    if drone_only:  # per-class AP of the drone class
        idx = r["classes"].tolist().index(CLASSES.index("drone"))
        return float(r["map_per_class"][idx])
    return float(r["map_50"] if "map_50" in r else r["map"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(0)
    result = {"a": args.a, "b": args.b, "resamples": args.resamples}
    for name, folder in SETS.items():
        pa, pb = predictions(args.a, args.size, folder), predictions(args.b, args.size, folder)
        n = len(pa)
        drone = name == "drone"
        diffs = []
        for _ in range(args.resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            diffs.append(ap50([pb[i] for i in idx], drone) - ap50([pa[i] for i in idx], drone))
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
        result[name] = {"images": n, "AP50_a": round(ap50(pa, drone), 3), "AP50_b": round(ap50(pb, drone), 3),
                        "diff_b_minus_a": round(sum(diffs) / len(diffs), 3), "ci95": [round(lo, 3), round(hi, 3)],
                        "contains_zero": lo <= 0 <= hi}
        print(name, result[name], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
