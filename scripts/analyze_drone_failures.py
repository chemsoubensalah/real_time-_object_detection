"""Where does the model miss drones? Recall by drone SIZE and by image BRIGHTNESS, for several inference modes.

For every ground-truth drone we check whether a predicted drone box with confidence >= --conf overlaps it with
IoU >= 0.5. Sizes are in pixels of the 640x640 image (sqrt of the box area):
    tiny < 16 | small 16-32 | medium 32-96 | large >= 96
Brightness = mean gray level of the image: dark < 70, normal >= 70.
Also reports false drone boxes per image, and AP@0.5 per size (torchmetrics) for the whole drone set.

Modes (comma separated), e.g.  plain:512,plain:800,tiles2:512,tiles3:512
    plain:S     one pass at input size S
    tilesG:S    GxG overlapping tiles + the full frame, tiles enlarged to S (see tiled_inference.py)

Usage:
    python scripts/analyze_drone_failures.py --weights models/v3_yolo11n_12cls.pt --folder data/v2_tune_drone \
        --modes plain:512,tiles2:512 --out results/v4/failures_tune.json
"""
import argparse
import glob
import json
import os
import time

import cv2
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

import detect_realtime as dr
from tiled_inference import detect_tiled

DRONE = 11
BUCKETS = [("tiny <16", 0, 16), ("small 16-32", 16, 32), ("medium 32-96", 32, 96), ("large >=96", 96, 1e9)]


def bucket_of(box):
    s = float(((box[2] - box[0]) * (box[3] - box[1])) ** 0.5)
    return next(name for name, lo, hi in BUCKETS if lo <= s < hi)


def read_gt(path, w, h):
    boxes = []
    for line in open(path):
        p = line.split()
        if len(p) == 5 and int(p[0]) == DRONE:
            xc, yc, bw, bh = (float(v) for v in p[1:])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def make_predictor(model, mode):
    kind, size = mode.split(":")
    size = int(size)
    if kind == "plain":
        return lambda frame: dr.detect(model, frame, size)
    grid = int(kind.replace("tiles", ""))
    return lambda frame: detect_tiled(model, frame, size, grid=grid)


def analyze(model, mode, folder, conf):
    predict = make_predictor(model, mode)
    found = {b[0]: [0, 0] for b in BUCKETS}  # bucket -> [found, total]
    light = {"dark (<70)": [0, 0], "normal (>=70)": [0, 0]}
    fp, n_images, times = 0, 0, []
    ap_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    for path in sorted(glob.glob(os.path.join(folder, "images", "*"))):
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        gt = read_gt(os.path.join(folder, "labels", os.path.splitext(os.path.basename(path))[0] + ".txt"), w, h)
        t0 = time.perf_counter()
        det = predict(frame)
        times.append(time.perf_counter() - t0)
        det = det[det[:, 5] == DRONE]
        ap_metric.update([{"boxes": det[:, :4], "scores": det[:, 4], "labels": torch.zeros(len(det), dtype=torch.int64)}],
                         [{"boxes": gt, "labels": torch.zeros(len(gt), dtype=torch.int64)}])
        sure = det[det[:, 4] >= conf]
        matched = torch.zeros(len(gt), dtype=torch.bool)
        used = torch.zeros(len(sure), dtype=torch.bool)
        if len(gt) and len(sure):
            ious = box_iou(sure[:, :4], gt)
            for i in sure[:, 4].argsort(descending=True).tolist():  # best predictions first
                cand = ious[i].clone()
                cand[matched] = -1
                j = int(cand.argmax())
                if cand[j] >= 0.5:
                    matched[j], used[i] = True, True
        fp += int((~used).sum())
        dark = "dark (<70)" if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < 70 else "normal (>=70)"
        for k, box in enumerate(gt.tolist()):
            b = bucket_of(box)
            found[b][0] += int(matched[k]); found[b][1] += 1
            light[dark][0] += int(matched[k]); light[dark][1] += 1
        n_images += 1
    r = ap_metric.compute()
    rec = lambda d: {k: {"found": v[0], "total": v[1], "recall": round(v[0] / v[1], 3) if v[1] else None} for k, v in d.items()}  # noqa: E731
    total_found, total = sum(v[0] for v in found.values()), sum(v[1] for v in found.values())
    return {"mode": mode, "recall_by_size": rec(found), "recall_by_light": rec(light), "recall_all": round(total_found / total, 3),
            "false_boxes_per_image": round(fp / n_images, 3), "AP50_all": round(float(r["map"]), 3),
            "AP50_small(<32px)": round(float(r["map_small"]), 3), "AP50_medium": round(float(r["map_medium"]), 3), "AP50_large": round(float(r["map_large"]), 3),
            "ms_per_image": round(1000 * sum(times[2:]) / max(len(times) - 2, 1), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/v3_yolo11n_12cls.pt")
    ap.add_argument("--folder", default="data/v2_tune_drone")
    ap.add_argument("--modes", default="plain:512")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(4)
    model = dr.load_model(args.weights, 0.001, "cpu", "../yolov5")
    results = []
    for mode in args.modes.split(","):
        res = analyze(model, mode, args.folder, args.conf)
        results.append(res)
        print(f"{mode:12} recall all={res['recall_all']}  " + "  ".join(f"{k.split()[0]}={v['recall']}({v['found']}/{v['total']})" for k, v in res["recall_by_size"].items())
              + f"  | dark={res['recall_by_light']['dark (<70)']['recall']}({res['recall_by_light']['dark (<70)']['found']}/{res['recall_by_light']['dark (<70)']['total']})"
              + f"  | FP/img={res['false_boxes_per_image']}  AP50={res['AP50_all']} small={res['AP50_small(<32px)']}  {res['ms_per_image']}ms", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"weights": args.weights, "folder": args.folder, "conf": args.conf, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
