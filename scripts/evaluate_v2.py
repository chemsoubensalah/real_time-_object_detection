"""Evaluate several models on the held-out v2 test sets with the same code.

Test sets (made by build_dataset_v2.py, none was used for training any model compared here):
    data/v2_test_coco    COCO val2017, the 11 COCO classes
    data/v2_test_drone   drones from the Seraphim test split (drone label only)

Every model is evaluated only on the classes it can predict, so a model is never
punished for a class it was not trained on. Class ids are matched by NAME.
Reported per class: AP@0.5 and AP@0.5:0.95; plus the mean over the 4 classes all
models share (person, car, cat, dog) and over every class a model supports.

Usage:
    python scripts/evaluate_v2.py --models old=models/multi_yolov5n.pt@416 new=runs/v2/weights/best.pt@512 \
        coco11=yolo11n.pt@640 --out results/v2/evaluation.json
    (name=path@image_size)
"""
import argparse
import glob
import json
import os
import time

import cv2
import torch
from torchmetrics.detection import MeanAveragePrecision

import detect_realtime as dr

CLASSES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "airplane", "bird", "cat", "dog", "horse", "drone"]
SHARED = ["person", "car", "cat", "dog"]
TEST_SETS = {"coco": "data/v2_test_coco", "drone": "data/v2_test_drone"}


def read_gt(path, w, h):
    boxes, labels = [], []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) != 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *(float(v) for v in p[1:])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
            labels.append(c)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64)


def run_model(spec, images_by_set, limit):
    name, rest = spec.split("=", 1)
    weights, size = rest.rsplit("@", 1)
    size = int(size)
    model = dr.load_model(weights, 0.001, "cpu", "../yolov5")  # very low threshold: AP needs the whole ranking
    to_ours = {i: CLASSES.index(n) for i, n in model.names.items() if n in CLASSES}
    supported = set(to_ours.values())
    result = {"supported": [CLASSES[i] for i in sorted(supported)], "size": size, "sets": {}}
    times = []
    for set_name, folder in TEST_SETS.items():
        images = images_by_set[set_name][:limit] if limit else images_by_set[set_name]
        m_all = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        m_50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
        evaluated = set()
        for path in images:
            frame = cv2.imread(path)
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            det = dr.detect(model, frame, size)
            times.append(time.perf_counter() - t0)
            boxes, scores, labels = det[:, :4], det[:, 4], det[:, 5].long()
            keep = [k for k, c in enumerate(labels.tolist()) if c in to_ours]
            boxes, scores = boxes[keep], scores[keep]
            labels = torch.tensor([to_ours[int(labels[k])] for k in keep], dtype=torch.int64)
            gb, gl = read_gt(os.path.join(folder, "labels", os.path.splitext(os.path.basename(path))[0] + ".txt"), w, h)
            gk = [k for k, c in enumerate(gl.tolist()) if c in supported]
            gb, gl = gb[gk], gl[gk]
            evaluated.update(gl.tolist())
            pred = {"boxes": boxes, "scores": scores, "labels": labels}
            gt = {"boxes": gb, "labels": gl}
            m_all.update([pred], [gt])
            m_50.update([pred], [gt])
        r_all, r_50 = m_all.compute(), m_50.compute()
        per = {}
        for c, a, b in zip(r_all["classes"].tolist(), r_all["map_per_class"].tolist(), r_50["map_per_class"].tolist()):
            per[CLASSES[c]] = {"AP50": round(b, 3), "AP50_95": round(a, 3)}
        result["sets"][set_name] = {"images": len(images), "per_class": per,
                                    "mAP50": round(float(r_50["map"]), 3), "mAP50_95": round(float(r_all["map"]), 3)}
    result["ms_per_image"] = round(1000 * sum(times[3:]) / max(len(times) - 3, 1), 1)
    return name, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path@image_size")
    ap.add_argument("--out", default="results/v2/evaluation.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images of each set (quick test)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--coco-dir", default=TEST_SETS["coco"], help="folder with images/ and labels/ (COCO classes)")
    ap.add_argument("--drone-dir", default=TEST_SETS["drone"], help="folder with images/ and labels/ (drone label only)")
    args = ap.parse_args()
    TEST_SETS["coco"], TEST_SETS["drone"] = args.coco_dir, args.drone_dir
    torch.set_num_threads(args.threads)
    images_by_set = {s: sorted(glob.glob(os.path.join(d, "images", "*"))) for s, d in TEST_SETS.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for spec in args.models:
        print("==", spec, flush=True)
        name, res = run_model(spec, images_by_set, args.limit)
        results[name] = res
        print(json.dumps({k: v for k, v in res.items() if k != "sets"}), flush=True)
        for s, v in res["sets"].items():
            print(f"   {s}: mAP50={v['mAP50']} mAP50-95={v['mAP50_95']} ({v['images']} images)", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
