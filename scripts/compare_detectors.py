"""Compare different object-detection algorithms on the same images.

Every detector is wrapped to return boxes in one format:
    boxes [N, 4] (x1, y1, x2, y2 pixels), scores [N], labels [N] in OUR ids
    (0 person, 1 car, 2 cat, 3 dog)
and all of them are scored with the same mAP code (torchmetrics, COCO-style).

Detectors:
  ours-multi        YOLOv5n trained in this project (5 classes)
  yolov5n / yolov5s YOLOv5 pretrained on COCO-80            (one-stage, anchors)
  ssdlite           SSDlite320 + MobileNetV3                (one-stage, mobile)
  retinanet         RetinaNet ResNet50-FPN v2               (one-stage, focal loss)
  fcos              FCOS ResNet50-FPN                       (one-stage, anchor-free)
  frcnn-mobile      Faster R-CNN MobileNetV3-FPN            (two-stage)
  frcnn-r50         Faster R-CNN ResNet50-FPN v2            (two-stage)
  detr              DETR ResNet-50                          (transformer)

Use images none of the pretrained models trained on (COCO val2017).

Usage:
    python scripts/compare_detectors.py --data data/compare_val2017 --out results/compare
"""
import argparse
import glob
import json
import os
import time

import cv2
import torch
from torchmetrics.detection import MeanAveragePrecision

CLASSES = ["person", "car", "cat", "dog"]
COCO80_TO_OURS = {0: 0, 2: 1, 15: 2, 16: 3}    # YOLOv5 COCO index -> ours
COCO91_TO_OURS = {1: 0, 3: 1, 17: 2, 18: 3}    # torchvision / DETR COCO id -> ours


def keep_ours(boxes, scores, labels, mapping):
    """Drop classes we do not evaluate and rename the rest to our ids."""
    keep = [i for i, l in enumerate(labels.tolist()) if int(l) in mapping]
    new_labels = torch.tensor([mapping[int(labels[i])] for i in keep], dtype=torch.int64)
    return boxes[keep], scores[keep], new_labels


# ----------------------------------------------------------------- detectors
class YoloV5:
    def __init__(self, weights, mapping, size, yolov5_dir):
        self.model = torch.hub.load(yolov5_dir, "custom", path=weights, source="local", device="cpu")
        self.model.conf, self.model.max_det = 0.001, 100   # low threshold: mAP needs the whole ranking
        self.mapping, self.size = mapping, size

    def __call__(self, bgr):
        det = self.model(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size=self.size).xyxy[0].cpu()
        return keep_ours(det[:, :4], det[:, 4], det[:, 5].long(), self.mapping)


class TorchvisionDetector:
    def __init__(self, name):
        import torchvision.models.detection as d
        builders = {
            "ssdlite": (d.ssdlite320_mobilenet_v3_large, d.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
            "retinanet": (d.retinanet_resnet50_fpn_v2, d.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
            "fcos": (d.fcos_resnet50_fpn, d.FCOS_ResNet50_FPN_Weights.COCO_V1),
            "frcnn-mobile": (d.fasterrcnn_mobilenet_v3_large_fpn, d.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
            "frcnn-r50": (d.fasterrcnn_resnet50_fpn_v2, d.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
        }
        build, weights = builders[name]
        kwargs = {"box_score_thresh": 0.001} if name.startswith("frcnn") else {"score_thresh": 0.001}
        self.model = build(weights=weights, **kwargs).eval()

    @torch.no_grad()
    def __call__(self, bgr):
        x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        out = self.model([x])[0]
        return keep_ours(out["boxes"], out["scores"], out["labels"], COCO91_TO_OURS)


class Detr:
    def __init__(self):
        from transformers import DetrForObjectDetection, DetrImageProcessor
        self.proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()

    @torch.no_grad()
    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, return_tensors="pt")
        out = self.model(**inputs)
        res = self.proc.post_process_object_detection(out, threshold=0.001, target_sizes=[rgb.shape[:2]])[0]
        return keep_ours(res["boxes"], res["scores"], res["labels"], COCO91_TO_OURS)


def build(name, args):
    if name == "ours-multi":
        mapping = {i: i for i in range(len(CLASSES))}  # our ids already; drone (4) is dropped
        return YoloV5(args.ours, mapping, 416, args.yolov5_dir)
    if name in ("yolov5n", "yolov5s"):
        return YoloV5(os.path.join(args.yolov5_dir, name + ".pt"), COCO80_TO_OURS, 640, args.yolov5_dir)
    if name == "detr":
        return Detr()
    return TorchvisionDetector(name)


# ----------------------------------------------------------------- evaluation
def load_targets(label_path, w, h):
    boxes, labels = [], []
    if os.path.exists(label_path):
        for line in open(label_path):
            c, xc, yc, bw, bh = line.split()
            c, xc, yc, bw, bh = int(c), float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            if c < len(CLASSES):
                boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
                labels.append(c)
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64)}


def evaluate(name, det, images, labels_dir):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    times = []
    for i, path in enumerate(images):
        bgr = cv2.imread(path)
        h, w = bgr.shape[:2]
        t0 = time.perf_counter()
        boxes, scores, labels = det(bgr)
        if i >= 3:  # skip warm-up images in timing
            times.append(time.perf_counter() - t0)
        stem = os.path.splitext(os.path.basename(path))[0]
        target = load_targets(os.path.join(labels_dir, stem + ".txt"), w, h)
        metric.update([{"boxes": boxes.float(), "scores": scores.float(), "labels": labels}], [target])
    r = metric.compute()
    per_class = {CLASSES[int(c)]: round(float(v), 3) for c, v in zip(r["classes"], r["map_per_class"])}
    ms = 1000 * sum(times) / max(len(times), 1)
    return {"model": name, "mAP50": round(float(r["map_50"]), 3), "mAP50_95": round(float(r["map"]), 3),
            "AP_small": round(float(r["map_small"]), 3), "per_class_mAP50_95": per_class,
            "ms_per_image": round(ms, 1), "fps": round(1000 / ms, 1) if ms else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/compare_val2017", help="folder with images/ and labels/")
    ap.add_argument("--models", nargs="+", default=["ours-multi", "yolov5n", "yolov5s", "ssdlite", "retinanet",
                                                     "fcos", "frcnn-mobile", "frcnn-r50", "detr"])
    ap.add_argument("--ours", default="models/multi_yolov5n.pt")
    ap.add_argument("--yolov5-dir", default="../yolov5")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (quick test)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/compare")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    images = sorted(glob.glob(os.path.join(args.data, "images", "*")))
    if args.limit:
        images = images[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    results = []
    for name in args.models:
        print(f"== {name}", flush=True)
        res = evaluate(name, build(name, args), images, os.path.join(args.data, "labels"))
        print(json.dumps(res, ensure_ascii=False), flush=True)
        results.append(res)
        with open(os.path.join(args.out, "compare.json"), "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
