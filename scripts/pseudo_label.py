"""Pseudo-labelling: fill in missing labels with a pretrained model.

The drone dataset only labels drones. Its images also contain people, cars...
that have no box, so training would learn "a person here is background".
We run YOLOv5s pretrained on the 80 COCO classes over those images and add its
confident detections of our other classes to the label files.

Usage:
    python scripts/pseudo_label.py --images data/multi_raw/images --labels data/multi_raw/labels \
        --pattern "drone_*" --weights ../yolov5/yolov5s.pt --conf 0.5
"""
import argparse
import glob
import os

import cv2
import torch

# COCO-80 class index -> our class id (0 person, 1 car, 2 cat, 3 dog)
COCO80_TO_OURS = {0: 0, 2: 1, 15: 2, 16: 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/multi_raw/images")
    ap.add_argument("--labels", default="data/multi_raw/labels")
    ap.add_argument("--pattern", default="drone_*", help="only these images get pseudo-labels")
    ap.add_argument("--weights", default="../yolov5/yolov5s.pt", help="model pretrained on COCO-80")
    ap.add_argument("--yolov5-dir", default="../yolov5")
    ap.add_argument("--conf", type=float, default=0.5, help="keep only confident detections")
    args = ap.parse_args()

    model = torch.hub.load(args.yolov5_dir, "custom", path=args.weights, source="local", device="cpu")
    model.conf = args.conf

    added, touched = 0, 0
    for path in sorted(glob.glob(os.path.join(args.images, args.pattern))):
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        det = model(rgb, size=640).xywhn[0].cpu().tolist()  # normalised xc, yc, w, h, conf, cls
        new = ["%d %.6f %.6f %.6f %.6f" % (COCO80_TO_OURS[int(c)], x, y, w, h)
               for x, y, w, h, _, c in det if int(c) in COCO80_TO_OURS]
        if not new:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.labels, stem + ".txt"), "a") as f:
            f.write("\n".join(new) + "\n")
        added += len(new)
        touched += 1
    print(f"Added {added} pseudo-labels to {touched} images")


if __name__ == "__main__":
    main()
