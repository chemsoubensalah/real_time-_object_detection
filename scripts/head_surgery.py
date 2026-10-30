"""Build a model with a different set of classes from a pretrained one by keeping only some class rows.

A YOLO11 detection head ends with one 1x1 convolution per scale that has one output row per class.
NOTE: when you train with a different class list, ultralytics already copies the rows of classes whose NAMES
match (its log says "Remapped 11/12 cls head rows from pretrained weights by class name"), so for those classes
this script changes nothing compared with a normal training run. What it adds:
  * a NEW class can start from a similar pretrained class: "drone=kite" copies the kite row to the drone row
    (a normal run starts the 12th class from random numbers);
  * the result can be evaluated BEFORE any training, which gives a zero-shot baseline.

Backbone, neck, box branch and shared class features are untouched, so the result behaves exactly like the
pretrained model on the kept classes.

Usage:
    python scripts/head_surgery.py --src yolo11n.pt --out models/yolo11n_12cls_init.pt \
        --classes person bicycle motorcycle car bus truck airplane bird cat dog horse drone=kite
"""
import argparse
import datetime
import os

import torch
import torch.nn as nn
from ultralytics import YOLO


def prune_head(model, classes):
    """classes: list of 'name' or 'new_name=source_name'. Modifies `model` in place."""
    src_ids = {n: i for i, n in model.names.items()}
    new_names, rows = [], []
    for c in classes:
        new, _, source = c.partition("=")
        source = source or new
        if source not in src_ids:
            raise SystemExit(f"'{source}' is not a class of the source model")
        new_names.append(new)
        rows.append(src_ids[source])
    det = model.model[-1]
    for seq in det.cv3:  # one classification branch per scale; its last layer has one row per class
        old = seq[-1]
        layer = nn.Conv2d(old.in_channels, len(rows), kernel_size=1)
        layer.weight.data = old.weight.data[rows].clone()
        layer.bias.data = old.bias.data[rows].clone()
        seq[-1] = layer
    det.nc = len(rows)
    det.no = det.nc + det.reg_max * 4
    model.names = dict(enumerate(new_names))
    model.nc = len(rows)
    model.yaml["nc"] = len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="yolo11n.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    args = ap.parse_args()

    model = YOLO(args.src).model.float()
    rows = prune_head(model, args.classes)
    ckpt = {"epoch": -1, "best_fitness": None, "model": model.half(), "ema": None, "updates": None, "optimizer": None,
            "train_args": {}, "date": datetime.datetime.now().isoformat()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"saved {args.out}: {len(rows)} classes, source rows {rows}")


if __name__ == "__main__":
    main()
