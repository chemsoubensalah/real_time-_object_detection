"""Build a smaller-class-count model from a pretrained one by keeping only some class rows.

A YOLO11 detection head ends with one 1x1 convolution per scale that has one output row per class.
Changing the number of classes normally re-creates these layers with RANDOM numbers, throwing away
what the pretrained model learned about e.g. "person" or "car". Here we instead COPY the rows of the
classes we want, in our own order. A new class can start from a similar pretrained class:
"drone=kite" means the drone row starts as a copy of the kite row.

Everything else (backbone, neck, box branch, shared class features) is untouched, so the result
behaves exactly like the pretrained model on the kept classes, before any fine-tuning.

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
