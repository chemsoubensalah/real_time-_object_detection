"""Save a trained YOLOv5 checkpoint so it loads on any Python version.

train.py saves extra things in best.pt (optimizer, training options with
pathlib paths...). A Path object pickled on Python 3.13 cannot be loaded on
Python 3.12 ("No module named 'pathlib._local'"). We keep only the model,
in half precision, which is all that detection needs.

Usage (run from the project folder):
    python scripts/export_weights.py ../yolov5/runs/train/multi/weights/best.pt models/multi_yolov5n.pt
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--yolov5-dir", default="../yolov5")
    args = ap.parse_args()

    sys.path.insert(0, args.yolov5_dir)  # the checkpoint references YOLOv5's model classes
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).half()
    for module in model.modules():  # replace any pathlib object by a plain string
        for key, value in list(vars(module).items()):
            if type(value).__module__.startswith("pathlib"):
                setattr(module, key, str(value))
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    torch.save({"model": model, "epoch": -1, "date": ckpt.get("date")}, args.dst)
    print(f"saved {args.dst} ({os.path.getsize(args.dst) / 1e6:.1f} MB), classes: {model.names}")


if __name__ == "__main__":
    main()
