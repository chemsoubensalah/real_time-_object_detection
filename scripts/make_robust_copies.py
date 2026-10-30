"""Offline augmentation: darkened and noisy copies of training images, to teach the model hard conditions.

Our analysis showed the model misses drones most often in dark images and in images with speckle noise,
and that such images are rare in training (about 6% dark). Each copy gets ONE of:
    dark   gamma darkening + lower brightness + sensor-like Gaussian noise (night / low light)
    noise  salt-and-pepper speckles (+ a little Gaussian noise)
    both   dark and speckles
Boxes do not move, so the label file is copied unchanged.

Usage:
    python scripts/make_robust_copies.py --src data/v2_dataset/images/train --pattern "drone_*" --fraction 0.5 \
        --out data/v4_extra --prefix rob_
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np


def darken(img, rng):
    gamma = rng.uniform(1.8, 3.2)                       # >1 pushes mid-tones to black
    out = 255.0 * (img / 255.0) ** gamma * rng.uniform(0.45, 0.85)
    out += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 10), img.shape)  # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)


def speckle(img, rng):
    g = np.random.default_rng(rng.randrange(1 << 30))
    out = img.astype(np.float32) + g.normal(0, rng.uniform(0, 6), img.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    amount = rng.uniform(0.01, 0.05)                    # 1-5% of pixels become black or white
    mask = g.random(img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="images folder; labels are found by replacing 'images' with 'labels'")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--fraction", type=float, default=0.5, help="fraction of the matching images that get a copy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="rob_")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.src, args.pattern)))
    chosen = [p for p in paths if rng.random() < args.fraction]
    kinds = {"dark": 0, "noise": 0, "both": 0}
    for p in chosen:
        stem = os.path.splitext(os.path.basename(p))[0]
        lbl = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl):
            continue
        img = cv2.imread(p)
        kind = rng.choice(["dark", "dark", "noise", "both"])  # darkness was the weakest group, so it is chosen more often
        if kind in ("dark", "both"):
            img = darken(img, rng)
        if kind in ("noise", "both"):
            img = speckle(img, rng)
        kinds[kind] += 1
        name = f"{args.prefix}{kind}_{stem}"
        cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy2(lbl, os.path.join(args.out, "labels", name + ".txt"))
    print(f"{sum(kinds.values())} copies from {len(paths)} images -> {args.out}  {kinds}")


if __name__ == "__main__":
    main()
