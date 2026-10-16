"""Split labelled images into train / val / test and write data.yaml for YOLOv5.

Input (YOLO format, one .txt per image with the same name):
    data/raw/images/*.jpg
    data/raw/labels/*.txt

Usage:
    python scripts/split_dataset.py --src data/raw --dst data/dataset --names person
    python scripts/split_dataset.py --names upper_body full_body   # two classes
"""
import argparse
import os
import random
import shutil

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(src):
    img_dir, lbl_dir = os.path.join(src, "images"), os.path.join(src, "labels")
    pairs, missing = [], 0
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXT:
            continue
        lbl = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lbl):
            pairs.append((os.path.join(img_dir, name), lbl))
        else:
            missing += 1
    return pairs, missing


def split(pairs, train, val, seed):
    pairs = pairs[:]
    random.Random(seed).shuffle(pairs)
    n_train = int(len(pairs) * train)
    n_val = int(len(pairs) * val)
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", default="data/dataset")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1, help="test gets the rest")
    ap.add_argument("--names", nargs="+", default=["person"], help="class names in label-id order")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs, missing = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f"No labelled images found in {args.src}/images + {args.src}/labels")
    if missing:
        print(f"warning: {missing} images have no label file and were skipped")

    for part, items in split(pairs, args.train, args.val, args.seed).items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, sub, part), exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, os.path.join(args.dst, "images", part))
            shutil.copy2(lbl, os.path.join(args.dst, "labels", part))
        print(f"{part:5s}: {len(items)} images")

    yaml_path = os.path.join(args.dst, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.dst)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(args.names)}\n")
        f.write("names: [" + ", ".join(f"'{n}'" for n in args.names) + "]\n")
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
