"""Add images from another YOLO-format dataset as a new class (e.g. drones).

The source dataset has its own class ids (usually 0). They are rewritten to
--class-id so the new class fits after the COCO classes.

Usage:
    python scripts/add_yolo_dataset.py --src-images X/images --src-labels X/labels \
        --class-id 4 --num 700 --prefix drone --out data/multi_raw
"""
import argparse
import os
import random
import shutil

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def read_label(path, class_id):
    """Return label lines with the class replaced, or None if the file is malformed."""
    lines = []
    with open(path) as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            if len(parts) != 5:
                return None
            try:
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                return None
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w == 0 or h == 0:
                return None
            lines.append("%d %.6f %.6f %.6f %.6f" % (class_id, xc, yc, w, h))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-images", required=True)
    ap.add_argument("--src-labels", required=True)
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--num", type=int, default=700)
    ap.add_argument("--prefix", default="extra")
    ap.add_argument("--out", default="data/multi_raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_out, lbl_out = os.path.join(args.out, "images"), os.path.join(args.out, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    names = sorted(n for n in os.listdir(args.src_images) if os.path.splitext(n)[1].lower() in IMG_EXT)
    random.Random(args.seed).shuffle(names)

    added = skipped = boxes = 0
    for name in names:
        if added >= args.num:
            break
        stem, ext = os.path.splitext(name)
        lbl = os.path.join(args.src_labels, stem + ".txt")
        lines = read_label(lbl, args.class_id) if os.path.exists(lbl) else None
        if not lines:  # no label file, empty (no object) or malformed
            skipped += 1
            continue
        new_stem = f"{args.prefix}_{added:05d}"
        shutil.copy2(os.path.join(args.src_images, name), os.path.join(img_out, new_stem + ext.lower()))
        with open(os.path.join(lbl_out, new_stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        added += 1
        boxes += len(lines)
    print(f"Added {added} images ({boxes} boxes) as class {args.class_id}; skipped {skipped}")


if __name__ == "__main__":
    main()
