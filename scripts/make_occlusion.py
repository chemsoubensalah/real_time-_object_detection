"""Create synthetically occluded copies of a YOLO dataset split.

For every object we cover a band of the box from one random side (left, right,
top or bottom) with a patch cut from another image, so it looks like the object
is behind something. The label becomes the VISIBLE part of the object only:

    before:  +---------+          after:  +----+####+
             | person  |                  |vis.|occl|
             +---------+                  +----+####+

Boxes of other objects that the patch also covers are shrunk the same way, or
removed when nothing of them is left visible.

Usage (test set, 40% occlusion):
    python scripts/make_occlusion.py --src data/multi_dataset --split test --ratio 0.4 \
        --dst data/occluded/test_40
Usage (training augmentation, random 20-60%, half the images):
    python scripts/make_occlusion.py --src data/multi_dataset --split train --ratio-min 0.2 \
        --ratio-max 0.6 --fraction 0.5 --dst data/occ_train_extra
"""
import argparse
import glob
import os
import random

import cv2

SIDES = ("left", "right", "top", "bottom")
MIN_VISIBLE_PX = 4  # a box thinner than this is considered fully hidden


def read_labels(path, w, h):
    """YOLO lines -> list of [cls, x1, y1, x2, y2] in pixels."""
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) != 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
            boxes.append([c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return boxes


def write_labels(path, boxes, w, h):
    with open(path, "w") as f:
        for c, x1, y1, x2, y2 in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))


def occluder_rect(box, side, ratio):
    """Rectangle covering `ratio` of the box, starting from `side`."""
    _, x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if side == "left":
        return x1, y1, x1 + bw * ratio, y2
    if side == "right":
        return x2 - bw * ratio, y1, x2, y2
    if side == "top":
        return x1, y1, x2, y1 + bh * ratio
    return x1, y2 - bh * ratio, x2, y2


def visible_part(box, occ):
    """Shrink `box` to what `occ` leaves visible. None if fully hidden.

    The box is only shrunk when the occluder spans its whole height (or width);
    a patch over a corner leaves the visible extent of the object unchanged.
    """
    c, x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = occ
    if ox2 <= x1 or ox1 >= x2 or oy2 <= y1 or oy1 >= y2:
        return box  # no overlap
    spans_h = oy1 <= y1 and oy2 >= y2
    spans_w = ox1 <= x1 and ox2 >= x2
    if spans_h and spans_w:
        return None  # completely covered
    if spans_h:
        if ox1 <= x1:
            x1 = max(x1, ox2)  # covered from the left
        elif ox2 >= x2:
            x2 = min(x2, ox1)  # covered from the right
    elif spans_w:
        if oy1 <= y1:
            y1 = max(y1, oy2)  # covered from the top
        elif oy2 >= y2:
            y2 = min(y2, oy1)  # covered from the bottom
    if x2 - x1 < MIN_VISIBLE_PX or y2 - y1 < MIN_VISIBLE_PX:
        return None
    return [c, x1, y1, x2, y2]


def paste_patch(img, rect, donor, rng):
    """Fill `rect` of img with a random crop of `donor` (realistic texture)."""
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return
    dh, dw = donor.shape[:2]
    if dw < pw or dh < ph:
        donor = cv2.resize(donor, (max(dw, pw), max(dh, ph)))
        dh, dw = donor.shape[:2]
    sx, sy = rng.randint(0, dw - pw), rng.randint(0, dh - ph)
    img[y1:y2, x1:x2] = donor[sy:sy + ph, sx:sx + pw]


def occlude_image(img, boxes, ratio_min, ratio_max, donor, rng):
    h, w = img.shape[:2]
    boxes = [b[:] for b in boxes]
    for i in range(len(boxes)):
        target = boxes[i]
        if target is None:
            continue
        ratio = rng.uniform(ratio_min, ratio_max)
        occ = occluder_rect(target, rng.choice(SIDES), ratio)
        paste_patch(img, occ, donor, rng)
        boxes = [None if b is None else visible_part(b, occ) for b in boxes]
    return img, [b for b in boxes if b is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/multi_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ratio", type=float, help="fixed occlusion ratio (sets min = max)")
    ap.add_argument("--ratio-min", type=float, default=0.2)
    ap.add_argument("--ratio-max", type=float, default=0.6)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of images to occlude")
    ap.add_argument("--prefix", default="", help="prefix for output file names")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.ratio is not None:
        args.ratio_min = args.ratio_max = args.ratio

    rng = random.Random(args.seed)
    img_paths = sorted(glob.glob(os.path.join(args.src, "images", args.split, "*")))
    out_img, out_lbl = os.path.join(args.dst, "images"), os.path.join(args.dst, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    chosen = [p for p in img_paths if rng.random() < args.fraction]
    kept_boxes = removed = 0
    for path in chosen:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(path))[0]
        boxes = read_labels(os.path.join(args.src, "labels", args.split, stem + ".txt"), w, h)
        donor = cv2.imread(rng.choice(img_paths))  # texture source: another image of the set
        img, new_boxes = occlude_image(img, boxes, args.ratio_min, args.ratio_max, donor, rng)
        removed += len(boxes) - len(new_boxes)
        kept_boxes += len(new_boxes)
        cv2.imwrite(os.path.join(out_img, args.prefix + stem + ".jpg"), img)
        write_labels(os.path.join(out_lbl, args.prefix + stem + ".txt"), new_boxes, w, h)
    print(f"{len(chosen)} images -> {args.dst}  ({kept_boxes} visible boxes, {removed} fully hidden)")


if __name__ == "__main__":
    main()
