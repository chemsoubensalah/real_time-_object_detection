"""Build a ready-labelled "person" dataset from COCO 2017 in YOLO format.

This gives you hundreds of labelled human images without drawing boxes by hand.
Mix them with your own phone images (labelled with LabelImg / Roboflow).

Usage:
    python scripts/download_coco_person.py --num 1000 --out data/raw

Output:
    data/raw/images/coco_XXXX.jpg
    data/raw/labels/coco_XXXX.txt     (class x_center y_center width height, normalised)
"""
import argparse
import json
import os
import random
import urllib.request
import zipfile

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
PERSON_ID = 1


def coco_to_yolo(bbox, img_w, img_h):
    """COCO [x_min, y_min, w, h] in pixels -> YOLO (xc, yc, w, h) normalised to 0..1."""
    x, y, w, h = bbox
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def select_images(coco, num, min_area, seed=0):
    """Pick images that contain at least one non-crowd person bigger than min_area (fraction of image)."""
    images = {im["id"]: im for im in coco["images"]}
    boxes = {}
    skip = set()
    for ann in coco["annotations"]:
        if ann["category_id"] != PERSON_ID:
            continue
        if ann.get("iscrowd"):
            # Crowd regions are not individual people; skip whole image to keep labels clean.
            skip.add(ann["image_id"])
            continue
        im = images[ann["image_id"]]
        if ann["bbox"][2] * ann["bbox"][3] < min_area * im["width"] * im["height"]:
            continue
        boxes.setdefault(ann["image_id"], []).append(ann["bbox"])
    ids = sorted(i for i in boxes if i not in skip)
    random.Random(seed).shuffle(ids)
    return [(images[i], boxes[i]) for i in ids[:num]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000, help="number of images")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    ap.add_argument("--min-area", type=float, default=0.02,
                    help="ignore tiny people smaller than this fraction of the image")
    ap.add_argument("--cache", default="data/coco_cache")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    ann_json = os.path.join(args.cache, f"instances_{args.split}.json")
    if not os.path.exists(ann_json):
        zip_path = os.path.join(args.cache, "annotations_trainval2017.zip")
        if not os.path.exists(zip_path):
            print("Downloading COCO annotations (~250 MB, one time)...")
            urllib.request.urlretrieve(ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            with z.open(f"annotations/instances_{args.split}.json") as src, open(ann_json, "wb") as dst:
                dst.write(src.read())

    with open(ann_json) as f:
        coco = json.load(f)
    selected = select_images(coco, args.num, args.min_area)

    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for n, (im, bboxes) in enumerate(selected, 1):
        stem = f"coco_{im['id']:012d}"
        img_path = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            try:
                urllib.request.urlretrieve(im["coco_url"], img_path)
            except Exception as e:  # keep going if one image fails
                print("skip", im["coco_url"], e)
                continue
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            for b in bboxes:
                f.write("0 %.6f %.6f %.6f %.6f\n" % coco_to_yolo(b, im["width"], im["height"]))
        if n % 50 == 0:
            print(f"{n}/{len(selected)}")
    print(f"Done: {len(selected)} images in {args.out}")


if __name__ == "__main__":
    main()
