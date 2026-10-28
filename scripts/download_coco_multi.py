"""Build a multi-class YOLO dataset from COCO 2017 (e.g. person, car, cat, dog).

Class ids follow the order of --classes, so with the default order:
    0 person, 1 car, 2 cat, 3 dog
Extra classes that COCO does not have (e.g. drone) are added afterwards with
scripts/add_yolo_dataset.py using the next free id.

Rules:
- An image is picked for a class when it has at least one instance of that class
  covering >= --min-area of the image.
- Every instance of every chosen class in a picked image is labelled, however small.
  An unlabelled object would teach the model "this is background".
- Images with a crowd annotation of a chosen class are skipped (one box around
  many objects is not a valid target).

Usage:
    python scripts/download_coco_multi.py --classes person car cat dog --per-class 600 --out data/multi_raw
"""
import argparse
import json
import os
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def coco_to_yolo(bbox, img_w, img_h):
    """COCO [x_min, y_min, w, h] in pixels -> YOLO (xc, yc, w, h) normalised to 0..1."""
    x, y, w, h = bbox
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def load_annotations(cache, split):
    os.makedirs(cache, exist_ok=True)
    ann_json = os.path.join(cache, f"instances_{split}.json")
    if not os.path.exists(ann_json):
        zip_path = os.path.join(cache, "annotations_trainval2017.zip")
        if not os.path.exists(zip_path):
            print("Downloading COCO annotations (~250 MB, one time)...")
            urllib.request.urlretrieve(ANN_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            with z.open(f"annotations/instances_{split}.json") as src, open(ann_json, "wb") as dst:
                dst.write(src.read())
    with open(ann_json) as f:
        return json.load(f)


def select_images(coco, class_names, per_class, min_area, seed=0):
    """Return {image_id: [(class_idx, bbox), ...]} with about per_class images for each class."""
    name_to_coco = {c["name"]: c["id"] for c in coco["categories"]}
    missing = [n for n in class_names if n not in name_to_coco]
    if missing:
        raise SystemExit(f"Not COCO classes: {missing}")
    coco_to_idx = {name_to_coco[n]: i for i, n in enumerate(class_names)}
    images = {im["id"]: im for im in coco["images"]}

    boxes, crowd = {}, set()
    big_enough = {i: set() for i in range(len(class_names))}  # class idx -> image ids
    for ann in coco["annotations"]:
        idx = coco_to_idx.get(ann["category_id"])
        if idx is None:
            continue
        if ann.get("iscrowd"):
            crowd.add(ann["image_id"])
            continue
        boxes.setdefault(ann["image_id"], []).append((idx, ann["bbox"]))
        im = images[ann["image_id"]]
        if ann["bbox"][2] * ann["bbox"][3] >= min_area * im["width"] * im["height"]:
            big_enough[idx].add(ann["image_id"])

    rng = random.Random(seed)
    chosen = set()
    # Rarest class first, so common classes (person) do not use up the images of rare ones.
    for idx in sorted(big_enough, key=lambda i: len(big_enough[i])):
        candidates = sorted(big_enough[idx] - crowd - chosen)
        rng.shuffle(candidates)
        chosen.update(candidates[:per_class])
    return {i: boxes[i] for i in chosen}, images


def download(job):
    url, path = job
    if os.path.exists(path):
        return True
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:  # keep going if one image fails
        print("skip", url, e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=["person", "car", "cat", "dog"])
    ap.add_argument("--per-class", type=int, default=600, help="images picked for each class")
    ap.add_argument("--min-area", type=float, default=0.01,
                    help="an image is picked for a class if one instance covers at least this fraction")
    ap.add_argument("--out", default="data/multi_raw")
    ap.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    ap.add_argument("--cache", default="data/coco_cache")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    coco = load_annotations(args.cache, args.split)
    selected, images = select_images(coco, args.classes, args.per_class, args.min_area)

    img_dir, lbl_dir = os.path.join(args.out, "images"), os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    jobs, stems = [], []
    for image_id in sorted(selected):
        im = images[image_id]
        stem = f"coco_{image_id:012d}"
        jobs.append((im["coco_url"], os.path.join(img_dir, stem + ".jpg")))
        stems.append((stem, image_id))
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(pool.map(download, jobs))

    counts = [0] * len(args.classes)
    written = 0
    for (stem, image_id), good in zip(stems, ok):
        if not good:
            continue
        im = images[image_id]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            for idx, b in selected[image_id]:
                f.write("%d %.6f %.6f %.6f %.6f\n" % ((idx,) + coco_to_yolo(b, im["width"], im["height"])))
                counts[idx] += 1
        written += 1
    print(f"Done: {written} images in {args.out}")
    for name, n in zip(args.classes, counts):
        print(f"  {name:8s} {n} boxes")


if __name__ == "__main__":
    main()
