"""Build the version-2 dataset: 12 classes from COCO + the Seraphim drone dataset.

Classes (id = position in CLASSES):
    0 person 1 bicycle 2 motorcycle 3 car 4 bus 5 truck 6 airplane 7 bird 8 cat 9 dog 10 horse 11 drone

Steps (run in this order; each can be re-run):
  coco     COCO train2017 images for the 11 COCO classes + background images (no target class at all)
  drones   drone images from the Seraphim dataset, relabelled to class 11
  pseudo   fill missing labels in the drone images (people, cars...) with a pretrained COCO model
  split    train / val split and data.yaml
  tests    two held-out test sets the old model never saw:
             data/v2_test_coco   COCO val2017 (11 COCO classes)
             data/v2_test_drone  drones taken from the Seraphim test split, drone label only

Usage:
    python scripts/build_dataset_v2.py coco --per-class 400
    python scripts/build_dataset_v2.py drones --zip-images X.zip --zip-labels Y.zip --num 900
    python scripts/build_dataset_v2.py pseudo
    python scripts/build_dataset_v2.py split
    python scripts/build_dataset_v2.py tests --seraphim-test-images A --seraphim-test-labels B
"""
import argparse
import glob
import os
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

import add_yolo_dataset
import download_coco_multi as dcm

CLASSES = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "airplane", "bird", "cat", "dog", "horse", "drone"]
COCO_CLASSES = CLASSES[:-1]
DRONE_ID = CLASSES.index("drone")
# Classes the teacher model may add to drone images. Birds and airplanes are excluded on purpose:
# the teacher would label the drone itself as "bird"/"airplane".
PSEUDO_CLASSES = [c for c in COCO_CLASSES if c not in ("bird", "airplane")]
RAW = "data/v2_raw"


def cmd_coco(args):
    coco = dcm.load_annotations(args.cache, "train2017")
    selected, images = dcm.select_images(coco, COCO_CLASSES, args.per_class, args.min_area)
    wanted = {c["id"] for c in coco["categories"] if c["name"] in COCO_CLASSES}
    has_target = {a["image_id"] for a in coco["annotations"] if a["category_id"] in wanted}
    pool = sorted(i for i in images if i not in has_target)
    random.Random(1).shuffle(pool)
    backgrounds = pool[:args.background]

    img_dir, lbl_dir = os.path.join(RAW, "images"), os.path.join(RAW, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    jobs = [(images[i]["coco_url"], os.path.join(img_dir, f"coco_{i:012d}.jpg")) for i in list(selected) + backgrounds]
    with ThreadPoolExecutor(args.workers) as pool_:
        ok = list(pool_.map(dcm.download, jobs))
    ids = list(selected) + backgrounds
    counts = [0] * len(CLASSES)
    for image_id, good in zip(ids, ok):
        if not good:
            continue
        im = images[image_id]
        with open(os.path.join(lbl_dir, f"coco_{image_id:012d}.txt"), "w") as f:  # empty file = background image
            for idx, b in selected.get(image_id, []):
                f.write("%d %.6f %.6f %.6f %.6f\n" % ((idx,) + dcm.coco_to_yolo(b, im["width"], im["height"])))
                counts[idx] += 1
    print(f"{len(selected)} COCO images + {len(backgrounds)} background images")
    print({c: n for c, n in zip(CLASSES, counts) if n})


def cmd_drones(args):
    work = args.work
    os.makedirs(work, exist_ok=True)
    for z, sub in ((args.zip_images, "images"), (args.zip_labels, "labels")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(os.path.join(work, sub))
    src_img, src_lbl = os.path.join(work, "images"), os.path.join(work, "labels")
    add_yolo_dataset_main(src_img, src_lbl, args.num, args.prefix, args.seed, args.skip)
    shutil.rmtree(work)  # free the disk: only the sampled images are kept


def add_yolo_dataset_main(src_images, src_labels, num, prefix, seed, skip=0):
    """Copy `num` valid drone images (class rewritten to DRONE_ID) into RAW,
    after skipping the first `skip` valid ones of the same shuffled order (already used by an earlier run)."""
    img_out, lbl_out = os.path.join(RAW, "images"), os.path.join(RAW, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    names = sorted(n for n in os.listdir(src_images) if os.path.splitext(n)[1].lower() in add_yolo_dataset.IMG_EXT)
    random.Random(seed).shuffle(names)
    added = skipped = 0
    for name in names:
        if added >= num:
            break
        stem, ext = os.path.splitext(name)
        lbl = os.path.join(src_labels, stem + ".txt")
        lines = add_yolo_dataset.read_label(lbl, DRONE_ID) if os.path.exists(lbl) else None
        if not lines:
            continue
        if skipped < skip:
            skipped += 1
            continue
        new_stem = f"{prefix}_{added:05d}"
        shutil.copy2(os.path.join(src_images, name), os.path.join(img_out, new_stem + ext.lower()))
        with open(os.path.join(lbl_out, new_stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        added += 1
    print(f"added {added} drone images")


def iou_and_cover(a, b):
    """a, b = (x1, y1, x2, y2). Returns (IoU, intersection / area of a)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return (inter / union if union else 0.0), (inter / area_a if area_a else 0.0)


def cmd_pseudo(args):
    from ultralytics import YOLO
    model = YOLO(args.teacher)
    name_to_id = {n: i for i, n in model.names.items()}
    teacher_ids = [name_to_id[c] for c in PSEUDO_CLASSES]
    ours = {name_to_id[c]: CLASSES.index(c) for c in PSEUDO_CLASSES}
    paths = sorted(glob.glob(os.path.join(RAW, "images", args.pattern)))
    added = touched = 0
    def predictions():
        for i in range(0, len(paths), 8):  # chunks of 8: a list source would otherwise become one giant batch
            yield from model.predict(paths[i:i + 8], conf=args.conf, imgsz=640, device="cpu", classes=teacher_ids, verbose=False)

    for r in predictions():
        lbl_path = os.path.join(RAW, "labels", os.path.splitext(os.path.basename(r.path))[0] + ".txt")
        drones = []
        for line in open(lbl_path):
            p = line.split()
            xc, yc, w, h = map(float, p[1:])
            drones.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        new = []
        for (x, y, w, h), c in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist()):
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if any(max(iou_and_cover(box, d)[0], iou_and_cover(box, d)[1] * 0.5) > 0.1 for d in drones):
                continue  # overlaps a drone: probably the teacher mislabelling the drone itself
            new.append("%d %.6f %.6f %.6f %.6f" % (ours[int(c)], x, y, w, h))
        if new:
            with open(lbl_path, "a") as f:
                f.write("\n".join(new) + "\n")
            added += len(new)
            touched += 1
    print(f"pseudo-labels: {added} boxes in {touched} of {len(paths)} drone images")


def cmd_split(args):
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(RAW, "images", "*")))
    random.Random(42).shuffle(names)
    n_val = int(len(names) * args.val)
    parts = {"val": names[:n_val], "train": names[n_val:]}
    ext = {os.path.splitext(os.path.basename(p))[0]: os.path.splitext(p)[1] for p in glob.glob(os.path.join(RAW, "images", "*"))}
    dst = "data/v2_dataset"
    for part, stems in parts.items():
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(dst, sub, part), exist_ok=True)
        for s in stems:
            shutil.copy2(os.path.join(RAW, "images", s + ext[s]), os.path.join(dst, "images", part))
            shutil.copy2(os.path.join(RAW, "labels", s + ".txt"), os.path.join(dst, "labels", part))
    with open(os.path.join(dst, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(dst)}\ntrain: images/train\nval: images/val\nnc: {len(CLASSES)}\n")
        f.write("names: [" + ", ".join(f"'{c}'" for c in CLASSES) + "]\n")
    print({k: len(v) for k, v in parts.items()}, "->", dst)
    per = [0] * len(CLASSES)
    for p in glob.glob(os.path.join(dst, "labels", "train", "*.txt")):
        for line in open(p):
            if line.strip():
                per[int(line.split()[0])] += 1
    print("train boxes:", dict(zip(CLASSES, per)))


def cmd_tests(args):
    # COCO val2017: images the old model (trained on train2017) never saw
    coco = dcm.load_annotations(args.cache, "val2017")
    selected, images = dcm.select_images(coco, COCO_CLASSES, args.per_class, args.min_area, seed=7)
    out = "data/v2_test_coco"
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    jobs = [(images[i]["coco_url"], os.path.join(out, "images", f"coco_{i:012d}.jpg")) for i in selected]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(dcm.download, jobs))
    for i, boxes in selected.items():
        im = images[i]
        with open(os.path.join(out, "labels", f"coco_{i:012d}.txt"), "w") as f:
            for idx, b in boxes:
                f.write("%d %.6f %.6f %.6f %.6f\n" % ((idx,) + dcm.coco_to_yolo(b, im["width"], im["height"])))
    print(f"{out}: {len(selected)} images")

    # Drones from the Seraphim TEST split, skipping the first 700 shuffled names the old model used
    names = sorted(n for n in os.listdir(args.seraphim_test_images) if os.path.splitext(n)[1].lower() in add_yolo_dataset.IMG_EXT)
    random.Random(0).shuffle(names)
    fresh = names[args.old_used:args.old_used + args.num_drone]
    out = "data/v2_test_drone"
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    kept = 0
    for name in fresh:
        stem, e = os.path.splitext(name)
        lines = add_yolo_dataset.read_label(os.path.join(args.seraphim_test_labels, stem + ".txt"), DRONE_ID)
        if not lines:
            continue
        shutil.copy2(os.path.join(args.seraphim_test_images, name), os.path.join(out, "images", f"drone_{kept:05d}{e.lower()}"))
        with open(os.path.join(out, "labels", f"drone_{kept:05d}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        kept += 1
    print(f"{out}: {kept} images (drone label only)")


def main():
    global RAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW, help="raw folder (images/ + labels/) used by coco, drones, pseudo and split")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("coco")
    p.add_argument("--per-class", type=int, default=400)
    p.add_argument("--background", type=int, default=250)
    p.add_argument("--min-area", type=float, default=0.01)
    p.add_argument("--cache", default="data/coco_cache")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(fn=cmd_coco)
    p = sub.add_parser("drones")
    p.add_argument("--zip-images", required=True)
    p.add_argument("--zip-labels", required=True)
    p.add_argument("--num", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work", default="data/_drone_extract")
    p.add_argument("--prefix", default="drone", help="file name prefix of the copied images")
    p.add_argument("--skip", type=int, default=0, help="skip the first N valid images (used by an earlier run)")
    p.set_defaults(fn=cmd_drones)
    p = sub.add_parser("pseudo")
    p.add_argument("--teacher", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--pattern", default="drone_*", help="which images of the raw folder get pseudo-labels")
    p.set_defaults(fn=cmd_pseudo)
    p = sub.add_parser("split")
    p.add_argument("--val", type=float, default=0.08)
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("tests")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--min-area", type=float, default=0.01)
    p.add_argument("--cache", default="data/coco_cache")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seraphim-test-images", required=True)
    p.add_argument("--seraphim-test-labels", required=True)
    p.add_argument("--old-used", type=int, default=700, help="images of the shuffled test split used by the old model")
    p.add_argument("--num-drone", type=int, default=300)
    p.set_defaults(fn=cmd_tests)
    args = ap.parse_args()
    RAW = args.raw
    args.fn(args)


if __name__ == "__main__":
    main()
