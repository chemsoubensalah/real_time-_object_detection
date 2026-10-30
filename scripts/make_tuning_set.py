"""Build a small TUNING set, separate from the test sets, to choose hyper-parameters (e.g. the mix ratio alpha).

Choosing a value on the test set and then reporting the test score would be optimistic, because the value
was picked to look good on exactly those images. So we pick on this tuning set and report on the untouched test sets.

    data/v2_tune_coco    COCO val2017 images that are NOT in data/v2_test_coco
    data/v2_tune_drone   drones from the Seraphim test split, after the 700 used by the old model and the 300 of the test set

Usage:
    python scripts/make_tuning_set.py --seraphim-test-images A --seraphim-test-labels B
"""
import argparse
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

import add_yolo_dataset
import build_dataset_v2 as bd
import download_coco_multi as dcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seraphim-test-images", required=True)
    ap.add_argument("--seraphim-test-labels", required=True)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--num-drone", type=int, default=150)
    ap.add_argument("--cache", default="data/coco_cache")
    args = ap.parse_args()

    test_ids = {int(os.path.splitext(n)[0].split("_")[1]) for n in os.listdir("data/v2_test_coco/images")}
    coco = dcm.load_annotations(args.cache, "val2017")
    selected, images = dcm.select_images(coco, bd.COCO_CLASSES, args.per_class, 0.01, seed=11)
    selected = {i: b for i, b in selected.items() if i not in test_ids}
    out = "data/v2_tune_coco"
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(dcm.download, [(images[i]["coco_url"], os.path.join(out, "images", f"coco_{i:012d}.jpg")) for i in selected]))
    for i, boxes in selected.items():
        im = images[i]
        with open(os.path.join(out, "labels", f"coco_{i:012d}.txt"), "w") as f:
            for idx, b in boxes:
                f.write("%d %.6f %.6f %.6f %.6f\n" % ((idx,) + dcm.coco_to_yolo(b, im["width"], im["height"])))
    print(f"{out}: {len(selected)} images (none shared with the test set)")

    names = sorted(n for n in os.listdir(args.seraphim_test_images) if os.path.splitext(n)[1].lower() in add_yolo_dataset.IMG_EXT)
    random.Random(0).shuffle(names)
    fresh = names[1000:1000 + args.num_drone * 2]  # 0-700 old model, 700-1000 test set
    out = "data/v2_tune_drone"
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    kept = 0
    for name in fresh:
        if kept >= args.num_drone:
            break
        stem, e = os.path.splitext(name)
        lines = add_yolo_dataset.read_label(os.path.join(args.seraphim_test_labels, stem + ".txt"), bd.DRONE_ID)
        if not lines:
            continue
        shutil.copy2(os.path.join(args.seraphim_test_images, name), os.path.join(out, "images", f"drone_{kept:05d}{e.lower()}"))
        with open(os.path.join(out, "labels", f"drone_{kept:05d}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        kept += 1
    print(f"{out}: {kept} images")


if __name__ == "__main__":
    main()
