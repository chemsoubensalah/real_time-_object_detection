"""Occlusion experiment: does training with occluded images help?

Steps (all use the same base model, the same epochs and the same seed, so the
only difference between B1 and B2 is the training data):
  A.  evaluate the base model on clean test + test occluded 20%, 40%, 60%
  B1. control:   fine-tune base model 10 epochs on the clean training set
  B2. occlusion: fine-tune base model 10 epochs on clean + occluded copies
  C.  evaluate B1 and B2 on the same four test sets

Results are written to results/occlusion/occlusion.json.

Usage (run from the project folder, after the base training finished):
    python scripts/run_occlusion_experiment.py --base ../yolov5/runs/train/multi/weights/best.pt
"""
import argparse
import json
import os
import re
import subprocess
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_SETS = {
    "clean": f"{PROJECT}/data/multi_dataset/data.yaml",
    "occ20": f"{PROJECT}/data/occluded/test_20.yaml",
    "occ40": f"{PROJECT}/data/occluded/test_40.yaml",
    "occ60": f"{PROJECT}/data/occluded/test_60.yaml",
}
ROW = re.compile(r"^\s+(\w+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def run(cmd, cwd):
    print("$", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:], out.stderr[-3000:])
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return out.stdout + out.stderr


def evaluate(weights, yolov5_dir, name):
    """val.py on every test set -> {set: {class: {P, R, mAP50, mAP50_95}}}"""
    res = {}
    for set_name, yaml in TEST_SETS.items():
        log = run([sys.executable, "val.py", "--weights", weights, "--data", yaml, "--task", "test",
                   "--img", "416", "--device", "cpu", "--verbose", "--name", f"{name}_{set_name}", "--exist-ok"],
                  yolov5_dir)
        rows = {}
        for line in log.replace("\r", "\n").splitlines():
            m = ROW.match(line)
            if m:
                cls, _, _, p, r, m50, m5095 = m.groups()
                rows[cls] = {"P": float(p), "R": float(r), "mAP50": float(m50), "mAP50_95": float(m5095)}
        res[set_name] = rows
        print(name, set_name, rows.get("all"), flush=True)
    return res


def finetune(base, data_yaml, name, epochs, yolov5_dir):
    run([sys.executable, "train.py", "--img", "416", "--batch", "32", "--epochs", str(epochs), "--data", data_yaml,
         "--weights", base, "--device", "cpu", "--workers", "4", "--name", name, "--exist-ok", "--seed", "0"],
        yolov5_dir)
    return os.path.join(yolov5_dir, "runs", "train", name, "weights", "best.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="best.pt of the multi-class training")
    ap.add_argument("--yolov5-dir", default=os.path.join(PROJECT, "..", "yolov5"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(PROJECT, "results", "occlusion"))
    args = ap.parse_args()
    y5 = os.path.abspath(args.yolov5_dir)
    base = os.path.abspath(args.base)
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "occlusion.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}

    def save():
        with open(out_json, "w") as f:
            json.dump(results, f, indent=1)

    if "base" not in results:
        results["base"] = evaluate(base, y5, "occ_base")
        save()
    if "finetune_clean" not in results:
        w = finetune(base, TEST_SETS["clean"], "ft_clean", args.epochs, y5)
        results["finetune_clean"] = evaluate(w, y5, "occ_ftclean")
        save()
    if "finetune_occlusion" not in results:
        w = finetune(base, f"{PROJECT}/data/multi_occ.yaml", "ft_occ", args.epochs, y5)
        results["finetune_occlusion"] = evaluate(w, y5, "occ_ftocc")
        save()
    print("saved", out_json)


if __name__ == "__main__":
    main()
