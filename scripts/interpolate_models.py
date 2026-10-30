"""Weight-space interpolation of two checkpoints with the same architecture (idea from WiSE-FT).

    weights = (1 - alpha) * model_a + alpha * model_b

model_a: the model before fine-tuning (knows the pretrained classes perfectly, the new class poorly)
model_b: the fine-tuned model (knows the new class, forgot a little of the old)
Fine-tuned models usually stay in the same "valley" as their starting point, so points between the two
often keep most of the new skill while recovering part of the old one, without any extra training.

Usage:
    python scripts/interpolate_models.py --a runs_v3/init/yolo11n_12cls_init.pt --b models/v3_yolo11n_12cls.pt \
        --alpha 0.5 --out runs_v3/mix/alpha_0.50.pt
"""
import argparse
import datetime
import os

import torch
from ultralytics import YOLO


def interpolate(model_a, model_b, alpha):
    """Return model_b's structure with weights (1-alpha)*a + alpha*b (floating-point tensors only)."""
    sa, sb = model_a.state_dict(), model_b.state_dict()
    mixed = {}
    for k, vb in sb.items():
        va = sa[k]
        if vb.is_floating_point():
            mixed[k] = (1 - alpha) * va.float() + alpha * vb.float()
        else:  # counters such as BatchNorm's num_batches_tracked
            mixed[k] = vb
    model_b.load_state_dict(mixed)
    return model_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = YOLO(args.a).model.float(), YOLO(args.b).model.float()
    assert list(a.state_dict()) == list(b.state_dict()), "the two models must have the same architecture"
    model = interpolate(a, b, args.alpha)
    ckpt = {"epoch": -1, "best_fitness": None, "model": model.half(), "ema": None, "updates": None, "optimizer": None,
            "train_args": {}, "date": datetime.datetime.now().isoformat()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"saved {args.out} (alpha={args.alpha})")


if __name__ == "__main__":
    main()
