"""Train the 12-class YOLO11n model (the "v2" model).

- Starts from yolo11n.pt (pretrained on COCO 80 classes) and fine-tunes on data/v2_dataset.
- Safe to restart: if a checkpoint exists the run continues from it, so a machine restart costs at most one epoch.

Usage:
    python scripts/train_v2.py                      # 25 epochs at 512 px
    python scripts/train_v2.py --epochs 40 --imgsz 640
"""
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2_dataset/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained starting point")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--close-mosaic", type=int, default=5, help="turn mosaic off for the last N epochs")
    ap.add_argument("--project", default="/home/user/runs_v2")
    ap.add_argument("--name", default="yolo11n_12cls")
    args = ap.parse_args()

    last = os.path.join(args.project, args.name, "weights", "last.pt")
    if os.path.exists(last):
        print("resuming from", last, flush=True)
        YOLO(last).train(resume=True)
        return
    model = YOLO(args.model)
    model.train(data=os.path.abspath(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device="cpu", project=args.project, name=args.name, exist_ok=True,
                close_mosaic=args.close_mosaic, cache="ram", patience=100, seed=0, plots=True)


if __name__ == "__main__":
    main()
