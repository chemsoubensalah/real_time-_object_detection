"""Real-time detection on the phone camera with a trained YOLOv5 .pt model.

Usage:
    python scripts/detect_realtime.py --weights best.pt --source http://192.168.1.5:8080/video
    python scripts/detect_realtime.py --weights best.pt --source 0            # laptop webcam
    python scripts/detect_realtime.py --weights best.pt --source test.mp4     # video file
    python scripts/detect_realtime.py --weights best.pt --source photo.jpg    # single image

Shows boxes, confidence and FPS. Press q to quit, s to save a screenshot.
"""
import argparse
import os
import time

import cv2
import torch

from camera import VideoStream

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_local_yolov5(path):
    """Return a cloned yolov5 folder if one exists (given path, ./yolov5 or ../yolov5)."""
    for p in [path, "yolov5", os.path.join("..", "yolov5")]:
        if p and os.path.isfile(os.path.join(p, "hubconf.py")):
            return p
    return None


def load_model(weights, conf, device, yolov5_dir=""):
    local = find_local_yolov5(yolov5_dir)
    if local:
        model = torch.hub.load(local, "custom", path=weights, device=device, source="local")
    else:
        # Downloads the YOLOv5 code from GitHub the first time, then uses the local cache.
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights, device=device)
    model.conf = conf
    return model


COLORS = [(0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 200, 255), (0, 0, 255), (255, 255, 0)]  # BGR, one per class


def draw(frame, det, names):
    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        color = COLORS[int(cls) % len(COLORS)]
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def detect(model, frame, img_size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return model(rgb, size=img_size).xyxy[0].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source", default="0", help="webcam index, phone stream URL, video or image")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--device", default="", help="'' auto, 'cpu', or '0' for GPU 0")
    ap.add_argument("--save-dir", default="runs/realtime")
    ap.add_argument("--yolov5-dir", default="", help="path to a cloned yolov5 repo (default: ./yolov5 or ../yolov5)")
    ap.add_argument("--save-video", default="", help="also record the annotated video to this .mp4 path")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (servers / recording only)")
    args = ap.parse_args()

    model = load_model(args.weights, args.conf, args.device or None, args.yolov5_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    if os.path.splitext(args.source)[1].lower() in IMG_EXT:
        frame = cv2.imread(args.source)
        det = detect(model, frame, args.img_size)
        out = os.path.join(args.save_dir, os.path.basename(args.source))
        cv2.imwrite(out, draw(frame, det, model.names))
        print(f"{len(det)} detections, saved {out}")
        return

    is_file = os.path.isfile(args.source)
    cap = cv2.VideoCapture(args.source) if is_file else None
    stream = None if is_file else VideoStream(args.source)

    fps, prev = 0.0, time.time()
    fps_log = []
    writer = None
    try:
        while True:
            ok, frame = cap.read() if is_file else stream.read()
            if not ok or frame is None:
                if is_file:
                    break
                continue

            det = detect(model, frame, args.img_size)
            draw(frame, det, model.names)

            now = time.time()
            inst = 1.0 / max(now - prev, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst  # smoothed so the number is readable
            prev = now
            fps_log.append(inst)
            info = f"FPS: {fps:.1f}   objects: {len(det)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if args.save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    rate = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), rate or 20, (w, h))
                writer.write(frame)
            if args.no_show:
                continue
            cv2.imshow("YOLOv5 real-time (q quit, s screenshot)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = os.path.join(args.save_dir, f"shot_{int(now)}.jpg")
                cv2.imwrite(path, frame)
                print("saved", path)
    finally:
        if cap:
            cap.release()
        if stream:
            stream.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    if len(fps_log) > 10:
        avg = sum(fps_log[10:]) / len(fps_log[10:])  # skip warm-up frames
        print(f"Average FPS: {avg:.1f}")


if __name__ == "__main__":
    main()
