"""Collect training images from the phone camera (or a webcam).

Usage:
    python scripts/capture_images.py --source http://192.168.1.5:8080/video --out data/raw/images

Keys in the preview window:
    SPACE  save the current frame
    a      toggle automatic capture (one image every --interval seconds)
    q/ESC  quit
"""
import argparse
import os
import time

import cv2

from camera import VideoStream


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or phone stream URL")
    ap.add_argument("--out", default="data/raw/images", help="folder to save images")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto captures")
    ap.add_argument("--prefix", default="img", help="file name prefix, e.g. upper or full")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stream = VideoStream(args.source)
    auto, last_save, count = False, 0.0, len(os.listdir(args.out))
    print("SPACE = save, a = auto capture on/off, q = quit")

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                continue

            now = time.time()
            save = auto and now - last_save >= args.interval
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                save = True
            elif key == ord("a"):
                auto = not auto
                print("auto capture:", "ON" if auto else "OFF")
            elif key in (ord("q"), 27):
                break

            if save:
                name = f"{args.prefix}_{int(now * 1000)}.jpg"
                cv2.imwrite(os.path.join(args.out, name), frame)
                count, last_save = count + 1, now
                print(f"saved {name} (total {count})")

            preview = frame.copy()
            status = f"saved: {count}  auto: {'ON' if auto else 'OFF'}"
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("capture (SPACE save, a auto, q quit)", preview)
    finally:
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
