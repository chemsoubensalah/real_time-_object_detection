"""List camera numbers that work on this computer and save a preview of each.

Use it to find which number is the phone (iVCam / DroidCam client / OBS virtual camera):
    python scripts/list_cameras.py

Then open the saved camera_N.jpg images and use that N:
    python scripts/detect_realtime.py --weights models/person_yolov5n.pt --source N
"""
import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")  # hide warnings for missing camera numbers

import cv2  # noqa: E402

from camera import open_capture  # noqa: E402


def main(max_index=6):
    found = []
    for i in range(max_index):
        cap = open_capture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            name = f"camera_{i}.jpg"
            cv2.imwrite(name, frame)
            h, w = frame.shape[:2]
            print(f"camera {i}: OK {w}x{h}  -> preview saved to {name}")
            found.append(i)
        else:
            print(f"camera {i}: not available")
    if not found:
        print("No camera found. For iVCam make sure the PC app is open and shows the phone image.")


if __name__ == "__main__":
    main()
