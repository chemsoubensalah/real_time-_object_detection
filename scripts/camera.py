"""Camera source helper.

Opens either a local webcam (index such as "0") or a phone camera stream URL,
for example from the Android app "IP Webcam":  http://192.168.1.5:8080/video
or from DroidCam:                               http://192.168.1.5:4747/video

Frames are read in a background thread so we always process the newest frame.
Network streams otherwise build up a buffer and the video lags behind reality.
"""
import sys
import threading
import time

import cv2


def parse_source(source):
    """Return an int for webcam indices ("0", "1"), otherwise the URL/path string."""
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def open_capture(source):
    """Open a capture. On Windows, virtual webcams (iVCam, DroidCam client, OBS)
    open more reliably with DirectShow than with the default MSMF backend."""
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(source)


class VideoStream:
    def __init__(self, source):
        self.source = parse_source(source)
        self.cap = open_capture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {source}\n"
                "- iVCam: open the iVCam app on the PC first, then try --source 1 or 2\n"
                "  (run scripts/list_cameras.py to find the right number).\n"
                "- Phone and computer must be on the same Wi-Fi network.\n"
                "- Check the URL shown in the phone app and open it in a browser first."
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ok, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                # Short network hiccups are common with phone streams; retry.
                time.sleep(0.05)
                continue
            with self.lock:
                self.ok, self.frame = ok, frame

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()
