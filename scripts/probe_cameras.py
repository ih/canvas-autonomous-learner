"""Quick probe: open camera indices 0..1 with both DSHOW and MSMF, longer warmup,
save PNG + report mean brightness over multiple frames."""
from __future__ import annotations
import sys, time
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).parent.parent / "local" / "camera_probe"
OUT.mkdir(parents=True, exist_ok=True)

BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]

for backend_name, backend in BACKENDS:
    for idx in range(2):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            print(f"backend={backend_name} index={idx}: not opened")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Long warmup: many cameras drop the first 1-2s of frames.
        means = []
        last_frame = None
        for i in range(40):
            ok, frame = cap.read()
            if ok and frame is not None:
                means.append(float(frame.mean()))
                last_frame = frame
            time.sleep(0.05)
        if last_frame is None:
            print(f"backend={backend_name} index={idx}: opened but never returned a frame")
            cap.release()
            continue
        h, w = last_frame.shape[:2]
        peak_mean = max(means) if means else 0.0
        last_mean = means[-1] if means else 0.0
        nonblack = float((last_frame.sum(axis=2) > 5).mean())
        out = OUT / f"{backend_name}_cam_{idx}.png"
        cv2.imwrite(str(out), last_frame)
        print(
            f"backend={backend_name} index={idx}: {w}x{h} "
            f"frames={len(means)} peak_mean={peak_mean:.1f} last_mean={last_mean:.1f} "
            f"nonblack={nonblack:.3f} -> {out}"
        )
        cap.release()
        time.sleep(0.3)
