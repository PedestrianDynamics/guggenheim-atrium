"""Median background plate: warp sampled frames into frame-0 coordinates
using the registration homographies, take the per-pixel median."""

import cv2
import numpy as np

VIDEO = "data/video_sdr.mp4"
REG = "data/register.npz"
OUT = "data/background.png"
N_SAMPLES = 60


def main():
    H = np.load(REG)["H"]
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, min(n, len(H)) - 1, N_SAMPLES).astype(int)
    stack = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        warped = cv2.warpPerspective(frame, H[idx], (1920, 1080))
        stack.append(warped)
    cap.release()
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    cv2.imwrite(OUT, bg)
    print(f"background from {len(stack)} frames -> {OUT}")


if __name__ == "__main__":
    main()
