"""Register every KEYSTEP-th frame to frame 0 with ORB + RANSAC homography,
interpolate parameters for the frames in between. Saves H[t] (maps frame-t
pixel coords into frame-0 pixel coords) for every frame as data/register.npz.
"""

import cv2
import numpy as np

VIDEO = "data/video_sdr.mp4"
OUT = "data/register.npz"
KEYSTEP = 5

orb = cv2.ORB_create(4000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def homography_to_ref(ref_kp, ref_des, gray):
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return None
    matches = sorted(matcher.match(ref_des, des), key=lambda m: m.distance)[:800]
    if len(matches) < 30:
        return None
    src = np.float32([kp[m.trainIdx].pt for m in matches])
    dst = np.float32([ref_kp[m.queryIdx].pt for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    if H is None or mask.sum() < 30:
        return None
    return H


def main():
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, ref = cap.read()
    assert ok
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)

    key_idx, key_H = [0], [np.eye(3)]
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % KEYSTEP:
            continue
        H = homography_to_ref(ref_kp, ref_des,
                              cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if H is not None:
            key_idx.append(fi)
            key_H.append(H)
        if fi % 1000 == 0:
            print(f"registered up to frame {fi}/{n}", flush=True)
    cap.release()

    key_idx = np.array(key_idx)
    key_H = np.array(key_H)          # (k, 3, 3)
    full = np.empty((n, 3, 3))
    for a in range(3):
        for b in range(3):
            full[:, a, b] = np.interp(np.arange(n), key_idx, key_H[:, a, b])
    np.savez_compressed(OUT, H=full)
    print(f"saved {n} homographies ({len(key_idx)} keyframes) -> {OUT}")


if __name__ == "__main__":
    main()
