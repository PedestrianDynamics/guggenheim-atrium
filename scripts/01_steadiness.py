"""Check camera steadiness: match ORB features between frame 0 and sampled
frames, estimate translation of the static background over the video."""

import cv2
import numpy as np

VIDEO = "data/video_sdr.mp4"


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_shift(ref, img):
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(img, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)[:500]
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    H, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=2.0)
    return H[0, 2], H[1, 2], int(mask.sum())


def main():
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref = grab(cap, 0)
    print("frame  dx(px)  dy(px)  inliers")
    for idx in np.linspace(0, n - 1, 9).astype(int)[1:]:
        dx, dy, inl = estimate_shift(ref, grab(cap, idx))
        print(f"{idx:6d} {dx:7.2f} {dy:7.2f} {inl:6d}")
    cap.release()


if __name__ == "__main__":
    main()
