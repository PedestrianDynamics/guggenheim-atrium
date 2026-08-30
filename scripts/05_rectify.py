"""Metric rectification of the floor plane from the tile grid.

1. LSD line segments on the background plate (floor region only).
2. Split into two orientation families, RANSAC a vanishing point each.
3. Projective + affine rectification such that the two joint families
   become orthogonal with equal spacing (square tiles).
4. Save H_metric (frame-0 px -> tile units) and a warped preview.
"""

import cv2
import numpy as np

BG = "data/background.png"
OUT = "data/rectify.npz"
PREVIEW = "data/rectified_preview.png"

# generous floor polygon in frame-0 coords (excludes walls/elevators)
FLOOR = np.array([[420, 1075], [230, 700], [520, 420], [700, 260],
                  [1050, 100], [1300, 170], [1420, 420], [1660, 640],
                  [1500, 980], [1200, 1079]], np.int32)


def segments(gray, mask):
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.empty((0, 4))
    lines = lines.reshape(-1, 4)
    keep = []
    for x1, y1, x2, y2 in lines:
        if np.hypot(x2 - x1, y2 - y1) < 18:
            continue
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and mask[my, mx]:
            keep.append((x1, y1, x2, y2))
    return np.array(keep)


def to_homog_lines(segs):
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def ransac_vp(lines, iters=3000, tol=0.015, rng=np.random.default_rng(0)):
    best, best_in = None, 0
    n = len(lines)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) < 1e-9 and np.linalg.norm(vp[:2]) < 1e-6:
            continue
        vpn = vp / np.linalg.norm(vp)
        d = np.abs(lines @ vpn)
        inl = (d < tol).sum()
        if inl > best_in:
            best_in, best = inl, vpn
    d = np.abs(lines @ best)
    inliers = lines[d < tol]
    # least-squares refit: vp minimizing sum (l . v)^2  -> smallest eigvec
    _, _, V = np.linalg.svd(inliers)
    vp = V[-1]
    return vp / (vp[2] if abs(vp[2]) > 1e-9 else 1.0), best_in


def main():
    bg = cv2.imread(BG)
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [FLOOR], 255)
    segs = segments(gray, mask)
    print(f"{len(segs)} segments on floor")

    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1],
                                segs[:, 2] - segs[:, 0])) % 180
    # two orientation families from histogram peaks
    hist, edges = np.histogram(ang, bins=36, range=(0, 180))
    p1 = edges[np.argmax(hist)] + 2.5
    # second peak at least 40 deg away
    away = np.minimum(np.abs(edges[:-1] + 2.5 - p1),
                      180 - np.abs(edges[:-1] + 2.5 - p1)) > 40
    p2 = edges[:-1][away][np.argmax(hist[away])] + 2.5
    print(f"orientation peaks: {p1:.0f} deg / {p2:.0f} deg")

    def family(peak):
        diff = np.minimum(np.abs(ang - peak), 180 - np.abs(ang - peak))
        return segs[diff < 15]

    lines1 = to_homog_lines(family(p1))
    lines2 = to_homog_lines(family(p2))
    vp1, in1 = ransac_vp(lines1)
    vp2, in2 = ransac_vp(lines2)
    print(f"vp1 {vp1[:2].round(1)} ({in1}/{len(lines1)} inl), "
          f"vp2 {vp2[:2].round(1)} ({in2}/{len(lines2)} inl)")

    # projective rectification: send vanishing line to infinity
    vline = np.cross(vp1, vp2)
    vline /= np.linalg.norm(vline[:2])
    Hp = np.array([[1, 0, 0], [0, 1, 0], list(vline)])

    # after Hp the two families are parallel; find their directions
    def direction(vp):
        d = (Hp @ vp)
        d = d[:2] / np.linalg.norm(d[:2])
        return d

    d1, d2 = direction(vp1), direction(vp2)
    # affine correction A st A d1 -> (1,0), A d2 -> (0,1) (orthogonalize)
    A = np.linalg.inv(np.c_[d1, d2])
    Ha = np.eye(3)
    Ha[:2, :2] = A
    H = Ha @ Hp

    def warp_pts(H, pts):
        p = np.c_[pts, np.ones(len(pts))] @ H.T
        return p[:, :2] / p[:, 2:]

    # provisional scale: map the floor polygon into a ~1400 px canvas,
    # then measure the tile period by autocorrelation along each axis
    fp = warp_pts(H, FLOOR.astype(float))
    span = fp.max(0) - fp.min(0)
    pre = 1400 / span.max()
    Hc = np.diag([pre, pre, 1]) @ H
    fp = warp_pts(Hc, FLOOR.astype(float))
    T0 = np.array([[1, 0, -fp.min(0)[0]], [0, 1, -fp.min(0)[1]], [0, 0, 1.0]])
    Hc = T0 @ Hc
    size = np.ceil(warp_pts(Hc, FLOOR.astype(float)).max(0)).astype(int) + 1
    wbg = cv2.warpPerspective(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY),
                              Hc, tuple(size))
    wmask = cv2.warpPerspective(mask, Hc, tuple(size))
    cv2.imwrite("data/ortho_debug.png", wbg)

    def period(img, axis):
        # 2-D autocorrelation of joint gradients in a central floor patch;
        # first peak along the axis is one tile module
        h, w = img.shape
        patch = img[h // 3:2 * h // 3, w // 4:3 * w // 4].astype(float)
        g = np.abs(cv2.Sobel(patch, cv2.CV_64F, 1 - axis, axis))
        g -= g.mean()
        f = np.fft.rfft2(g)
        ac = np.fft.irfft2(f * np.conj(f))
        ac /= ac[0, 0]
        prof = ac[0, :150] if axis == 0 else ac[:150, 0]
        from scipy.signal import find_peaks
        peaks, props = find_peaks(prof[10:], prominence=0.02)
        if len(peaks) == 0:
            raise RuntimeError("no period found")
        return 10 + peaks[np.argmax(props["prominences"])]

    per_x = period(wbg, 0)
    per_y = period(wbg, 1)
    print(f"tile period: {per_x} px x, {per_y} px y (canvas), "
          f"aspect {per_x / per_y:.3f}")
    S = np.diag([1 / per_x, 1 / per_y, 1])
    H = S @ Hc

    # normalize: keep floor center at origin-ish and positive orientation
    c = warp_pts(H, np.array([[960, 700]]))[0]
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], float)
    H = T @ H
    np.savez(OUT, H=H, floor=FLOOR)

    # preview: warp background, 30 px per tile unit
    px = 30
    Hprev = np.diag([px, px, 1]) @ H
    Hprev[0, 2] += 900
    Hprev[1, 2] += 600
    warped = cv2.warpPerspective(bg, Hprev, (1800, 1200))
    for k in range(-30, 31):
        cv2.line(warped, (0, 600 + k * px), (1800, 600 + k * px), (0, 0, 255), 1)
        cv2.line(warped, (900 + k * px, 0), (900 + k * px, 1200), (0, 0, 255), 1)
    cv2.imwrite(PREVIEW, warped)
    print(f"saved {OUT} and {PREVIEW}")


if __name__ == "__main__":
    main()
