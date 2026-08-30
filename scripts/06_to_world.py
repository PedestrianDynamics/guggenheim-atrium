"""Raw pixel tracks -> cleaned world trajectories.

Pipeline: undo camera shake (per-frame registration H), map to tile units
(metric rectification H), stitch broken track ids, drop short tracks,
Savitzky-Golay smoothing, then calibrate meters-per-tile from the modal
free-walking speed and write tracks_world.parquet.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 29.97
MIN_TRACK_S = 5.0
STITCH_GAP_S = 2.0
STITCH_DIST_TILES = 3.0
TARGET_MODE_SPEED = 1.30        # m/s, free walking


def apply_homographies(df):
    Hreg = np.load("data/register.npz")["H"]
    Hm = np.load("data/rectify.npz")["H"]
    pts = df[["x_px", "y_px"]].to_numpy()
    frames = df["frame"].to_numpy()
    H_all = Hm @ Hreg[frames]                      # (n,3,3)
    p = np.einsum("nij,nj->ni", H_all, np.c_[pts, np.ones(len(pts))])
    df["x_t"] = p[:, 0] / p[:, 2]
    df["y_t"] = p[:, 1] / p[:, 2]
    return df


def clip_to_floor(df):
    """Drop points whose ground position falls outside the floor polygon."""
    import matplotlib.path as mpath
    rect = np.load("data/rectify.npz")
    Hm = rect["H"]
    floor_px = rect["floor"].astype(float)
    p = np.c_[floor_px, np.ones(len(floor_px))] @ Hm.T
    poly = mpath.Path(p[:, :2] / p[:, 2:])
    keep = poly.contains_points(df[["x_t", "y_t"]].to_numpy())
    return df[keep].copy()


def despike(df):
    """Hampel filter per track: replace >4*MAD outliers by rolling median."""
    out = []
    for tid, g in df.groupby("id"):
        g = g.sort_values("frame").copy()
        for c in ["x_t", "y_t"]:
            med = g[c].rolling(15, center=True, min_periods=1).median()
            mad = (g[c] - med).abs().rolling(15, center=True,
                                             min_periods=1).median()
            bad = (g[c] - med).abs() > 4 * mad + 1e-9
            g.loc[bad, c] = med[bad]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def stitch(df):
    """Greedy stitch: join a track that ends to one that starts soon after
    nearby. Iterates once over end/start pairs sorted by time gap."""
    info = df.groupby("id").agg(start=("frame", "min"), end=("frame", "max"))
    first = df.loc[df.groupby("id")["frame"].idxmin()].set_index("id")
    last = df.loc[df.groupby("id")["frame"].idxmax()].set_index("id")
    parent = {i: i for i in info.index}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    cands = []
    ids = info.index.to_numpy()
    for a in ids:
        for b in ids:
            if a == b:
                continue
            gap = info.loc[b, "start"] - info.loc[a, "end"]
            if not (0 < gap <= STITCH_GAP_S * FPS):
                continue
            d = np.hypot(first.loc[b, "x_t"] - last.loc[a, "x_t"],
                         first.loc[b, "y_t"] - last.loc[a, "y_t"])
            if d < STITCH_DIST_TILES:
                cands.append((gap, a, b))
    used_a, used_b = set(), set()
    for gap, a, b in sorted(cands):
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        parent[root(b)] = root(a)
    df["id"] = df["id"].map(lambda i: root(i))
    return df


def smooth(df):
    out = []
    for tid, g in df.groupby("id"):
        g = g.sort_values("frame").drop_duplicates("frame")
        if len(g) < MIN_TRACK_S * FPS:
            continue
        # resample to every frame via interpolation over gaps
        fr = np.arange(g.frame.min(), g.frame.max() + 1)
        x = np.interp(fr, g.frame, g.x_t)
        y = np.interp(fr, g.frame, g.y_t)
        win = min(31, len(fr) // 2 * 2 - 1)
        x = savgol_filter(x, win, 2)
        y = savgol_filter(y, win, 2)
        out.append(pd.DataFrame({"id": tid, "frame": fr, "x_t": x, "y_t": y}))
    return pd.concat(out, ignore_index=True)


def calibrate_scale(df):
    """Meters per tile from the modal walking speed (target 1.30 m/s)."""
    vx, vy = [], []
    for _, g in df.groupby("id"):
        vx.append(np.diff(g.x_t) * FPS)
        vy.append(np.diff(g.y_t) * FPS)
    vx = np.concatenate(vx)
    vy = np.concatenate(vy)
    v = np.hypot(vx, vy)
    walk = (v > 0.5) & (v < 4)           # walking band, tile units/s
    hist, edges = np.histogram(v[walk], bins=70, range=(0.5, 4))
    mode = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    tile_m = TARGET_MODE_SPEED / mode
    print(f"walking-mode speed {mode:.3f} tiles/s -> tile = {tile_m:.3f} m")
    # isotropy diagnostic: median walking speed of x- vs y-dominant motion
    xdom = walk & (np.abs(vx) > 2 * np.abs(vy))
    ydom = walk & (np.abs(vy) > 2 * np.abs(vx))
    print(f"isotropy: median |v| x-dominant {np.median(v[xdom]):.3f}, "
          f"y-dominant {np.median(v[ydom]):.3f} tiles/s")
    return tile_m


def main():
    df = pd.read_parquet("data/tracks_raw.parquet")
    print(f"raw: {len(df)} rows, {df.id.nunique()} ids")
    df = apply_homographies(df)
    df = clip_to_floor(df)
    print(f"on floor: {len(df)} rows, {df.id.nunique()} ids")
    df = despike(df)
    df = stitch(df)
    print(f"stitched: {df.id.nunique()} ids")
    df = smooth(df)
    print(f"kept after min-duration: {df.id.nunique()} ids")
    # anisotropy correction: walking speed must not depend on heading.
    # Within-person estimator (same track walking both x- and y-dominant)
    # avoids population bias; iterate to convergence.
    for it in range(5):
        ratios = []
        for _, g in df.groupby("id"):
            vx = np.diff(g.x_t) * FPS
            vy = np.diff(g.y_t) * FPS
            v = np.hypot(vx, vy)
            walk = (v > 0.5) & (v < 4)
            xdom = walk & (np.abs(vx) > 2 * np.abs(vy))
            ydom = walk & (np.abs(vy) > 2 * np.abs(vx))
            if xdom.sum() > FPS and ydom.sum() > FPS:
                ratios.append(np.median(v[ydom]) / np.median(v[xdom]))
        r = float(np.median(ratios))
        print(f"anisotropy iter {it}: x *= {r:.3f} (n={len(ratios)})")
        df["x_t"] *= r
        if abs(r - 1) < 0.02:
            break
    tile_m = calibrate_scale(df)
    df["x_m"] = df.x_t * tile_m
    df["y_m"] = df.y_t * tile_m
    df["t_s"] = df.frame / FPS
    df.to_parquet("data/tracks_world.parquet")
    with open("data/scale.txt", "w") as f:
        f.write(f"tile_m={tile_m:.4f}\n")
    print(f"-> data/tracks_world.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()
