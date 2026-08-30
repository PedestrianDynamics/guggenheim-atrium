"""Extract a distribution-level demand model from the measured trajectories.

Outputs data/demand.json:
- walkable area polygon (concave hull of measured points, m)
- entry/exit zones (clusters of track endpoints near the hull boundary)
- OD counts between zones
- attractor points (clusters of stop episodes >2 s) with visit stats
- dwell-time lognormal fit, per-agent desired-speed distribution
- arrival times (track start times, s)
"""

import json

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from shapely import concave_hull
from shapely.geometry import MultiPoint, Point

FPS = 29.97
STAND_V = 0.3
DUR = 389.4

df = pd.read_parquet("data/tracks_world.parquet")

# kinematics
parts = []
for tid, g in df.groupby("id"):
    g = g.sort_values("frame").copy()
    g["v"] = np.hypot(np.gradient(g.x_m, g.t_s), np.gradient(g.y_m, g.t_s))
    parts.append(g)
df = pd.concat(parts, ignore_index=True)

# walkable area: concave hull of a point sample, slightly buffered
pts = df[["x_m", "y_m"]].to_numpy()[::37]
hull = concave_hull(MultiPoint([tuple(p) for p in pts]), ratio=0.25)
hull = hull.buffer(0.3).simplify(0.4)
boundary = hull.exterior

# track endpoints; an endpoint is a truthful entry/exit if near the boundary
# and not at the clip edges of the video
info = []
for tid, g in df.groupby("id"):
    g = g.sort_values("frame")
    info.append(dict(
        id=tid, t0=g.t_s.iloc[0], t1=g.t_s.iloc[-1],
        x0=g.x_m.iloc[0], y0=g.y_m.iloc[0],
        x1=g.x_m.iloc[-1], y1=g.y_m.iloc[-1],
        v_walk=g.v[g.v > 0.5].median() if (g.v > 0.5).any() else np.nan,
    ))
info = pd.DataFrame(info)

near0 = info.apply(lambda r: boundary.distance(Point(r.x0, r.y0)) < 2.5, axis=1)
near1 = info.apply(lambda r: boundary.distance(Point(r.x1, r.y1)) < 2.5, axis=1)

# cluster boundary endpoints into gates
ends = np.r_[info.loc[near0, ["x0", "y0"]].to_numpy(),
             info.loc[near1, ["x1", "y1"]].to_numpy()]
K_GATES = 5
cent, _ = kmeans2(ends, K_GATES, seed=7, minit="++")


def gate_of(x, y):
    return int(np.argmin(np.hypot(cent[:, 0] - x, cent[:, 1] - y)))


info["g0"] = [gate_of(r.x0, r.y0) if n else -1
              for (_, r), n in zip(info.iterrows(), near0)]
info["g1"] = [gate_of(r.x1, r.y1) if n else -1
              for (_, r), n in zip(info.iterrows(), near1)]

od = np.zeros((K_GATES, K_GATES), int)
both = info[(info.g0 >= 0) & (info.g1 >= 0)]
for _, r in both.iterrows():
    od[int(r.g0), int(r.g1)] += 1
print(f"tracks with boundary OD: {len(both)}/{len(info)}")
print("OD matrix:\n", od)

# stop episodes >2 s -> attractor clusters
stop_pts, stop_dur, stop_tid = [], [], []
for tid, g in df.groupby("id"):
    g = g.sort_values("frame")
    standing = (g.v < STAND_V).to_numpy()
    d = np.diff(np.r_[0, standing.astype(int), 0])
    starts, ends_ = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends_):
        if (e - s) / FPS > 2:
            stop_pts.append([g.x_m.iloc[s:e].mean(), g.y_m.iloc[s:e].mean()])
            stop_dur.append((e - s) / FPS)
            stop_tid.append(tid)
stop_pts = np.array(stop_pts)
K_ATTR = 8
acent, alab = kmeans2(stop_pts, K_ATTR, seed=7, minit="++")
attr_w = np.bincount(alab, minlength=K_ATTR).astype(float)
attr_w /= attr_w.sum()

# stops per track distribution, boundary-OD tracks only (the population
# the simulation reproduces; fragments would dilute the pmf)
od_ids = set(both.id)
n_stops = pd.Series([t for t in stop_tid if t in od_ids]).value_counts()
stops_per_track = np.r_[n_stops.to_numpy(),
                        np.zeros(len(both) - len(n_stops), int)]
p_nstops = np.bincount(stops_per_track, minlength=6)[:6].astype(float)
p_nstops /= p_nstops.sum()
print("stops-per-track pmf:", p_nstops.round(2))

ln_mu, ln_sig = np.log(stop_dur).mean(), np.log(stop_dur).std()
print(f"dwell lognormal mu={ln_mu:.2f} sig={ln_sig:.2f} "
      f"(median {np.exp(ln_mu):.1f}s)")

vw = info.v_walk.dropna()
print(f"desired speed: mean {vw.mean():.2f} std {vw.std():.2f}")

demand = dict(
    walkable=list(np.array(hull.exterior.coords).round(2).tolist()),
    gates=cent.round(2).tolist(),
    od=od.tolist(),
    attractors=acent.round(2).tolist(),
    attractor_p=attr_w.round(4).tolist(),
    p_nstops=p_nstops.round(4).tolist(),
    dwell_lognorm=[round(ln_mu, 3), round(ln_sig, 3)],
    dwell_samples=[round(s, 1) for s in stop_dur],
    speed_mean_std=[round(vw.mean(), 3), round(vw.std(), 3)],
    entries=sorted([[round(r.t0, 2), int(r.g0)]
                    for _, r in info[info.g0 >= 0].iterrows()]),
    duration=DUR,
)
with open("data/demand.json", "w") as f:
    json.dump(demand, f, indent=1)
print(f"entries: {len(demand['entries'])}")
print("-> data/demand.json")
