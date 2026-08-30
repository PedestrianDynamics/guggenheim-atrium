"""Attraction power vs holding power per attractor (Melton's indicators,
cf. Centorrino et al. 2021), plus Weibull-vs-lognormal AIC for dwell times.
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

FPS = 29.97
STAND_V = 0.3

demand = json.load(open("data/demand.json"))
attr = np.array(demand["attractors"])

df = pd.read_parquet("data/tracks_world.parquet")

n_tracks = df.id.nunique()
visits = {k: [] for k in range(len(attr))}      # dwell lists per attractor
visitors = {k: set() for k in range(len(attr))}
all_dwells = []
for tid, g in df.groupby("id"):
    g = g.sort_values("frame")
    v = np.hypot(np.gradient(g.x_m, g.t_s), np.gradient(g.y_m, g.t_s))
    d = np.diff(np.r_[0, (v < STAND_V).astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        dur = (e - s) / FPS
        if dur > 2:
            c = [g.x_m.iloc[s:e].mean(), g.y_m.iloc[s:e].mean()]
            a = int(np.argmin(np.hypot(*(attr - c).T)))
            visits[a].append(dur)
            visitors[a].add(tid)
            all_dwells.append(dur)

print(f"{n_tracks} tracks, {len(all_dwells)} stop episodes >2s\n")
print("attractor  (x, y) [m]      attraction  holding [s]  (median)")
rows = []
for k in range(len(attr)):
    ap = len(visitors[k]) / n_tracks
    hp = np.mean(visits[k]) if visits[k] else 0
    med = np.median(visits[k]) if visits[k] else 0
    rows.append((k, ap, hp))
    print(f"  A{k}   ({attr[k][0]:6.1f}, {attr[k][1]:6.1f})   "
          f"{ap * 100:5.1f}%     {hp:6.1f}      ({med:.1f})")

r = np.corrcoef([r[1] for r in rows], [r[2] for r in rows])[0, 1]
print(f"\ncorrelation attraction vs holding power: r={r:.2f}")

# dwell distribution: Weibull vs lognormal (both 2-parameter, loc=0)
x = np.array(all_dwells)
res = {}
for name, dist, kw in [("lognormal", stats.lognorm, dict(floc=0)),
                       ("weibull", stats.weibull_min, dict(floc=0)),
                       ("exponential", stats.expon, dict(floc=0))]:
    params = dist.fit(x, **kw)
    ll = np.sum(dist.logpdf(x, *params))
    k_free = len(params) - 1                    # loc fixed
    res[name] = (2 * k_free - 2 * ll, params)
    print(f"{name:12s} AIC {res[name][0]:8.1f}  params {np.round(params, 3)}")
best = min(res, key=lambda k: res[k][0])
print(f"best by AIC: {best}")
if best == "weibull":
    k_shape = res["weibull"][1][0]
    print(f"Weibull shape k={k_shape:.2f} "
          f"({'increasing' if k_shape > 1 else 'decreasing'} leave rate)")
