"""Strategic-level randomness test following Yoshimura et al. (2019).

Each boundary-OD track becomes a symbolic path [gate_in, attractors..., gate_out]
(attractor = stop episode >2 s assigned to its demand-model cluster).
Null model: random walk on the zone graph — entry gate and path length are
taken from the data; exit gate uniform over gates, attractor choices uniform
without immediate repetition. R = P_observed / P_null per path type.
Pattern strength per group (crossers = no stops, stoppers = >=1 stop):
mean R of observed paths and KL divergence from the null.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

FPS = 29.97
STAND_V = 0.3
N_G = 5

demand = json.load(open("data/demand.json"))
gates = np.array(demand["gates"])
attr = np.array(demand["attractors"])
hull = Polygon(demand["walkable"]).exterior

df = pd.read_parquet("data/tracks_world.parquet")

paths = []
for tid, g in df.groupby("id"):
    g = g.sort_values("frame")
    p0, p1 = Point(g.x_m.iloc[0], g.y_m.iloc[0]), Point(g.x_m.iloc[-1],
                                                        g.y_m.iloc[-1])
    if hull.distance(p0) > 2.5 or hull.distance(p1) > 2.5:
        continue
    g0 = int(np.argmin(np.hypot(*(gates - [g.x_m.iloc[0], g.y_m.iloc[0]]).T)))
    g1 = int(np.argmin(np.hypot(*(gates - [g.x_m.iloc[-1], g.y_m.iloc[-1]]).T)))
    v = np.hypot(np.gradient(g.x_m, g.t_s), np.gradient(g.y_m, g.t_s))
    standing = (v < STAND_V)
    d = np.diff(np.r_[0, standing.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    stops = []
    for s, e in zip(starts, ends):
        if (e - s) / FPS > 2:
            c = [g.x_m.iloc[s:e].mean(), g.y_m.iloc[s:e].mean()]
            a = int(np.argmin(np.hypot(*(attr - c).T)))
            if not stops or stops[-1] != a:      # merge repeated same-spot
                stops.append(a)
    paths.append((f"G{g0}", *[f"A{a}" for a in stops], f"G{g1}"))

print(f"symbolic paths: {len(paths)}")


def null_prob(path):
    """entry gate + length conditioned on data; rest uniform."""
    n_stops = len(path) - 2
    p = 1.0 / N_G                                # exit gate
    prev = None
    for _ in range(n_stops):
        opts = len(attr) - (1 if prev else 0)    # no immediate repetition
        p *= 1.0 / opts
        prev = True
    return p


def analyze(group, label, verbose=True):
    """Condition on (entry gate, path length): within each stratum compare
    the observed distribution of the remaining choices to the uniform null.
    R = P_obs(rest | entry, L) / P_null(rest | entry, L)."""
    n = len(group)
    strata = {}
    for path in group:
        strata.setdefault((path[0], len(path)), []).append(path)
    rows = []
    kl = 0.0
    for (entry, L), members in strata.items():
        m = len(members)
        counts = Counter(members)
        for path, c in counts.items():
            p_obs = c / m
            p_null = null_prob(path)
            r = p_obs / p_null
            rows.append((path, c, p_obs, p_null, r))
            kl += (m / n) * p_obs * np.log(r)
    rows.sort(key=lambda r: (-r[4], -r[1]))
    if verbose:
        print(f"\n== {label}: n={n}, {len(rows)} path types, "
              f"{len(strata)} strata ==")
        print("top by R (path, count, P_obs|stratum, P_null, R):")
        for path, c, po, pn, r in rows[:8]:
            print(f"  {'-'.join(path):24s} {c:3d}  {po:.3f}  {pn:.4f} "
                  f" R={r:5.1f}")
    mean_r = sum(c * r for _, c, _, _, r in rows) / n
    if verbose:
        print(f"pattern strength: mean R {mean_r:.1f}, KL {kl:.2f} nats")
    return mean_r, kl


crossers = [p for p in paths if len(p) == 2]
stoppers = [p for p in paths if len(p) > 2]
analyze(crossers, "crossers (no stops)")
analyze(stoppers, "stoppers (>=1 stop)")

# finite-sample bias check: KL estimated from n samples is biased upward.
# Compare each group's KL against KL of synthetic samples drawn from ITS OWN
# null (same strata sizes) — the excess over that baseline is real structure.
rng = np.random.default_rng(1)


def null_baseline(group, n_rep=500):
    n = len(group)
    strata = Counter((p[0], len(p)) for p in group)
    kls = []
    for _ in range(n_rep):
        synth = []
        for (entry, L), m in strata.items():
            for _ in range(m):
                exit_g = f"G{rng.integers(N_G)}"
                stops = []
                prev = -1
                for _ in range(L - 2):
                    opts = [a for a in range(len(attr)) if a != prev]
                    prev = opts[rng.integers(len(opts))]
                    stops.append(f"A{prev}")
                synth.append((entry, *stops, exit_g))
        kls.append(analyze(synth, "", verbose=False)[1])
    return np.mean(kls), np.std(kls)


for grp, lab in [(crossers, "crossers"), (stoppers, "stoppers")]:
    _, kl_obs = analyze(grp, lab, verbose=False)
    mu, sd = null_baseline(grp)
    print(f"{lab}: KL {kl_obs:.2f} vs null baseline {mu:.2f}+-{sd:.2f} "
          f"-> excess {kl_obs - mu:.2f} ({(kl_obs - mu) / sd:.1f} sigma)")
