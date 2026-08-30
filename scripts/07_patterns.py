"""Pattern analysis on the world trajectories.

Figures -> figures/, headline numbers -> stdout (captured into FINDINGS.md).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FPS = 29.97
STAND_V = 0.3        # m/s below which a person is "standing"
df = pd.read_parquet("data/tracks_world.parquet")

# per-point velocity (central difference per track)
def add_kinematics(df):
    out = []
    for tid, g in df.groupby("id"):
        g = g.sort_values("frame").copy()
        g["vx"] = np.gradient(g.x_m, g.t_s)
        g["vy"] = np.gradient(g.y_m, g.t_s)
        g["v"] = np.hypot(g.vx, g.vy)
        out.append(g)
    return pd.concat(out, ignore_index=True)


df = add_kinematics(df)
walk = df.v >= STAND_V

# ---------------------------------------------------------------- fig 1
# density heatmap + stop map + mean-speed map
bins = [np.arange(-13, 15.5, 0.75), np.arange(-10, 19.5, 0.75)]
fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=True)
h_all, xe, ye = np.histogram2d(df.x_m, df.y_m, bins=bins)
h_stand, *_ = np.histogram2d(df.x_m[~walk], df.y_m[~walk], bins=bins)
h_v, *_ = np.histogram2d(df.x_m[walk], df.y_m[walk], bins=bins,
                         weights=df.v[walk])
h_n, *_ = np.histogram2d(df.x_m[walk], df.y_m[walk], bins=bins)
ext = [xe[0], xe[-1], ye[-1], ye[0]]
for ax, data, title, cmap in [
        (axes[0], (h_all / FPS).T, "occupancy [person·s]", "inferno"),
        (axes[1], (h_stand / FPS).T, "standing time [person·s]", "inferno"),
        (axes[2], np.divide(h_v, h_n, out=np.full_like(h_v, np.nan),
                            where=h_n > 30).T, "mean walking speed [m/s]",
         "viridis")]:
    im = ax.imshow(data, extent=ext, cmap=cmap, aspect="equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.savefig("figures/maps_density_stop_speed.png", dpi=130,
            bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2
# flow field: mean velocity vector per cell
cell = 1.5
gx = np.arange(-13, 15, cell)
gy = np.arange(-10, 19, cell)
U = np.full((len(gy), len(gx)), np.nan)
V = np.full_like(U, np.nan)
C = np.zeros_like(U)
w = df[walk]
ix = np.digitize(w.x_m, gx) - 1
iy = np.digitize(w.y_m, gy) - 1
for (j, i), g in w.groupby([iy, ix]):
    if 0 <= j < len(gy) and 0 <= i < len(gx) and len(g) > 60:
        U[j, i] = g.vx.mean()
        V[j, i] = g.vy.mean()
        # polarization: |mean unit vector| in cell (1 = one-way street)
        C[j, i] = np.hypot(*(np.c_[g.vx, g.vy] / g.v.values[:, None]).mean(0))
fig, ax = plt.subplots(figsize=(10, 11))
bgc = ax.imshow(C, extent=[gx[0], gx[-1] + cell, gy[-1] + cell, gy[0]],
                cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.6)
ax.quiver(gx + cell / 2, gy + cell / 2, U, -V, scale=12, width=0.004)
ax.set_title("mean flow field; color = polarization (1 = one-way)")
fig.colorbar(bgc, ax=ax, shrink=0.7)
ax.set_aspect("equal")
fig.savefig("figures/flow_field.png", dpi=130, bbox_inches="tight")
plt.close(fig)
pol = C[C > 0]
print(f"polarization: median {np.median(pol):.2f}, "
      f"cells>0.7: {(pol > 0.7).mean() * 100:.0f}%")

# ---------------------------------------------------------------- fig 3
# random-walk tests: MSD, turning angles, speed distribution, dwell times
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# MSD over lag, averaged over tracks
lags = np.unique(np.geomspace(1, 20 * FPS, 40).astype(int))
msd = np.zeros(len(lags))
cnt = np.zeros(len(lags))
for _, g in df.groupby("id"):
    x, y = g.x_m.to_numpy(), g.y_m.to_numpy()
    for k, L in enumerate(lags):
        if len(x) > L:
            d = (x[L:] - x[:-L]) ** 2 + (y[L:] - y[:-L]) ** 2
            msd[k] += d.mean()
            cnt[k] += 1
msd /= np.maximum(cnt, 1)
tau = lags / FPS
axes[0, 0].loglog(tau, msd, "o-", ms=3)
sl = np.polyfit(np.log(tau[(tau > 0.5) & (tau < 8)]),
                np.log(msd[(tau > 0.5) & (tau < 8)]), 1)[0]
axes[0, 0].loglog(tau, msd[10] * (tau / tau[10]) ** 2, "--", c="gray",
                  label="ballistic (slope 2)")
axes[0, 0].loglog(tau, msd[10] * (tau / tau[10]), ":", c="gray",
                  label="diffusive (slope 1)")
axes[0, 0].set_title(f"MSD, slope 0.5-8 s = {sl:.2f}")
axes[0, 0].set_xlabel("lag [s]")
axes[0, 0].set_ylabel("MSD [m$^2$]")
axes[0, 0].legend()
print(f"MSD slope (0.5-8 s): {sl:.2f}")

# turning angle distribution over 1 s steps (walking only)
angs = []
step = int(FPS)
for _, g in df.groupby("id"):
    x, y = g.x_m.to_numpy(), g.y_m.to_numpy()
    dx = x[step::step] - x[:-step:step]
    dy = y[step::step] - y[:-step:step]
    ln = np.hypot(dx, dy)
    ok = ln > 0.5
    th = np.arctan2(dy, dx)
    dth = np.diff(th)
    keep = ok[1:] & ok[:-1]
    angs.append((dth[keep] + np.pi) % (2 * np.pi) - np.pi)
angs = np.concatenate(angs)
axes[0, 1].hist(np.degrees(angs), bins=72, density=True)
axes[0, 1].set_title("turning angle per 1 s (walking)")
axes[0, 1].set_xlabel("turn [deg]")
frac_straight = (np.abs(np.degrees(angs)) < 30).mean()
print(f"turning: {frac_straight * 100:.0f}% of 1-s steps turn < 30 deg")

# speed distribution
axes[1, 0].hist(df.v[df.v < 3], bins=90, density=True)
axes[1, 0].axvline(STAND_V, c="r", ls="--")
axes[1, 0].set_title("speed distribution")
axes[1, 0].set_xlabel("v [m/s]")
stand_frac = (~walk).mean()
print(f"standing fraction of all person-time: {stand_frac * 100:.0f}%")

# dwell (stop) episode durations
stops = []
for _, g in df.groupby("id"):
    standing = (g.v < STAND_V).to_numpy()
    d = np.diff(np.r_[0, standing.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    stops.extend((ends - starts) / FPS)
stops = np.array(stops)
stops = stops[stops > 1]
axes[1, 1].hist(stops, bins=np.geomspace(1, 200, 30), density=True)
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("stop episode duration")
axes[1, 1].set_xlabel("duration [s]")
print(f"stops>1s: n={len(stops)}, median {np.median(stops):.1f}s, "
      f"p90 {np.percentile(stops, 90):.1f}s")
fig.savefig("figures/randomwalk_tests.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4
# social structure: pair distance + velocity coherence -> groups
present = df.groupby("frame")["id"].apply(set)
pairs = {}
sample = df[df.frame % 15 == 0]
for f, g in sample.groupby("frame"):
    arr = g[["id", "x_m", "y_m", "vx", "vy", "v"]].to_numpy()
    for a in range(len(arr)):
        for b in range(a + 1, len(arr)):
            d = np.hypot(arr[a, 1] - arr[b, 1], arr[a, 2] - arr[b, 2])
            if d < 2.0:
                key = (int(arr[a, 0]), int(arr[b, 0]))
                co = 0.0
                if arr[a, 5] > 0.3 and arr[b, 5] > 0.3:
                    co = (arr[a, 3] * arr[b, 3] + arr[a, 4] * arr[b, 4]) / (
                        arr[a, 5] * arr[b, 5])
                rec = pairs.setdefault(key, [0, 0, 0])
                rec[0] += 1                      # co-proximate samples
                rec[1] += co
                rec[2] += 1 if co > 0.7 else 0
# group pairs: >20 s near each other and coherent when moving
group_pairs = {k: v for k, v in pairs.items()
               if v[0] * 15 / FPS > 20 and v[2] / v[0] > 0.35}
ids_in_groups = set(i for k in group_pairs for i in k)
n_ids = df.id.nunique()
print(f"group pairs: {len(group_pairs)}; "
      f"{len(ids_in_groups)}/{n_ids} tracked ids in a group "
      f"({len(ids_in_groups) / n_ids * 100:.0f}%)")

# speed of grouped vs solo (walking only)
grp = df[walk & df.id.isin(ids_in_groups)]
solo = df[walk & ~df.id.isin(ids_in_groups)]
print(f"median walking speed: grouped {grp.v.median():.2f} m/s, "
      f"solo {solo.v.median():.2f} m/s")

# stop share
grp_all = df[df.id.isin(ids_in_groups)]
solo_all = df[~df.id.isin(ids_in_groups)]
print(f"standing share: grouped {(grp_all.v < STAND_V).mean() * 100:.0f}%, "
      f"solo {(solo_all.v < STAND_V).mean() * 100:.0f}%")

print("figures written to figures/")
