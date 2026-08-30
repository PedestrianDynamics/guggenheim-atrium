"""Measured vs JuPedSim: occupancy maps, speed distribution, MSD, flow field.

Figures -> figures/sim_vs_measured_*.png, numbers -> stdout.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAND_V = 0.3
BINS = [np.arange(-13, 15.5, 1.0), np.arange(-10, 19.5, 1.0)]


def load(path):
    df = pd.read_parquet(path)
    out = []
    for _, g in df.groupby("id"):
        g = g.sort_values("t_s").copy()
        if len(g) < 5:
            continue
        g["vx"] = np.gradient(g.x_m, g.t_s)
        g["vy"] = np.gradient(g.y_m, g.t_s)
        g["v"] = np.hypot(g.vx, g.vy)
        out.append(g)
    return pd.concat(out, ignore_index=True)


meas = load("data/tracks_world.parquet")
sim = load("data/sim_world.parquet")
dt_meas = np.median(np.diff(meas.t_s.unique()))
dt_sim = np.median(np.diff(sim.t_s.unique()))

# ------------------------------------------------ occupancy + standing maps
def occ(df, dt, standing=False):
    m = df.v < STAND_V if standing else np.ones(len(df), bool)
    h, *_ = np.histogram2d(df.x_m[m], df.y_m[m], bins=BINS)
    return h * dt


fig, axes = plt.subplots(2, 2, figsize=(13, 12), sharex=True, sharey=True)
ext = [BINS[0][0], BINS[0][-1], BINS[1][-1], BINS[1][0]]
panels = [
    (occ(meas, dt_meas), "measured occupancy"),
    (occ(sim, dt_sim), "simulated occupancy"),
    (occ(meas, dt_meas, True), "measured standing"),
    (occ(sim, dt_sim, True), "simulated standing"),
]
for ax, (h, title) in zip(axes.ravel(), panels):
    im = ax.imshow(np.sqrt(h.T), extent=ext, cmap="inferno", aspect="equal")
    ax.set_title(title + " [sqrt person·s]")
    fig.colorbar(im, ax=ax, shrink=0.75)
fig.savefig("figures/sim_vs_measured_maps.png", dpi=130, bbox_inches="tight")
plt.close(fig)

h_m = occ(meas, dt_meas).ravel()
h_s = occ(sim, dt_sim).ravel()
use = (h_m > 0) | (h_s > 0)
r_occ = np.corrcoef(np.sqrt(h_m[use]), np.sqrt(h_s[use]))[0, 1]
hs_m = occ(meas, dt_meas, True).ravel()
hs_s = occ(sim, dt_sim, True).ravel()
r_stand = np.corrcoef(np.sqrt(hs_m[use]), np.sqrt(hs_s[use]))[0, 1]
print(f"occupancy map correlation (sqrt, 1 m cells): r={r_occ:.2f}")
print(f"standing map correlation: r={r_stand:.2f}")

# ------------------------------------------------ speed distribution + MSD
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for df, lab in [(meas, "measured"), (sim, "simulated")]:
    axes[0].hist(df.v[df.v < 3], bins=60, density=True, alpha=0.55, label=lab)
axes[0].axvline(STAND_V, c="r", ls="--", lw=1)
axes[0].set_xlabel("v [m/s]")
axes[0].set_title("speed distribution")
axes[0].legend()


def msd_curve(df, dt):
    lags = np.unique(np.geomspace(1, 20 / dt, 30).astype(int))
    msd = np.zeros(len(lags))
    cnt = np.zeros(len(lags))
    for _, g in df.groupby("id"):
        x, y = g.x_m.to_numpy(), g.y_m.to_numpy()
        for k, L in enumerate(lags):
            if len(x) > L:
                msd[k] += ((x[L:] - x[:-L]) ** 2 + (y[L:] - y[:-L]) ** 2).mean()
                cnt[k] += 1
    return lags * dt, msd / np.maximum(cnt, 1)


for df, dt, lab in [(meas, dt_meas, "measured"), (sim, dt_sim, "simulated")]:
    tau, m = msd_curve(df, dt)
    axes[1].loglog(tau, m, "o-", ms=3, label=lab)
    band = (tau > 0.5) & (tau < 8)
    sl = np.polyfit(np.log(tau[band]), np.log(m[band]), 1)[0]
    print(f"MSD slope 0.5-8 s ({lab}): {sl:.2f}")
axes[1].set_xlabel("lag [s]")
axes[1].set_ylabel("MSD [m$^2$]")
axes[1].set_title("mean squared displacement")
axes[1].legend()
fig.savefig("figures/sim_vs_measured_dynamics.png", dpi=130,
            bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ flow field polarization
def flow(df):
    cell = 1.5
    gx = np.arange(-13, 15, cell)
    gy = np.arange(-10, 19, cell)
    U = np.full((len(gy), len(gx)), np.nan)
    V = np.full_like(U, np.nan)
    C = np.full_like(U, np.nan)
    w = df[df.v >= STAND_V]
    ix = np.digitize(w.x_m, gx) - 1
    iy = np.digitize(w.y_m, gy) - 1
    for (j, i), g in w.groupby([iy, ix]):
        if 0 <= j < len(gy) and 0 <= i < len(gx) and len(g) * 1 > 20:
            U[j, i] = g.vx.mean()
            V[j, i] = g.vy.mean()
            C[j, i] = np.hypot(
                *(np.c_[g.vx, g.vy] / g.v.values[:, None]).mean(0))
    return gx, gy, U, V, C, cell


fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
pols = {}
for ax, (df, lab) in zip(axes, [(meas, "measured"), (sim, "simulated")]):
    gx, gy, U, V, C, cell = flow(df)
    im = ax.imshow(C, extent=[gx[0], gx[-1] + cell, gy[-1] + cell, gy[0]],
                   cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.6)
    ax.quiver(gx + cell / 2, gy + cell / 2, U, -V, scale=12, width=0.004)
    ax.set_title(f"{lab}: flow + polarization")
    ax.set_aspect("equal")
    pols[lab] = C[~np.isnan(C)]
fig.colorbar(im, ax=axes, shrink=0.6)
fig.savefig("figures/sim_vs_measured_flow.png", dpi=130, bbox_inches="tight")
plt.close(fig)
for lab, p in pols.items():
    print(f"polarization ({lab}): median {np.median(p):.2f}, "
          f">0.7: {(p > 0.7).mean() * 100:.0f}%")

for df, dt, lab in [(meas, dt_meas, "measured"), (sim, dt_sim, "simulated")]:
    print(f"standing fraction ({lab}): {(df.v < STAND_V).mean() * 100:.0f}%  "
          f"person-time {len(df) * dt:.0f} s")
print("figures written")
