"""Render post media: measured-track overlay GIF and simulated-agents GIF.

measured_tracks.gif: 20 s of video with trajectories drawn on the frames
(world coords projected back through the rectification and registration
homographies). simulated_agents.gif: the same seconds of the JuPedSim run.
"""

import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FPS = 29.97
T0, T1 = 100.0, 120.0            # excerpt window [s]
TRAIL_S = 4.0
TILE_M = 0.732
XCORR = 1.287 * 1.103 * 1.032 * 1.009   # anisotropy factors from 06
OUT_W = 720

Hreg = np.load("data/register.npz")["H"]
Hm = np.load("data/rectify.npz")["H"]
Hm_inv = np.linalg.inv(Hm)

df = pd.read_parquet("data/tracks_world.parquet")
df = df[(df.t_s > T0 - TRAIL_S) & (df.t_s < T1)].copy()
df["xu"] = df.x_m / TILE_M / XCORR
df["yu"] = df.y_m / TILE_M
p0 = np.einsum("ij,nj->ni", Hm_inv, np.c_[df.xu, df.yu, np.ones(len(df))])
df["px0"] = p0[:, 0] / p0[:, 2]
df["py0"] = p0[:, 1] / p0[:, 2]

rng = np.random.default_rng(3)
colors = {tid: tuple(int(c) for c in rng.integers(60, 255, 3))
          for tid in df.id.unique()}

import sys
RENDER_MEASURED = "--sim-only" not in sys.argv

if RENDER_MEASURED:
    cap = cv2.VideoCapture("data/video_sdr.mp4")
    f_start, f_end = int(T0 * FPS), int(T1 * FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h_out = int(1080 * OUT_W / 1920)
    vw = cv2.VideoWriter("media/_measured_tmp.mp4", fourcc, 15,
                         (OUT_W, h_out))
    by_id = {tid: g.sort_values("frame") for tid, g in df.groupby("id")}
    for fi in range(f_start, f_end):
        ok, frame = cap.read()
        if not ok:
            break
        if fi % 2:                # 15 fps output
            continue
        Hinv_t = np.linalg.inv(Hreg[fi])
        for tid, g in by_id.items():
            seg = g[(g.frame <= fi) & (g.frame >= fi - TRAIL_S * FPS)]
            if len(seg) < 2 or seg.frame.iloc[-1] < fi - 3:
                continue
            pts = np.einsum("ij,nj->ni", Hinv_t,
                            np.c_[seg.px0, seg.py0, np.ones(len(seg))])
            pts = (pts[:, :2] / pts[:, 2:]).astype(np.int32)
            cv2.polylines(
                frame, [np.ascontiguousarray(pts[::3]).reshape(-1, 1, 2)],
                False, colors[tid], 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[-1]), 6, colors[tid], -1, cv2.LINE_AA)
        vw.write(cv2.resize(frame, (OUT_W, h_out)))
    vw.release()
    cap.release()

# simulated agents on the floor plan: colored by speed, fading trails
sim = pd.read_parquet("data/sim_world.parquet")
parts = []
for tid, g in sim.groupby("id"):
    g = g.sort_values("t_s").copy()
    if len(g) < 3:
        continue
    g["v"] = np.hypot(np.gradient(g.x_m, g.t_s), np.gradient(g.y_m, g.t_s))
    parts.append(g)
sim = pd.concat(parts, ignore_index=True)
sim = sim[(sim.t_s >= T0 - TRAIL_S) & (sim.t_s < T1)]

import json
from matplotlib.collections import LineCollection

floor = np.array(json.load(open("data/demand.json"))["walkable"])
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(0, 1.8)
by_sid = {tid: g.sort_values("t_s") for tid, g in sim.groupby("id")}
frames = sorted(sim.t_s[sim.t_s >= T0].unique())
for i, t in enumerate(frames):
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=100)
    ax.fill(floor[:, 0], floor[:, 1], color="#e9e2d4", zorder=0)
    ax.plot(floor[:, 0], floor[:, 1], color="#555", lw=1)
    for tid, g in by_sid.items():
        seg = g[(g.t_s <= t) & (g.t_s >= t - TRAIL_S)]
        if len(seg) < 2 or seg.t_s.iloc[-1] < t - 0.15:
            continue
        pts = seg[["x_m", "y_m"]].to_numpy().reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = cmap(norm(seg.v.to_numpy()[1:]))
        cols[:, 3] = np.linspace(0.05, 0.8, len(segs))   # fade out the tail
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=2,
                                         zorder=2, capstyle="round"))
        ax.scatter(seg.x_m.iloc[-1], seg.y_m.iloc[-1], s=46,
                   c=[cmap(norm(seg.v.iloc[-1]))], edgecolors="white",
                   linewidths=0.8, zorder=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("speed [m/s]", fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"JuPedSim, t = {t:5.1f} s", fontsize=11, family="monospace")
    fig.savefig(f"media/_sim_{i:04d}.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-framerate", "15", "-pattern_type",
     "glob", "-i", "media/_sim_*.png",
     "-vf", "scale=720:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
     "media/simulated_agents.gif"], check=True)
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-framerate", "15", "-pattern_type",
     "glob", "-i", "media/_sim_*.png",
     "-vf", "scale=720:-2:flags=lanczos,format=yuv420p",
     "-c:v", "libx264", "-crf", "22", "media/simulated_agents.mp4"],
    check=True)
if RENDER_MEASURED:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", "media/_measured_tmp.mp4",
         "-vf", "fps=12,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer",
         "media/measured_tracks.gif"], check=True)
    subprocess.run(["bash", "-c", "rm media/_measured_tmp.mp4"], check=True)
subprocess.run(["bash", "-c", "rm media/_sim_*.png"], check=True)
print("sim media done" + (", measured media done" if RENDER_MEASURED else ""))
