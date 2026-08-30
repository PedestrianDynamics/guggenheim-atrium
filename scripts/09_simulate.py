"""JuPedSim simulation of the atrium driven by the empirical demand model.

Strategic level from data (data/demand.json): entries (time, gate), exit gate
sampled from the OD row, number of stops / attractor choice / dwell / desired
speed sampled from fitted distributions. Operational level: Collision Free
Speed Model. Tactical level: RoutingEngine waypoints between itinerary
targets, executed via direct steering.

Output: data/sim_trajectories.sqlite (+ parquet mirror for the comparison).
"""

import json

import jupedsim as jps
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

SEED = 42
DT = 0.05
OUT_FPS = 10                     # trajectory sampling for analysis
REACH = 0.9                      # m, target reached
rng = np.random.default_rng(SEED)

demand = json.load(open("data/demand.json"))
area = Polygon(demand["walkable"])
gates = np.array(demand["gates"])
od = np.array(demand["od"], float)
attr = np.array(demand["attractors"])
attr_p = np.array(demand["attractor_p"])
attr_p /= attr_p.sum()
p_nstops = np.array(demand["p_nstops"]); p_nstops /= p_nstops.sum()
mu, sig = demand["dwell_lognorm"]
dwell_samples = np.array(demand["dwell_samples"])
v_mean, v_std = demand["speed_mean_std"]


def inside(p):
    return area.contains(Point(p))


def clamp_inside(p):
    if inside(p):
        return tuple(p)
    q = area.exterior.interpolate(area.exterior.project(Point(p)))
    c = area.representative_point()
    q = np.array(q.coords[0])
    d = np.array(c.coords[0]) - q
    return tuple(q + 0.6 * d / np.linalg.norm(d))


def build_itinerary(g_in):
    row = od[g_in]
    row = row / row.sum() if row.sum() else np.ones(len(gates)) / len(gates)
    g_out = rng.choice(len(gates), p=row)
    n_stop = rng.choice(len(p_nstops), p=p_nstops)
    stops = rng.choice(len(attr), size=n_stop, replace=False, p=attr_p)
    targets = [(clamp_inside(attr[s]), float(rng.choice(dwell_samples)))
               for s in stops]
    targets.append((clamp_inside(gates[g_out]), 0.0))
    return targets


class Visitor:
    def __init__(self, aid, itinerary, v0):
        self.aid = aid
        self.itinerary = itinerary       # [(point, dwell_s), ...]
        self.leg = None                  # routed waypoints of current leg
        self.wait_until = -1.0
        self.anchor = None               # hold position during a dwell
        self.v0 = v0
        self.done = False

    def current_target(self):
        return self.itinerary[0][0]


sim = jps.Simulation(
    model=jps.CollisionFreeSpeedModel(),
    geometry=area,
    dt=DT,
    trajectory_writer=jps.SqliteTrajectoryWriter(
        output_file=__import__("pathlib").Path("data/sim_trajectories.sqlite"),
        every_nth_frame=int(1 / (OUT_FPS * DT)),
    ),
)
routing = jps.RoutingEngine(area)
ds_stage = sim.add_direct_steering_stage()
journey = jps.JourneyDescription([ds_stage])
journey_id = sim.add_journey(journey)

entries = demand["entries"]
visitors = {}
n_dwells = 0
n_stop_targets = 0
ei = 0
t_end = demand["duration"]
max_iter = int((t_end + 120) / DT)      # grace period to clear the floor

while sim.iteration_count() < max_iter:
    t = sim.iteration_count() * DT
    # spawn
    while ei < len(entries) and entries[ei][0] <= t:
        t0, g = entries[ei]
        ei += 1
        pos = clamp_inside(gates[g] + rng.normal(0, 0.5, 2))
        v = float(np.clip(rng.normal(v_mean, v_std), 0.6, 2.2))
        try:
            aid = sim.add_agent(jps.CollisionFreeSpeedModelAgentParameters(
                position=pos, journey_id=journey_id, stage_id=ds_stage,
                desired_speed=v, radius=0.18))
        except RuntimeError:
            continue                     # spawn spot momentarily blocked
        visitors[aid] = Visitor(aid, build_itinerary(g), v)
        n_stop_targets += len(visitors[aid].itinerary) - 1
    if t > t_end and sim.agent_count() == 0:
        break
    # steer
    removed = []
    for aid, vis in visitors.items():
        if vis.done:
            continue
        try:
            agent = sim.agent(aid)
        except RuntimeError:
            vis.done = True
            continue
        p = np.array(agent.position)
        if vis.wait_until > t:
            agent.target = vis.anchor    # hold position, v0 = 0
            continue
        if vis.anchor is not None:       # dwell just ended
            vis.anchor = None
            agent.model.desired_speed = vis.v0
        tgt = np.array(vis.current_target())
        if np.linalg.norm(p - tgt) < REACH:
            _, dwell = vis.itinerary.pop(0)
            vis.leg = None
            if not vis.itinerary:        # exit reached
                removed.append(aid)
                vis.done = True
                continue
            if dwell > 0:
                vis.wait_until = t + dwell
                vis.anchor = tuple(p)
                agent.model.desired_speed = 0.0
                agent.target = vis.anchor
                n_dwells += 1
                continue
            tgt = np.array(vis.current_target())
        if vis.leg is None:
            wp = routing.compute_waypoints(tuple(p), tuple(tgt))
            vis.leg = [np.array(w) for w in wp[1:]]
        while len(vis.leg) > 1 and np.linalg.norm(p - vis.leg[0]) < REACH:
            vis.leg.pop(0)
        agent.target = tuple(vis.leg[0])
    for aid in removed:
        sim.mark_agent_for_removal(aid)
    sim.iterate()

print(f"finished at t={sim.iteration_count() * DT:.0f}s, "
      f"agents left {sim.agent_count()}, spawned {len(visitors)}")
print(f"stop targets assigned {n_stop_targets}, dwells executed {n_dwells}")

# mirror to parquet in the measured format
import sqlite3
con = sqlite3.connect("data/sim_trajectories.sqlite")
tr = pd.read_sql("select frame, id, pos_x as x_m, pos_y as y_m "
                 "from trajectory_data", con)
fps_row = pd.read_sql("select value from metadata where key='fps'", con)
con.close()
sim_fps = float(fps_row.value.iloc[0])
tr["t_s"] = tr.frame / sim_fps
tr.to_parquet("data/sim_world.parquet")
print(f"sim fps {sim_fps}, rows {len(tr)}, ids {tr.id.nunique()} "
      f"-> data/sim_world.parquet")
