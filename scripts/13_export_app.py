"""Export the atrium scenario as a zip for app.jupedsim.org.

Format mirrors jupedsim-scenarios' journeys.zip: geometry.wkt + config.json.
Demand is condensed for interactive use: per entry gate one direct journey
(to its most common exit) and one stop journey (via the two most-visited
attractors, to the second most common exit), weighted 68/32 like the
measured crossers/stoppers split. Attractor checkpoints wait ~5 s.
"""

import json
import zipfile

import numpy as np
from shapely.geometry import Point, Polygon

demand = json.load(open("data/demand.json"))
area = Polygon(demand["walkable"])
gates = np.array(demand["gates"])
od = np.array(demand["od"], float)
attr = np.array(demand["attractors"])
attr_p = np.array(demand["attractor_p"])
entries = demand["entries"]
DUR = int(demand["duration"])
MEAN_DWELL = 5.4


def circle(center, r, n=32):
    th = np.linspace(0, 2 * np.pi, n + 1)
    return [[round(center[0] + r * np.cos(t), 3),
             round(center[1] + r * np.sin(t), 3)] for t in th]


def gate_square(g, size=2.4):
    c = Point(gates[g])
    if area.exterior.distance(c) < size / 2 + 0.1:
        # nudge inward along the direction to the area's representative point
        rp = np.array(area.representative_point().coords[0])
        d = rp - gates[g]
        c = Point(gates[g] + d / np.linalg.norm(d) * (size / 2 + 0.3))
    x, y = c.x, c.y
    h = size / 2
    return [[round(x - h, 2), round(y - h, 2)], [round(x + h, 2), round(y - h, 2)],
            [round(x + h, 2), round(y + h, 2)], [round(x - h, 2), round(y + h, 2)],
            [round(x - h, 2), round(y - h, 2)]]


exits = {f"jps-exits_{g}": {
    "type": "polygon", "coordinates": gate_square(g),
    "enable_throughput_throttling": False, "max_throughput": 0,
} for g in range(len(gates))}

checkpoints = {}
top_attr = np.argsort(-attr_p)
for k in range(len(attr)):
    c = attr[k].tolist()
    checkpoints[f"jps-checkpoints_{k}"] = {
        "type": "polygon", "coordinates": circle(c, 1.2),
        "waiting_time": MEAN_DWELL, "waiting_time_distribution": "constant",
        "waiting_time_std": 1, "enable_throughput_throttling": False,
        "max_throughput": 1, "speed_factor": 1,
        "shape": "circle", "center": [round(c[0], 3), round(c[1], 3)],
        "radius": 1.2,
    }

journeys = []
distributions = {}
n_per_gate = np.bincount([e[1] for e in entries], minlength=len(gates))
for g in range(len(gates)):
    row = od[g].copy()
    row[row.sum() == 0] = 1
    order = np.argsort(-row)
    exit1, exit2 = int(order[0]), int(order[1] if len(order) > 1 else order[0])
    j_direct = {
        "id": f"journey-g{g}-direct", "name": f"G{g} direct", "color": "#4F46E5",
        "sequence": [f"jps-exits_{exit1}"],
    }
    j_stops = {
        "id": f"journey-g{g}-stops", "name": f"G{g} with stops",
        "color": "#DB2777",
        "sequence": [f"jps-checkpoints_{int(top_attr[0])}",
                     f"jps-checkpoints_{int(top_attr[1])}",
                     f"jps-exits_{exit2}"],
    }
    journeys += [j_direct, j_stops]
    distributions[f"jps-distributions_{g}"] = {
        "type": "polygon", "coordinates": gate_square(g),
        "parameters": {
            "number": int(n_per_gate[g]), "radius": 0.15, "v0": 1.3,
            "use_flow_spawning": True, "flow_start_time": 0,
            "flow_end_time": DUR, "distribution_mode": "by_number",
            "percentage": None, "use_premovement": False,
            "premovement_distribution": "gamma", "premovement_param_a": None,
            "premovement_param_b": None, "premovement_seed": None,
            "radius_distribution": "constant",
            "v0_distribution": "gaussian", "v0_std": 0.33,
        },
        "journey_weights": [
            {"journey_id": f"journey-g{g}-direct", "weight": 68},
            {"journey_id": f"journey-g{g}-stops", "weight": 32},
        ],
    }

config = {
    "project_version": "2.0",
    "config": {
        "simulation_settings": {
            "simulationParams": {
                "max_simulation_time": DUR + 120, "dt": 0.01,
                "model_type": "CollisionFreeSpeedModel",
                "strength_neighbor_repulsion": 2.6,
                "range_neighbor_repulsion": 0.1, "mass": 80, "tau": 0.5,
                "a_v": 1, "a_min": 0.2, "b_min": 0.2, "b_max": 0.4,
                "relaxation_time": 0.5, "T": 1, "s0": 0.5,
            },
            "numberOfSimulations": 1, "baseSeed": 42,
        },
        "ui_state": {"useShortestPaths": True,
                     "boundaries": [{"mode": "manual"}]},
    },
    "exits": exits,
    "distributions": distributions,
    "checkpoints": checkpoints,
    "zones": {},
    "journeys": [],
    "transitions": [],
    "obstacles": {},
    "journeys_v2": journeys,
}

coords = ", ".join(f"{round(x, 2)} {round(y, 2)}"
                   for x, y in area.exterior.coords)
wkt = f"POLYGON(({coords}))"

with zipfile.ZipFile("guggenheim_atrium.zip", "w",
                     zipfile.ZIP_DEFLATED) as z:
    z.writestr("config.json", json.dumps(config, indent=1))
    z.writestr("geometry.wkt", wkt)
print(f"gates {len(gates)}, checkpoints {len(checkpoints)}, "
      f"journeys {len(journeys)}, agents {int(n_per_gate.sum())} "
      f"over {DUR}s -> guggenheim_atrium.zip")
