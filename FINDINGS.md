# Guggenheim Bilbao atrium — movement patterns from a phone video

Source: `IMG_2392.MOV`, 6.5 min handheld from an upper balcony, 2026-08-19.
Pipeline: YOLO11m + ByteTrack → per-frame shake registration → vanishing-point
metric rectification of the tile grid → cleaned world trajectories
(331 tracks, ~275k samples). Scale calibrated from the modal walking speed
(1.30 m/s); tile module ≈ 0.73 m; direction-dependence of speed used to
correct the running-bond grid anisotropy (within-person residual < 2%).

## Is the movement random? No.

- **MSD slope 1.83** (0.5–8 s): strongly superdiffusive, near-ballistic;
  saturates only at ~20 s from the room boundary. A random walk would give 1.
- **82% of 1-s steps turn < 30°** — people walk in directed, persistent legs.
- The "random-looking" impression comes from *composition*: many directed
  individual paths with different goals, plus one third standing.

## Structure found

- **Time budget**: 33% of all person-time is standing (v < 0.3 m/s).
  Speed distribution is bimodal: standing peak + walking mode at ~1.35 m/s.
- **Stop episodes** are short and heavy-tailed: median 2.4 s, p90 7.8 s,
  n=552 (>1 s). Long stops (>30 s) are rare and localized.
- **Standing hotspots** are not random: base of the glass elevator tower,
  spots next to the floor info placards, and free-floor cluster points
  (tour-group gatherings, incl. the sunlit patch in the atrium center).
- **Flow field**: interior is a low-polarization mixing zone (median cell
  polarization 0.25, only 3% of cells > 0.7). Two polarized streams hug the
  boundary: the upper-left corridor toward the Ruth Asawa gallery entrance
  and the lower edge toward the museum entrance/exit side. Classic
  "edge highways, interior plaza" pattern.
- **Groups**: 13% of tracks form coherent pairs/groups (>20 s co-proximity
  < 2 m with aligned velocities). Grouped visitors walk slower:
  1.05 vs 1.23 m/s median walking speed. Standing share is similar
  (31% vs 34%).

## Strategic-level randomness test (script 11, after Yoshimura et al. 2019)

Each boundary-OD track becomes a symbolic route [gate, stops..., gate] over
5 gates + 8 attractors; the null model is a random walk on that zone graph,
conditioned on entry gate and path length (necessary with multiple gates —
Yoshimura's single-entrance R-value transfers only after this conditioning).

- **Both visitor types are significantly non-random.** Crossers (no stops,
  n=134): KL 0.37 nats vs finite-sample null baseline 0.08±0.03 — 11σ excess.
  Stoppers (n=63): KL 2.57 vs baseline 2.12±0.05 — 9σ excess.
- **Crossers concentrate on few corridors** (R≈3 for G2→G3, G4→G0, G1→G0):
  directed transit, the atrium used as a thoroughfare.
- **Stoppers are diverse**: 43 route types among 63 tracks, most unique —
  mirroring Yoshimura's Louvre finding that short-stay visitors follow strong
  common patterns while long-stay visitors individualize. Our transit/dwell
  split reproduces their short/long-stay contrast inside a single hall.

## Attraction vs holding power (script 12, cf. Melton; Centorrino et al. 2021)

Per attractor: attraction power = share of tracks stopping there; holding
power = mean dwell. The two are **not the same thing** (r = −0.41): A5
(center-south, 8.5% of tracks) attracts most but holds ~6 s; A3 (east edge,
2.7%) attracts few but holds them longest (mean 14.7 s, median 9.9 s).
Dwell times are **lognormal, not Weibull**: AIC 1597 (lognormal) vs 1751
(Weibull, shape k=1.28) vs 1786 (exponential) — a contrast to Galleria
Borghese room-permanence times (Weibull, Centorrino et al.), plausibly
because atrium stops are pauses-in-transit rather than room visits.

## JuPedSim comparison (scripts 08–10)

Setup: Collision-Free Speed model; strategic demand purely distribution-level
from the data (entry times/gates of the 197 boundary-OD tracks, OD matrix,
stops-per-track pmf, attractor locations + visit probabilities, bootstrapped
dwell samples, desired speed N(1.30, 0.33)); tactical level via RoutingEngine
waypoints and direct steering; dwells as anchored holds with desired_speed=0.
No measured path is replayed. Seed 42, dt 0.05 s, output 10 fps.

What the model reproduces:
- **Dynamics**: MSD slope 1.95 vs 1.83; speed distribution walking mode matches
  by construction, shape agrees.
- **Flow structure**: polarization median 0.30 vs 0.25, >0.7 cells 7% vs 5% —
  the "no one-way streets, mixing interior" character carries over.
- **Stop geography**: standing-map correlation r=0.41; the simulated hotspots
  sit on the measured ones (they are the model's attractors, so this mainly
  validates the pipeline, not the model).

Where it fails, instructively:
- **Occupancy r=0.42**: simulated flow collapses onto gate-to-gate
  shortest-path *spokes*; measured occupancy is much more diffuse — real
  visitors meander and explore, spending time far from any ideal path.
- **Standing time 17% vs 29%** (like-for-like population): the demand model
  only encodes discrete stops >2 s; measured standing includes shuffling,
  micro-pauses and slow reorientation that a constant-desired-speed model
  cannot produce.

Takeaway: goal-driven shortest-path agents + empirical stop program get the
*statistics of motion* right but not the *spatial spread* of museum behavior;
an exploration/meander term (e.g. spatially varying desired direction noise
or intermediate soft attractors) is the missing ingredient.

## Caveats

- Monocular bbox-foot tracking: occlusion spikes were despiked/clipped but
  residual noise inflates short-lag speeds; per-99th-percentile speeds are
  unreliable.
- Absolute scale rests on the 1.30 m/s walking-mode assumption; relative
  geometry (aspect, angles) is calibrated independently from the tile grid.
- 6.5 min, one afternoon — no claim beyond this observation window.

## Reproduce

`venv` in `study/`; scripts run in order 01–10 from `study/`
(01 steadiness check, 02 track, 03 register, 04 background, 05 rectify,
06 world+clean, 07 analysis, 08 demand extraction, 09 JuPedSim simulation,
10 measured-vs-simulated comparison). Figures in `figures/`.
