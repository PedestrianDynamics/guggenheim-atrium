# Crowd dynamics in the Guggenheim Bilbao atrium

A 6.5-minute handheld phone video of the museum's atrium, filmed from an
upper balcony, turned into a crowd-science dataset and a simulation study:
331 pedestrian trajectories in metric coordinates, an analysis of how
visitors move and stop, and a JuPedSim reconstruction of the scene driven by
the measured demand.

`FINDINGS.md` holds the detailed results with exact numbers.

## What is here

- `scripts/` — the full pipeline, numbered in execution order:
  `01`–`06` tracking and metric rectification (YOLO11m + ByteTrack, shake
  registration, tile-grid calibration), `07`–`08` movement patterns and the
  demand model, `09`–`10` JuPedSim simulation and measured-vs-simulated
  comparison, `11`–`12` route-randomness test (after Yoshimura et al.) and
  attraction/holding power, `13`–`14` web-app scenario export and media.
- `data/` — extracted trajectories (`tracks_raw.parquet`,
  `tracks_world.parquet`), calibration transforms (`register.npz`,
  `rectify.npz`, `scale.txt`), the demand model (`demand.json`), and the
  simulation outputs.
- `figures/` — the analysis figures.
- `media/` — simulation animations.
- `guggenheim_atrium.zip` — the scenario (geometry, gates, stop points,
  arrival rates) for the JuPedSim web app: open it at
  [app.jupedsim.org](https://app.jupedsim.org) and run the atrium yourself.

## What is not here

The source video is not included: it is large and shows museum visitors.
The pipeline starts from it, so `01`–`06` are documentation of how the
trajectories were produced; everything from `07` on runs from the data in
this repository. The YOLO weights (`yolo11m.pt`) download automatically via
ultralytics.

## Running

```sh
pip install jupedsim pedpy ultralytics pandas shapely
python scripts/09_simulate.py   # rebuild the simulation from data/demand.json
python scripts/10_compare.py    # measured vs simulated figures
python scripts/11_route_null.py # route-randomness test
python scripts/12_attractors.py # attraction vs holding power
```

Each script reads from `data/` and writes its outputs next to the existing
ones.
