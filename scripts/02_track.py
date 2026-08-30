"""Detect and track pedestrians with YOLO11 + ByteTrack.

Writes one row per (frame, id): pixel coordinates of the bbox
bottom-center (ground contact point) plus bbox size and confidence.
"""

import pandas as pd
from ultralytics import YOLO

VIDEO = "data/video_sdr.mp4"
OUT = "data/tracks_raw.parquet"
MODEL = "yolo11m.pt"


def main():
    model = YOLO(MODEL)
    rows = []
    results = model.track(
        source=VIDEO,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],          # person
        imgsz=1920,
        conf=0.15,
        iou=0.5,
        device="mps",
        verbose=False,
    )
    for fi, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.int().tolist()
        confs = r.boxes.conf.tolist()
        for (x1, y1, x2, y2), tid, conf in zip(r.boxes.xyxy.tolist(), ids, confs):
            rows.append((fi, tid, (x1 + x2) / 2.0, y2, x2 - x1, y2 - y1, conf))
        if fi % 500 == 0:
            print(f"frame {fi}, tracks so far: {len(set(r[1] for r in rows))}",
                  flush=True)
    df = pd.DataFrame(rows, columns=["frame", "id", "x_px", "y_px",
                                     "w_px", "h_px", "conf"])
    df.to_parquet(OUT)
    print(f"done: {len(df)} rows, {df.id.nunique()} track ids -> {OUT}")


if __name__ == "__main__":
    main()
