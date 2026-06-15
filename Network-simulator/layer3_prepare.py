#!/usr/bin/env python3
"""layer3_prepare.py
Extract a single flight's trajectory from AADM/AFAR processed CSVs and write
a sim-frame trajectory.csv consumable by the _ReplayMission.

Coordinate convention (NO lat/lon projection):
    BS at (0, 0, 0).
    UAV at (distance_to_bs_m, 0, altitude_m).
Path-loss only needs slant distance + UAV altitude; this preserves both
without GPS-to-Cartesian drift.

Also emits ground_truth.csv with measured (time_rel_s, power_dBm, snr_dB)
for layer3_analysis.py to join against sim output.

Usage:
    python3 layer3_prepare.py --list                       # list flights
    python3 layer3_prepare.py --flight_id testbed_1004vol1_flight28 [--source aadm_testbed] [--alt_min 5]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path("../Datasets/Finetuning-processed")
OUT  = Path("../results/layer3")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "aadm_testbed":      ROOT / "aadm_testbed.csv",
    "aadm_development":  ROOT / "aadm_development.csv",
    "afar_testbed":      ROOT / "afar_testbed.csv",
    "afar_all_flights":  ROOT / "afar_all_flights.csv",
}


def _list_flights():
    for name, path in SOURCES.items():
        if not path.exists():
            continue
        print(f"\n=== {name} ===")
        df = pd.read_csv(path, usecols=lambda c: c in
                         ("flight_id", "altitude_m", "distance_to_bs_m",
                          "distance_to_tx_m"))
        if "flight_id" not in df.columns:
            print(f"  no flight_id column"); continue
        agg = df.groupby("flight_id").size().rename("n")
        for fid, n in agg.items():
            print(f"  {fid}  rows={n}")


def _prepare_one(source: str, flight_id: str, alt_min: float):
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source}; choices={list(SOURCES)}")
    path = SOURCES[source]
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    print(f"reading {path} (this is large — may take ~10s)...")
    df = pd.read_csv(path, low_memory=False)
    df = df[df["flight_id"] == flight_id].copy()
    if len(df) == 0:
        raise SystemExit(f"no rows for flight_id={flight_id}")
    print(f"  raw rows: {len(df):,}")

    # distance column varies by source
    if "distance_to_bs_m" in df.columns:
        dist_col = "distance_to_bs_m"
    elif "distance_to_tx_m" in df.columns:
        dist_col = "distance_to_tx_m"
    else:
        raise SystemExit("no distance column found")

    # filter ground samples
    if alt_min > 0:
        before = len(df)
        df = df[df["altitude_m"] >= alt_min].copy()
        print(f"  altitude >= {alt_min}m filter: {before} -> {len(df)}")
    df = df.sort_values("time_rel_s").reset_index(drop=True)

    # drop dup / non-monotone timestamps
    df = df.drop_duplicates(subset=["time_rel_s"], keep="first").reset_index(drop=True)
    print(f"  after dedup: {len(df):,}")
    if len(df) < 2:
        raise SystemExit("fewer than 2 usable samples")

    # trajectory.csv — what _ReplayMission consumes
    traj = pd.DataFrame({
        "time_rel_s": df["time_rel_s"].astype(float),
        "x_m":        df[dist_col].astype(float),
        "y_m":        0.0,
        "z_m":        df["altitude_m"].astype(float),
    })
    traj_path = OUT / f"trajectory_{flight_id}.csv"
    traj.to_csv(traj_path, index=False)
    print(f"  wrote {traj_path}  ({len(traj):,} rows, "
          f"duration={traj['time_rel_s'].iloc[-1]-traj['time_rel_s'].iloc[0]:.1f}s)")

    # ground_truth.csv — measured RSS + SNR for analysis stage
    gt_cols = {"time_rel_s": df["time_rel_s"].astype(float)}
    if "power_dBm" in df.columns:
        gt_cols["power_dBm"] = df["power_dBm"].astype(float)
    if "snr_dB" in df.columns:
        gt_cols["snr_dB"] = df["snr_dB"].astype(float)
    gt_cols["distance_m"] = df[dist_col].astype(float)
    gt_cols["altitude_m"] = df["altitude_m"].astype(float)
    gt = pd.DataFrame(gt_cols)
    gt_path = OUT / f"ground_truth_{flight_id}.csv"
    gt.to_csv(gt_path, index=False)
    print(f"  wrote {gt_path}")

    print(f"\nDone. Next: python3 layer3_run.py --flight_id {flight_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="list available flight_ids in each source")
    ap.add_argument("--source", default="aadm_testbed",
                    choices=list(SOURCES.keys()))
    ap.add_argument("--flight_id", default=None)
    ap.add_argument("--alt_min", type=float, default=5.0,
                    help="drop samples below this AGL (m) to skip ground; "
                         "set 0 to keep all")
    args = ap.parse_args()

    if args.list:
        _list_flights(); return
    if not args.flight_id:
        ap.error("--flight_id required (or use --list)")
    _prepare_one(args.source, args.flight_id, args.alt_min)


if __name__ == "__main__":
    main()
