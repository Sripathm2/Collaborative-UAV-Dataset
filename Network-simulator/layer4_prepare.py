#!/usr/bin/env python3
"""layer4_prepare.py
Extract RSS samples (dBm) from the three sources for three-way fidelity:
  Maeng (rsrp_dBm)             from maeng_rsrp.csv
  AFAR  (power_dBm)            from afar_all_flights.csv  (the DT)
  Sim   (rss_dBm)              from results/layer3/sim_<flight_id>.csv

Outputs (results/layer4/):
  rss_maeng.csv     single column rss_dBm
  rss_afar.csv      single column rss_dBm
  rss_sim.csv       single column rss_dBm (concat of given flight_ids)

Minimal filter: altitude_m >= alt_min (m AGL) where altitude column exists.
Default alt_min = 5.0 to skip ground points.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

ROOT  = Path("../Datasets/Finetuning-processed")
L3    = Path("../results/layer3")
OUT   = Path("../results/layer4")
OUT.mkdir(parents=True, exist_ok=True)

MAENG_PATH = ROOT / "maeng_rsrp.csv"
AFAR_PATH  = ROOT / "afar_all_flights.csv"

# Physical bounds for received power (dBm). Real UAV ground-link RSS lives
# roughly in [-130, +10] dBm; we widen to [-150, +30] for safety and discard
# anything outside (sensor glitches, parser errors, corrupted rows).
PHYSICAL_RSS_MIN = -150.0
PHYSICAL_RSS_MAX = 30.0


def _drop_unphysical(s: pd.Series, label: str) -> pd.Series:
    n_in = len(s)
    s = s[(s >= PHYSICAL_RSS_MIN) & (s <= PHYSICAL_RSS_MAX)]
    n_out = len(s)
    if n_out < n_in:
        print(f"  {label}: dropped {n_in - n_out:,} rows outside "
              f"[{PHYSICAL_RSS_MIN}, {PHYSICAL_RSS_MAX}] dBm "
              f"(kept {n_out:,} of {n_in:,})")
    return s.reset_index(drop=True)


def _extract_maeng(alt_min: float) -> pd.Series:
    print(f"reading {MAENG_PATH} ...")
    df = pd.read_csv(MAENG_PATH, usecols=["rsrp_dBm", "altitude_m"])
    n_raw = len(df)
    df = df[df["altitude_m"] >= alt_min]
    df = df[df["rsrp_dBm"].notna()]
    print(f"  maeng: {n_raw:,} -> {len(df):,} after altitude>={alt_min}m")
    s = df["rsrp_dBm"].astype(float).reset_index(drop=True)
    return _drop_unphysical(s, "maeng")


def _extract_afar(alt_min: float, flight_ids: list[str] | None) -> pd.Series:
    print(f"reading {AFAR_PATH} ...")
    df = pd.read_csv(AFAR_PATH, usecols=["power_dBm", "altitude_m", "flight_id"])
    n_raw = len(df)
    if flight_ids:
        df = df[df["flight_id"].isin(flight_ids)]
        print(f"  afar: filter to {len(flight_ids)} flights -> {len(df):,}")
    df = df[df["altitude_m"] >= alt_min]
    df = df[df["power_dBm"].notna()]
    print(f"  afar: {n_raw:,} -> {len(df):,} after filters")
    s = df["power_dBm"].astype(float).reset_index(drop=True)
    return _drop_unphysical(s, "afar")


def _extract_sim(flight_ids: list[str], variant_suffix: str = "") -> pd.Series:
    if not flight_ids:
        raise SystemExit("sim source needs --sim_flight_ids fid1 fid2 ...")
    chunks = []
    for fid in flight_ids:
        p = L3 / f"sim_{fid}{variant_suffix}.csv"
        if not p.exists():
            raise SystemExit(
                f"missing {p}; run layer3 for flight_id={fid} "
                f"(variant '{variant_suffix or 'default'}') first"
            )
        df = pd.read_csv(p, usecols=["rss_dBm"])
        df = df[df["rss_dBm"].notna()]
        chunks.append(df["rss_dBm"].astype(float))
        print(f"  sim {fid}{variant_suffix}: {len(df):,} rows")
    out = pd.concat(chunks, ignore_index=True)
    print(f"  sim total: {len(out):,}")
    return _drop_unphysical(out, "sim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt_min", type=float, default=5.0)
    ap.add_argument("--afar_flight_ids", nargs="*", default=None,
                    help="optional subset of afar flight_ids to keep "
                         "(default: all of afar_all_flights.csv)")
    ap.add_argument("--sim_flight_ids", nargs="+", required=True,
                    help="flight_ids whose sim_<fid>{suffix}.csv to concatenate "
                         "(must have been run via run_layer3.sh first)")
    ap.add_argument("--variant_suffix", default="",
                    help="layer3 variant suffix, e.g. '__logdist_noshadow'. "
                         "Empty means legacy sim_<fid>.csv (no suffix).")
    args = ap.parse_args()

    # --- Maeng ---
    s = _extract_maeng(args.alt_min)
    out_p = OUT / "rss_maeng.csv"
    s.to_frame("rss_dBm").to_csv(out_p, index=False)
    print(f"  wrote {out_p}")

    # --- AFAR ---
    s = _extract_afar(args.alt_min, args.afar_flight_ids)
    out_p = OUT / "rss_afar.csv"
    s.to_frame("rss_dBm").to_csv(out_p, index=False)
    print(f"  wrote {out_p}")

    # --- Sim ---
    s = _extract_sim(args.sim_flight_ids, args.variant_suffix)
    out_p = OUT / f"rss_sim{args.variant_suffix}.csv"
    s.to_frame("rss_dBm").to_csv(out_p, index=False)
    print(f"  wrote {out_p}")

    print("\nDone. Next: python3 layer4_run.py")


if __name__ == "__main__":
    main()