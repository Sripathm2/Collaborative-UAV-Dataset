#!/usr/bin/env python3
"""layer4_sim_packet_stats.py
Sim-only (Benign) packet-level summary: packet size + IAT distributions
from UAV-CAS_ts.csv. Companion table to the three-way RSS comparison,
since Maeng/AFAR are link-layer only (no packet-level ground truth).

Reads /home/shree/.../Datasets/UAV-CAS_ts.csv, filters Benign rows,
explodes packet_time and packet_size list columns, computes IAT per flow,
and emits a 5-stat (mean/std/p5/p50/p95) summary table.

Outputs (results/layer4/):
  tab_sim_packet_stats.tex     2 rows (size, IAT) x 5 cols
  metrics_sim_packets.csv      same data as CSV
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

UAVCAS_TS = Path("../Datasets/UAV-CAS_ts.csv")
OUT       = Path("../results/layer4")
OUT.mkdir(parents=True, exist_ok=True)


def _parse_list(cell):
    if isinstance(cell, list): return cell
    if not isinstance(cell, str): return []
    s = cell.strip()
    if not s or s in ("[]", "nan", "NaN"): return []
    try: return ast.literal_eval(s)
    except Exception: return []


def _five_stat(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                 "p5": float("nan"), "p50": float("nan"),
                 "p95": float("nan"), "n": 0}
    return {
        "mean":  float(np.mean(x)),
        "std":   float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "p5":    float(np.percentile(x,  5)),
        "p50":   float(np.percentile(x, 50)),
        "p95":   float(np.percentile(x, 95)),
        "n":     int(len(x)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row_cap", type=int, default=0,
                    help="cap on Benign rows to process (0 = all)")
    args = ap.parse_args()

    if not UAVCAS_TS.exists():
        raise SystemExit(f"missing {UAVCAS_TS}")
    print(f"reading {UAVCAS_TS} ...")
    df = pd.read_csv(UAVCAS_TS, low_memory=False)
    n_raw = len(df)
    df = df[df["Label"].astype(str).str.strip() == "Benign"].reset_index(drop=True)
    print(f"  raw rows: {n_raw:,}  Benign: {len(df):,}")
    if args.row_cap and len(df) > args.row_cap:
        df = df.sample(n=args.row_cap, random_state=41).reset_index(drop=True)
        print(f"  capped to {len(df):,}")

    sizes_all = []
    iats_all  = []
    for _, row in df.iterrows():
        ts = _parse_list(row["packet_time"])
        sz = _parse_list(row["packet_size"])
        if sz:
            sizes_all.extend(sz)
        if len(ts) >= 2:
            iats_all.extend(np.diff(np.asarray(ts, dtype=np.float64)).tolist())

    sizes = np.asarray(sizes_all, dtype=np.float64)
    iats  = np.asarray(iats_all,  dtype=np.float64)
    print(f"  total packets : {len(sizes):,}")
    print(f"  total IAT obs : {len(iats):,}")

    stat_size = _five_stat(sizes)
    stat_iat  = _five_stat(iats)

    print("\npacket size (bytes):")
    print(f"  N={stat_size['n']:,}  mean={stat_size['mean']:.2f}  "
          f"std={stat_size['std']:.2f}  "
          f"p5={stat_size['p5']:.2f}  p50={stat_size['p50']:.2f}  "
          f"p95={stat_size['p95']:.2f}")
    print("IAT (s):")
    print(f"  N={stat_iat['n']:,}  mean={stat_iat['mean']:.6f}  "
          f"std={stat_iat['std']:.6f}  "
          f"p5={stat_iat['p5']:.6f}  p50={stat_iat['p50']:.6f}  "
          f"p95={stat_iat['p95']:.6f}")

    # ---- LaTeX table ----
    lines = [
        "% Feature & Mean & Std & p5 & p50 & p95",
        f"Packet size (B) & {stat_size['mean']:.2f} & {stat_size['std']:.2f} & "
        f"{stat_size['p5']:.2f} & {stat_size['p50']:.2f} & {stat_size['p95']:.2f} \\\\",
        f"IAT (s)         & {stat_iat['mean']:.6f} & {stat_iat['std']:.6f} & "
        f"{stat_iat['p5']:.6f} & {stat_iat['p50']:.6f} & {stat_iat['p95']:.6f} \\\\",
    ]
    p = OUT / "tab_sim_packet_stats.tex"
    p.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {p}")

    # ---- CSV ----
    pd.DataFrame([
        {"feature": "packet_size_B", **stat_size},
        {"feature": "iat_s",         **stat_iat},
    ]).to_csv(OUT / "metrics_sim_packets.csv", index=False)
    print(f"wrote {OUT / 'metrics_sim_packets.csv'}")


if __name__ == "__main__":
    main()
