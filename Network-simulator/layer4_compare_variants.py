#!/usr/bin/env python3
"""layer4_compare_variants.py
Aggregate metrics_threeway<suffix>.csv across Layer-3 variants used as sim
sources. One row per variant; columns = the three Hellinger pair values and
sim 5-stat snapshot. Pick the variant with the lowest H(Maeng,Sim) +
H(AFAR,Sim) as the most realistic config.

Usage:
    python3 layer4_compare_variants.py \
        --suffixes __logdist_shadow __logdist_noshadow __3gpp_shadow __3gpp_noshadow
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("../results/layer4")

# Pair labels must match those produced by layer4_run.py (PAIRS order):
#   maeng_vs_sim, maeng_vs_afar, afar_vs_sim
PAIR_KEYS = ["maeng_vs_sim", "maeng_vs_afar", "afar_vs_sim"]


def _agg(suffix: str) -> dict:
    p = OUT / f"metrics_threeway{suffix}.csv"
    base = {"variant": suffix.lstrip("_") or "<none>"}
    if not p.exists():
        base.update({k: np.nan for k in PAIR_KEYS})
        base.update({"sim_mean_dBm": np.nan, "sim_p50_dBm": np.nan,
                     "sim_n": np.nan, "note": f"missing {p.name}"})
        return base

    df = pd.read_csv(p)
    # pair rows have 'source' = "<a>_vs_<b>" with 'hellinger' populated
    h = {}
    for k in PAIR_KEYS:
        r = df[df["source"] == k]
        h[k] = float(r["hellinger"].values[0]) if len(r) else np.nan

    # sim 5-stat row
    sim_r = df[df["source"] == "sim"]
    sim_mean = float(sim_r["mean"].values[0]) if len(sim_r) else np.nan
    sim_p50  = float(sim_r["p50"].values[0])  if len(sim_r) else np.nan
    sim_n    = int(sim_r["n"].values[0])      if len(sim_r) else 0

    base.update(h)
    base.update({"sim_mean_dBm": sim_mean,
                 "sim_p50_dBm":  sim_p50,
                 "sim_n":        sim_n,
                 "note":         ""})
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffixes", nargs="+", required=True)
    args = ap.parse_args()

    rows = [_agg(s) for s in args.suffixes]
    df = pd.DataFrame(rows)

    # rank variants by sim-vs-(Maeng+AFAR) Hellinger
    df["sim_vs_real_total_H"] = (df["maeng_vs_sim"].fillna(np.inf)
                                  + df["afar_vs_sim"].fillna(np.inf))
    df = df.sort_values("sim_vs_real_total_H").reset_index(drop=True)

    out = OUT / "metrics_variant_compare.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()