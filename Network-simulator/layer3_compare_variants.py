#!/usr/bin/env python3
"""layer3_compare_variants.py
Aggregate metrics_summary{suffix}.csv files across variants and emit a
single-row-per-variant comparison so you can pick the winning config.

Usage:
    python3 layer3_compare_variants.py \
        --suffixes __logdist_shadow __logdist_noshadow __3gpp_shadow __3gpp_noshadow

Outputs:
    results/layer3/metrics_variant_compare.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("../results/layer3")


def _agg(suffix: str) -> dict:
    p = OUT_DIR / f"metrics_summary{suffix}.csv"
    if not p.exists():
        return {"variant": suffix.lstrip("_"), "n_flights": 0,
                 "median_of_median_abs_rss_err_dB":             np.nan,
                 "median_of_median_abs_rss_err_corrected_dB":   np.nan,
                 "median_of_mean_rss_bias_dB":                  np.nan,
                 "median_of_snr_pearson_r":                     np.nan,
                 "note": f"missing {p.name}"}
    df = pd.read_csv(p)
    return {
        "variant":   suffix.lstrip("_"),
        "n_flights": int(df["n"].notna().sum()),
        "median_of_median_abs_rss_err_dB":
            float(df["median_abs_rss_err_dB"].median()),
        "median_of_median_abs_rss_err_corrected_dB":
            float(df["median_abs_rss_err_corrected_dB"].median()),
        "median_of_mean_rss_bias_dB":
            float(df["mean_rss_bias_dB"].median()),
        "median_of_snr_pearson_r":
            float(df["snr_pearson_r"].median()),
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffixes", nargs="+", required=True)
    args = ap.parse_args()

    rows = [_agg(s) for s in args.suffixes]
    df = pd.DataFrame(rows)
    out = OUT_DIR / "metrics_variant_compare.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()