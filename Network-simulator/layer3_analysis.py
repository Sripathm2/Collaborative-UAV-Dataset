#!/usr/bin/env python3
"""layer3_analysis.py
Join sim_<fid>{suffix}.csv (from layer3_run.py) with ground_truth_<fid>.csv
(from layer3_prepare.py) and compute §6.4 link-level metrics.

Per flight:
  median_abs_rss_err_dB             raw median |sim_rss - measured|
  mean_rss_bias_dB                  signed mean error = sensor calibration bias
  median_abs_rss_err_corrected_dB   median |err - mean_bias|  (cross-cal apples-to-apples)
  snr_pearson_r / _p                shape correlation
  n                                 joined samples

Per altitude bin (pooled across joined flights):
  alt_bin, n, median_abs_rss_err_dB, median_abs_rss_err_corrected_dB,
  mean_rss_bias_dB, snr_pearson_r

Outputs (suffix applied to every file so sweep variants don't collide):
  metrics_<fid>{suffix}.csv
  metrics_summary{suffix}.csv
  metrics_per_altitude_summary{suffix}.csv
  figures/fig_link_quality_<fid>{suffix}.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

OUT_DIR = Path("/home/shree/Repos/Collaborative-UAV-Dataset/results/layer3")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- ACM single-column = 3.33in. Authored at final size, no downscaling. ----
COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

# Altitude bins (m AGL). Edges aligned with Maeng calibration anchors
# (30, 50, 70, 90, 110 m) so per-bin numbers map onto §6.1 Layer-1 fit.
ALT_EDGES  = [0.0, 40.0, 60.0, 80.0, 100.0, np.inf]
ALT_LABELS = ["<40m", "40-60m", "60-80m", "80-100m", ">=100m"]


# =========================================================================
# JOIN
# =========================================================================
def _join(flight_id: str, suffix: str) -> pd.DataFrame:
    sim_p = OUT_DIR / f"sim_{flight_id}{suffix}.csv"
    gt_p  = OUT_DIR / f"ground_truth_{flight_id}.csv"
    if not sim_p.exists() or not gt_p.exists():
        raise SystemExit(f"missing inputs for {flight_id}{suffix}: "
                          f"sim={sim_p.exists()} gt={gt_p.exists()}")
    sim = pd.read_csv(sim_p).sort_values("time_rel_s").reset_index(drop=True)
    gt  = pd.read_csv(gt_p ).sort_values("time_rel_s").reset_index(drop=True)

    # explicit suffix on the column that conflicts BEFORE merge
    sim = sim.rename(columns={"snr_dB": "snr_dB_sim"})
    gt  = gt.rename(columns={"snr_dB":  "snr_dB_gt"})

    # Sim was walked at GT time points; merge_asof with 1us tolerance is
    # robust to any float CSV-roundtrip drift.
    merged = pd.merge_asof(sim, gt, on="time_rel_s",
                            direction="nearest", tolerance=1e-6)
    if "power_dBm" in merged.columns:
        merged = merged[merged["power_dBm"].notna()]
    if "snr_dB_gt" in merged.columns:
        merged = merged[merged["snr_dB_gt"].notna()]
    return merged.reset_index(drop=True)


# =========================================================================
# METRICS
# =========================================================================
def _metrics_from_df(m: pd.DataFrame) -> dict:
    """Compute the 5 metrics from an already-joined DataFrame."""
    n = len(m)
    if n == 0:
        return {"n": 0,
                "median_abs_rss_err_dB":            np.nan,
                "median_abs_rss_err_corrected_dB":  np.nan,
                "mean_rss_bias_dB":                 np.nan,
                "snr_pearson_r":                    np.nan,
                "snr_pearson_p":                    np.nan}

    err = m["rss_dBm"].values - m["power_dBm"].values
    bias = float(np.mean(err))
    out = {
        "n":                                int(n),
        "median_abs_rss_err_dB":            float(np.median(np.abs(err))),
        "median_abs_rss_err_corrected_dB":  float(np.median(np.abs(err - bias))),
        "mean_rss_bias_dB":                 bias,
    }

    have_both = ("snr_dB_gt" in m.columns and "snr_dB_sim" in m.columns)
    if (have_both and n >= 2
            and m["snr_dB_gt"].std()  > 0
            and m["snr_dB_sim"].std() > 0):
        r, p = pearsonr(m["snr_dB_sim"].values, m["snr_dB_gt"].values)
        out["snr_pearson_r"] = float(r)
        out["snr_pearson_p"] = float(p)
    else:
        out["snr_pearson_r"] = np.nan
        out["snr_pearson_p"] = np.nan
    return out


def _metrics_one(flight_id: str, suffix: str):
    """Returns (per-flight metrics dict, joined DataFrame for pooling)."""
    m = _join(flight_id, suffix)
    row = {"flight_id": flight_id, **_metrics_from_df(m)}
    return row, m


# =========================================================================
# PER-ALTITUDE AGGREGATION
# =========================================================================
def _per_altitude_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    """Bin pooled rows by altitude_m and compute metrics per bin."""
    if "altitude_m" not in pooled.columns or len(pooled) == 0:
        return pd.DataFrame(
            columns=["alt_bin"] + list(_metrics_from_df(pd.DataFrame()).keys())
        )
    pooled = pooled.copy()
    pooled["alt_bin"] = pd.cut(pooled["altitude_m"],
                                bins=ALT_EDGES, labels=ALT_LABELS,
                                right=False, include_lowest=True)
    rows = []
    for label in ALT_LABELS:
        sub = pooled[pooled["alt_bin"] == label]
        row = {"alt_bin": label, **_metrics_from_df(sub)}
        rows.append(row)
    rows.append({"alt_bin": "ALL", **_metrics_from_df(pooled)})
    return pd.DataFrame(rows)


# =========================================================================
# FIGURE
# =========================================================================
def _fig_link_quality(flight_id: str, suffix: str):
    """Single-column 3-panel stacked figure (RSS, SNR, throughput).
    Shared x-axis; x-label only on bottom panel. No titles; panel
    identifiers as small corner text."""
    m = _join(flight_id, suffix)
    if len(m) == 0:
        print(f"  skip fig for {flight_id}{suffix}: 0 joined rows"); return

    t = m["time_rel_s"].values
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.6),
                              sharex=True, constrained_layout=True)

    # panel 1: RSS
    ax = axes[0]
    if "power_dBm" in m.columns:
        ln_m, = ax.plot(t, m["power_dBm"].values, color="C0",
                         linewidth=0.5, alpha=0.7, label="measured")
        ln_m.set_rasterized(True)
    ln_s, = ax.plot(t, m["rss_dBm"].values, color="C1",
                     linewidth=0.5, alpha=0.7, label="sim")
    ln_s.set_rasterized(True)
    ax.set_ylabel("RSS (dBm)")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower left", fontsize=6, ncol=2, frameon=False)
    ax.text(0.02, 0.96, "(a) RSS", transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    # panel 2: SNR
    ax = axes[1]
    if "snr_dB_gt" in m.columns:
        ln, = ax.plot(t, m["snr_dB_gt"].values, color="C0",
                       linewidth=0.5, alpha=0.7)
        ln.set_rasterized(True)
    ln, = ax.plot(t, m["snr_dB_sim"].values, color="C1",
                   linewidth=0.5, alpha=0.7)
    ln.set_rasterized(True)
    ax.set_ylabel("SNR (dB)")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.text(0.02, 0.96, "(b) SNR", transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    # panel 3: sim throughput
    ax = axes[2]
    ln, = ax.plot(t, m["bw_mbit"].values, color="C2", linewidth=0.5)
    ln.set_rasterized(True)
    ax.set_ylabel("BW (Mb/s)")
    ax.set_xlabel("time (s)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.text(0.02, 0.96, "(c) throughput (sim)", transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    out = FIG_DIR / f"fig_link_quality_{flight_id}{suffix}.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  wrote {out}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight_ids", nargs="+", required=True)
    ap.add_argument("--out_suffix", default="",
                    help="suffix on sim_<fid>{suffix}.csv and all output files")
    ap.add_argument("--no_fig", action="store_true",
                    help="skip per-flight figure generation")
    args = ap.parse_args()

    suffix = args.out_suffix

    rows = []
    pooled_list = []
    for fid in args.flight_ids:
        print(f"\n=== {fid}{suffix} ===")
        try:
            row, m = _metrics_one(fid, suffix)
        except SystemExit as e:
            print(f"  {e}"); continue
        rows.append(row)
        if len(m) > 0:
            pooled_list.append(
                m[["altitude_m", "rss_dBm", "power_dBm",
                   "snr_dB_sim", "snr_dB_gt"]].copy()
            )
        print(f"  n                            = {row['n']:,}")
        print(f"  median |RSS err|       (dB)  = {row['median_abs_rss_err_dB']:.3f}")
        print(f"  median |RSS err| (cor) (dB)  = {row['median_abs_rss_err_corrected_dB']:.3f}")
        print(f"  mean RSS bias          (dB)  = {row['mean_rss_bias_dB']:+.3f}")
        print(f"  SNR Pearson r                = {row['snr_pearson_r']:.4f}")

        per_p = OUT_DIR / f"metrics_{fid}{suffix}.csv"
        pd.DataFrame([row]).to_csv(per_p, index=False)
        print(f"  wrote {per_p}")

        if not args.no_fig:
            _fig_link_quality(fid, suffix)

    # across-flight summary
    if rows:
        df = pd.DataFrame(rows)
        sum_path = OUT_DIR / f"metrics_summary{suffix}.csv"
        df.to_csv(sum_path, index=False)
        print(f"\nwrote summary -> {sum_path}")
        print(df.to_string(index=False))

    # per-altitude summary (pooled across flights)
    if pooled_list:
        pooled = pd.concat(pooled_list, ignore_index=True)
        alt_df = _per_altitude_summary(pooled)
        alt_path = OUT_DIR / f"metrics_per_altitude_summary{suffix}.csv"
        alt_df.to_csv(alt_path, index=False)
        print(f"\nwrote per-altitude summary -> {alt_path}")
        print(alt_df.to_string(index=False))


if __name__ == "__main__":
    main()
