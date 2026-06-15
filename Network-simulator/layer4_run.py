#!/usr/bin/env python3
"""layer4_run.py
Three-way fidelity analysis. Consumes rss_{maeng,afar,sim}.csv from
layer4_prepare.py and emits:

  metrics_threeway.csv         pairwise Hellinger + per-source N
  tab_threeway_stats.tex       3 rows (sources) x 5 cols (mean/std/p5/p50/p95)
  tab_threeway_hellinger.tex   3 rows (pairs)   x 1 col  (Hellinger)
  figures/fig_threeway.pdf     2-panel: (a) overlaid PDFs (b) Hellinger bars

Hellinger distance on RSS (dBm) using histogram densities with fixed bins
[rss_min, rss_max] / n_bins.  H(P,Q) = (1/sqrt(2)) * sqrt(sum((sqrt(p)-sqrt(q))^2))
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path("/home/shree/Repos/Collaborative-UAV-Dataset/results/layer4")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- ACM single-column = 3.33in. Authored at final size, no downscaling. ----
COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

SOURCES = ["maeng", "afar", "sim"]
PRETTY  = {"maeng": "Maeng (measured)",
           "afar":  "AFAR DT",
           "sim":   "Sim"}
PAIRS = [("maeng", "sim"),
         ("maeng", "afar"),
         ("afar",  "sim")]


# =========================================================================
# IO
# =========================================================================
def _load(suffix: str = "") -> dict:
    """Load rss_<source>{suffix?}.csv. Only the 'sim' source uses the
    variant suffix; Maeng and AFAR are variant-independent."""
    d = {}
    for s in SOURCES:
        if s == "sim":
            p = OUT / f"rss_{s}{suffix}.csv"
        else:
            p = OUT / f"rss_{s}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}; run layer4_prepare.py first")
        d[s] = pd.read_csv(p)["rss_dBm"].astype(float).values
        print(f"  loaded {s:6s}{'' if s != 'sim' else suffix} : N={len(d[s]):,}  "
              f"range=[{d[s].min():.1f}, {d[s].max():.1f}] dBm")
    return d


# =========================================================================
# METRICS
# =========================================================================
def _hist_density(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, density=False)
    s = h.sum()
    if s == 0:
        return h.astype(float)
    return h.astype(float) / float(s)


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    sp = np.sqrt(p); sq = np.sqrt(q)
    return float(np.sqrt(0.5 * np.sum((sp - sq) ** 2)))


def _five_stat(x: np.ndarray) -> dict:
    return {
        "mean":  float(np.mean(x)),
        "std":   float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "p5":    float(np.percentile(x,  5)),
        "p50":   float(np.percentile(x, 50)),
        "p95":   float(np.percentile(x, 95)),
        "n":     int(len(x)),
    }


# =========================================================================
# TABLES
# =========================================================================
def _emit_stat_table(stats: dict, suffix: str) -> Path:
    rows = []
    for s in SOURCES:
        v = stats[s]
        rows.append(
            f"{PRETTY[s]} & "
            f"{v['mean']:.2f} & {v['std']:.2f} & "
            f"{v['p5']:.2f} & {v['p50']:.2f} & {v['p95']:.2f} \\\\"
        )
    out = ("% Source & Mean & Std & p5 & p50 & p95   (all dBm)\n"
           + "\n".join(rows) + "\n")
    p = OUT / f"tab_threeway_stats{suffix}.tex"
    p.write_text(out)
    print(f"  wrote {p}")
    return p


def _emit_hellinger_table(hell: dict, suffix: str) -> Path:
    rows = []
    for a, b in PAIRS:
        h = hell[(a, b)]
        rows.append(f"{PRETTY[a]} $\\leftrightarrow$ {PRETTY[b]} & {h:.4f} \\\\")
    out = ("% Pair & Hellinger distance (RSS, dBm)\n"
           + "\n".join(rows) + "\n")
    p = OUT / f"tab_threeway_hellinger{suffix}.tex"
    p.write_text(out)
    print(f"  wrote {p}")
    return p


# =========================================================================
# FIGURE
# =========================================================================
def _figure(rss: dict, stats: dict, hell: dict, bins: np.ndarray,
            suffix: str):
    """Single-column 1x2 figure: (a) overlaid PDFs, (b) Hellinger bars.
    No titles; panel identifiers as small corner text."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9),
                              constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1]})

    # --- panel (a): overlaid PDFs ---
    ax = axes[0]
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = {"maeng": "C0", "afar": "C1", "sim": "C2"}
    handles = []
    for s in SOURCES:
        p = _hist_density(rss[s], bins)
        ln, = ax.plot(centers, p, color=colors[s], linewidth=1.0,
                       label=PRETTY[s])
        ax.fill_between(centers, 0, p, color=colors[s], alpha=0.12)
        handles.append(ln)
    ax.set_xlabel("RSS (dBm)")
    ax.set_ylabel("density")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(handles=handles, loc="best", fontsize=6, frameon=False)
    ax.text(0.04, 0.96, "(a) RSS PDFs", transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    # --- panel (b): Hellinger bars ---
    ax = axes[1]
    labels = [f"{PRETTY[a].split()[0]}\n$\\leftrightarrow$\n{PRETTY[b].split()[0]}"
              for a, b in PAIRS]
    vals = [hell[p] for p in PAIRS]
    bars = ax.bar(range(len(PAIRS)), vals,
                   color=["C3", "C4", "C5"], width=0.7)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Hellinger")
    ax.set_ylim(0, max(0.5, max(vals) * 1.2))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.text(0.04, 0.96, "(b) pairwise H", transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    out = FIG / f"fig_threeway{suffix}.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  wrote {out}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rss_min",  type=float, default=-130.0)
    ap.add_argument("--rss_max",  type=float, default=-20.0)
    ap.add_argument("--n_bins",   type=int,   default=50)
    ap.add_argument("--out_suffix", default="",
                    help="suffix on rss_sim<suffix>.csv input and all "
                         "output files (matches layer3 variant suffix)")
    args = ap.parse_args()
    suffix = args.out_suffix

    print(f"=== Layer-4 three-way fidelity  (variant='{suffix}') ===")
    rss = _load(suffix)

    # 5-stat per source
    stats = {s: _five_stat(rss[s]) for s in SOURCES}

    # pairwise Hellinger on histogram densities
    bins = np.linspace(args.rss_min, args.rss_max, args.n_bins + 1)
    dens = {s: _hist_density(rss[s], bins) for s in SOURCES}
    hell = {(a, b): _hellinger(dens[a], dens[b]) for a, b in PAIRS}

    # ---- emit metrics CSV ----
    rows = []
    for s in SOURCES:
        v = stats[s]
        rows.append({"source": s, **v})
    for (a, b), h in hell.items():
        rows.append({"source": f"{a}_vs_{b}",
                      "mean": np.nan, "std": np.nan,
                      "p5": np.nan, "p50": np.nan, "p95": np.nan,
                      "n": np.nan, "hellinger": h})
    df = pd.DataFrame(rows)
    p = OUT / f"metrics_threeway{suffix}.csv"
    df.to_csv(p, index=False)
    print(f"  wrote {p}")

    print("\n5-stat (RSS dBm):")
    for s in SOURCES:
        v = stats[s]
        print(f"  {s:6s}  N={v['n']:>8,}  mean={v['mean']:>7.2f}  "
              f"std={v['std']:>6.2f}  "
              f"p5={v['p5']:>7.2f}  p50={v['p50']:>7.2f}  p95={v['p95']:>7.2f}")
    print("\nHellinger:")
    for (a, b), h in hell.items():
        print(f"  {a} <-> {b}  H = {h:.4f}")

    _emit_stat_table(stats, suffix)
    _emit_hellinger_table(hell, suffix)
    _figure(rss, stats, hell, bins, suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
