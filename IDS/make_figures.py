#!/usr/bin/env python3
"""make_figures.py

Renders:
  - fig_confusion.pdf      [Fig 10 — full-width figure*, 1x2]
      Two confusion matrices side by side: best model (highest macro-F1)
      and worst model (lowest macro-F1). NO cell annotations — color is
      the encoding. One shared colorbar (0-1, "row-normalized").
      figsize=(7.0, 3.6). No subplot titles; each panel identified by an
      axis label ("<best> (best)" / "<worst> (worst)").
      Pass --rank to print macro-F1 of every model and exit.
      Pass --best / --worst to override auto-pick.
  - fig_baseline_bars.pdf  [single-column bar chart]
      Cross-dataset T11 F1 grouped bars. Single column, no title.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =========================================================================
# ACM rc + column constants
# =========================================================================
COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

# =========================================================================
# CONFIG
# =========================================================================
DATASETS_DIR = Path("../Datasets")
RESULTS_DIR  = Path("../results/ids")
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILES = {
    "UAV-CAS":    {"ts":   DATASETS_DIR / "UAV-CAS_ts.csv",
                    "stat": DATASETS_DIR / "UAV-CAS_stat.csv"},
    "UNSW-NB15":  {"ts":   DATASETS_DIR / "UNSW-NB15_ts.csv",
                    "stat": DATASETS_DIR / "UNSW-NB15_stat.csv"},
    "CICIOT2023": {"ts":   DATASETS_DIR / "CICIOT2023_ts.csv",
                    "stat": DATASETS_DIR / "CICIOT2023_stat.csv"},
    "UAV-NIDD":   {"ts":   DATASETS_DIR / "UAVNIDD_ts.csv"},
    "CICIDS2017": {"stat": DATASETS_DIR / "CICIDS2017_stat.csv"},
}

MODELS = ["1D-CNN", "LSTM", "RF", "SGD", "LR",
          "MLP", "LightGBM", "ConvNet", "TinyML", "CNN-BiLSTM"]
CANONICAL_SCALER = "standard"


def _has_input(name, kind):
    return name in DATASET_FILES and kind in DATASET_FILES[name]


# =========================================================================
# Macro-F1 from a raw confusion matrix
# =========================================================================
def _macro_f1_from_cm(cm: np.ndarray) -> float:
    """Macro-F1 averaged over classes that actually appear (row_sum > 0)."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.shape[0]
    f1s = []
    for i in range(n):
        tp = cm[i, i]
        row_sum = cm[i, :].sum()      # true class i (recall denom)
        col_sum = cm[:, i].sum()      # predicted class i (precision denom)
        if row_sum == 0:
            continue                  # class absent from test set: skip
        precision = tp / col_sum if col_sum > 0 else 0.0
        recall    = tp / row_sum
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _rank_models() -> list[tuple[str, float, int]]:
    """Return [(model, macro_f1, cm_sum), ...] sorted high -> low."""
    cdir = RESULTS_DIR / "confusion"
    rows = []
    for m in MODELS:
        p = cdir / f"cm_{m}.npy"
        if not p.exists():
            print(f"  {m:>11s}  MISSING")
            continue
        cm = np.load(p)
        total = int(cm.sum())
        if total == 0:
            print(f"  {m:>11s}  EMPTY (sum=0)")
            continue
        f1 = _macro_f1_from_cm(cm)
        rows.append((m, f1, total))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def print_ranking():
    """Pretty-print the full ranking. Call with --rank from CLI."""
    rows = _rank_models()
    print(f"\n{'model':<12s}  {'macro_F1':>9s}  {'cm_sum':>10s}")
    print("-" * 36)
    for m, f1, total in rows:
        print(f"{m:<12s}  {f1:>9.4f}  {total:>10d}")
    if rows:
        print(f"\nauto-pick:  best={rows[0][0]}  worst={rows[-1][0]}")


# =========================================================================
# Fig 10 — confusion (best + worst), NO cell text
# =========================================================================
def fig_confusion(best: str | None = None, worst: str | None = None):
    cdir = RESULTS_DIR / "confusion"
    classes_path = cdir / "classes.csv"
    if not classes_path.exists():
        print("classes.csv missing -- run run_confusion.py first")
        return
    classes = pd.read_csv(classes_path, header=None).iloc[:, 0].tolist()
    n = len(classes)

    # Always print the ranking so the auto-pick (or override) is visible
    print("Macro-F1 ranking of available confusion matrices:")
    ranking = _rank_models()
    for m, f1, total in ranking:
        print(f"  {m:<12s}  macro_F1={f1:.4f}  cm_sum={total}")
    if not ranking:
        print("no confusion matrices found")
        return

    if best is None:
        best = ranking[0][0]
    if worst is None:
        worst = ranking[-1][0]
    print(f"\nrendering best={best}  worst={worst}")

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.6),
                              constrained_layout=True)
    im = None
    panels = [(axes[0], best,  f"{best} (best)"),
              (axes[1], worst, f"{worst} (worst)")]

    for ax, model, panel_label in panels:
        p = cdir / f"cm_{model}.npy"
        if not p.exists():
            ax.text(0.5, 0.5, f"{model} (missing)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        cm = np.load(p)
        total = int(cm.sum())
        # Sanity print — confirms the matrix is non-empty
        print(f"  {model}: cm.sum()={total}  shape={cm.shape}")
        assert total > 0, f"confusion matrix for {model} sums to zero"

        rs = cm.sum(axis=1, keepdims=True)
        cmn = np.where(rs > 0, cm / np.maximum(rs, 1), 0)

        im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        im.set_rasterized(True)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        # Per spec: axis label identifies the panel, NOT a title
        ax.set_xlabel(f"predicted\n{panel_label}", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("true", fontsize=7)

        # NO cell annotations — color is the encoding

    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.7,
                     label="row-normalized", pad=0.02)

    out = FIGURES_DIR / "fig_confusion.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"wrote {out}")


# =========================================================================
# Bar chart — T11 F1 across datasets (single column, untouched)
# =========================================================================
def fig_baseline_bars():
    p = RESULTS_DIR / "t11" / f"t11_{CANONICAL_SCALER}.csv"
    if not p.exists():
        print(f"{p} missing -- run run_t11.py first")
        return
    df = pd.read_csv(p)
    df = df[df["input_type"] == "ts"]
    datasets = [d for d in DATASET_FILES.keys() if _has_input(d, "ts")]

    mat = np.full((len(MODELS), len(datasets)), np.nan)
    for i, m in enumerate(MODELS):
        for j, d in enumerate(datasets):
            r = df[(df["model"] == m) & (df["dataset"] == d)]
            if len(r):
                mat[i, j] = float(r["f1_weighted"].values[0])

    fig, ax = plt.subplots(figsize=(COL_W, 2.4),
                            constrained_layout=True)
    bar_w = 0.18
    x = np.arange(len(MODELS))
    for j, d in enumerate(datasets):
        ax.bar(x + j * bar_w, mat[:, j] * 100, width=bar_w, label=d)
    ax.set_xticks(x + bar_w * (len(datasets) - 1) / 2)
    ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"F1 (\%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="lower right", ncol=len(datasets), fontsize=6,
              frameon=False, columnspacing=0.6, handletextpad=0.3)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out = FIGURES_DIR / "fig_baseline_bars.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"))
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",  action="store_true",
                    help="Print macro-F1 ranking of every model and exit "
                         "(no figure rendered)")
    ap.add_argument("--best",  default=None,
                    help="Override best-model auto-pick (default: top of ranking)")
    ap.add_argument("--worst", default=None,
                    help="Override worst-model auto-pick (default: bottom of ranking)")
    ap.add_argument("--skip-bars", action="store_true",
                    help="Skip rendering fig_baseline_bars.pdf")
    args = ap.parse_args()

    if args.rank:
        print_ranking()
    else:
        fig_confusion(best=args.best, worst=args.worst)
        if not args.skip_bars:
            fig_baseline_bars()