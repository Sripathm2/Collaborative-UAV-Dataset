#!/usr/bin/env python3
"""
fig_attack_distributions.py   [Fig 8 — single-column figure, 1x2]

2-panel violin plot of flow features per attack class.
KEEPS ONLY (a) mean IAT and (b) packet rate. Drops flow duration and
mean packet size per spec. Layout: figsize=(3.33, 2.0).
Panel identifiers as small corner text, NOT titles.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

CLASSES = ['Benign', 'DoS', 'DDoS', 'Blackhole', 'Wormhole', 'Replay']

# Only two panels now (per spec): mean IAT + packet rate.
# Both use log scale.
PANELS = [
    ('Flow IAT Mean',  'mean IAT (s)',      True,  '(a) mean IAT'),
    ('Flow Packets/s', 'packet rate (pps)', True,  '(b) packet rate'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv',  required=True, type=Path)
    ap.add_argument('--out-pdf',
                    default=Path('../results/Step_7/fig_attack_distributions.pdf'),
                    type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv,
                     usecols=['Label'] + [p[0] for p in PANELS])
    df = df[df['Label'].isin(CLASSES)].copy()
    print(f"loaded {len(df):,} flows")
    print(df['Label'].value_counts())

    fig, axarr = plt.subplots(1, 2, figsize=(COL_W, 2.0),
                               constrained_layout=True)
    for i, (col, ylab, use_log, label) in enumerate(PANELS):
        ax = axarr[i]
        data = []
        for c in CLASSES:
            v = df.loc[df['Label'] == c, col].values
            if use_log:
                v = v[v > 0]
                v = np.log10(v) if v.size else np.array([0.0])
            data.append(v)

        parts = ax.violinplot(data, positions=range(len(CLASSES)),
                               showmeans=True, showmedians=True,
                               widths=0.75)
        for body in parts['bodies']:
            body.set_alpha(0.55)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=6)
        ylab_full = f"$\\log_{{10}}$({ylab})" if use_log else ylab
        ax.set_ylabel(ylab_full, fontsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.text(0.04, 0.98, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
