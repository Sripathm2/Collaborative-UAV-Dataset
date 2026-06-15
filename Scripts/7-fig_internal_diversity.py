#!/usr/bin/env python3
"""
fig_internal_diversity.py   [Fig 7 — full-width figure*, 1x4]

4-panel comparison of flow-feature distributions induced by extreme values
of each configuration axis. Layout: figsize=(7.0, 2.1). One row.
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

AXES = [
    ('(a) 5 vs 20 drones',           'num_drones', '5',       '20'),
    ('(b) logdist vs 3GPP',          'pathloss',   'logdist', '3gpp'),
    ('(c) spiral vs random',         'mission',    'spiral',  'random'),
    ('(d) TX 10 vs 30 dBm',          'tx_power',   '10',      '30'),
]

FEATURE_COL = 'Flow IAT Mean'
FEATURE_LABEL = r'$\log_{10}$(mean IAT, s)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv',  required=True, type=Path)
    ap.add_argument('--out-pdf',
                    default=Path('../results/Step_7/fig_internal_diversity.pdf'),
                    type=Path)
    ap.add_argument('--class-filter', default='Benign')
    args = ap.parse_args()

    needed_cols = ['Label', FEATURE_COL] + [a[1] for a in AXES]
    df = pd.read_csv(args.in_csv, usecols=needed_cols,
                     dtype={a[1]: str for a in AXES})
    print(f"loaded {len(df):,} flows")

    if args.class_filter != 'all':
        df = df[df['Label'] == args.class_filter].copy()
        print(f"  {args.class_filter}-only: {len(df):,} flows")

    fig, axarr = plt.subplots(1, 4, figsize=(FULL_W, 2.1),
                               sharey=True, constrained_layout=True)
    for i, (label, axis_key, low_val, high_val) in enumerate(AXES):
        ax = axarr[i]
        lo = df.loc[df[axis_key] == low_val, FEATURE_COL].values
        hi = df.loc[df[axis_key] == high_val, FEATURE_COL].values
        lo = lo[lo > 0]; hi = hi[hi > 0]
        if lo.size == 0 or hi.size == 0:
            ax.text(0.5, 0.5,
                    f'no data\n({low_val}: {lo.size}, {high_val}: {hi.size})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            lo_log = np.log10(lo); hi_log = np.log10(hi)
            parts = ax.violinplot([lo_log, hi_log], positions=[1, 2],
                                   showmeans=True, showmedians=True,
                                   widths=0.7)
            for body in parts['bodies']:
                body.set_alpha(0.55)
            ax.set_xticks([1, 2])
            ax.set_xticklabels([low_val, high_val], fontsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(True, alpha=0.25, linewidth=0.5)
        if i == 0:
            ax.set_ylabel(FEATURE_LABEL)
        ax.text(0.04, 0.96, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
