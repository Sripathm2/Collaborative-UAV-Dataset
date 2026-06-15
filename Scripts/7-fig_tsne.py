#!/usr/bin/env python3
"""
fig_tsne.py   [Fig 9 — single-column, figsize=(3.33, 3.0)]

t-SNE 2D embedding of UAV-CAS flows colored by attack class. Square
panel. Small marker size; legend in 2 columns. No title.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

CLASSES = ['Benign', 'DoS', 'DDoS', 'Blackhole', 'Wormhole', 'Replay']

META_COLS = {'config_idx', 'num_drones', 'num_bs', 'payload', 'pathloss',
             'modulation', 'mission', 'tx_power', 'noise', 'src_ip', 'dst_ip',
             'Label'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv',      required=True, type=Path)
    ap.add_argument('--out-pdf',
                    default=Path('../results/Step_7/fig_tsne.pdf'), type=Path)
    ap.add_argument('--n-per-class', type=int, default=1000)
    ap.add_argument('--perplexity',  type=float, default=30.0)
    ap.add_argument('--n-iter',      type=int, default=1000)
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df[df['Label'].isin(CLASSES)].copy()
    print(f"loaded {len(df):,} single-class flows")

    sampled = []
    for c in CLASSES:
        sub = df[df['Label'] == c]
        if len(sub) == 0:
            continue
        n = min(len(sub), args.n_per_class)
        sampled.append(sub.sample(n=n, random_state=args.seed))
    samp = pd.concat(sampled, ignore_index=True)
    print(f"after stratified sampling: {len(samp):,}")

    feature_cols = [c for c in samp.columns if c not in META_COLS]
    X = samp[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)

    tsne_kwargs = dict(n_components=2, random_state=args.seed,
                       perplexity=args.perplexity,
                       init='pca', learning_rate='auto')
    try:
        Z = TSNE(max_iter=args.n_iter, **tsne_kwargs).fit_transform(Xs)
    except TypeError:
        Z = TSNE(n_iter=args.n_iter, **tsne_kwargs).fit_transform(Xs)

    y = samp['Label'].values
    fig, ax = plt.subplots(figsize=(COL_W, 3.0), constrained_layout=True)
    palette = {
        'Benign':    '#1f77b4',
        'DoS':       '#ff7f0e',
        'DDoS':      '#2ca02c',
        'Blackhole': '#d62728',
        'Wormhole':  '#9467bd',
        'Replay':    '#17becf',
    }
    for c in CLASSES:
        m = y == c
        if not m.any():
            continue
        sc = ax.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.55,
                         color=palette[c], label=c)
        sc.set_rasterized(True)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(loc='upper right', ncol=2, frameon=True, framealpha=0.85,
              markerscale=2, handletextpad=0.3, columnspacing=0.6,
              borderpad=0.3)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
