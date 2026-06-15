#!/usr/bin/env python3
"""
fig_pathloss_curves.py   [Fig 1 — full-width figure*, 1x5]

5-panel grid (one per altitude) of path-loss vs distance.

Layout: figsize=(7.0, 2.2). All axis ticks AND tick labels removed; axis
labels and a single shared legend kept. Panels are distinguished by a
small corner text annotation, NOT a title.

Reads layer1_comparison.json + maeng_rsrp.csv.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- ACM column constants + global rc ----
COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

# 5 altitudes the layer-1 fit was calibrated at
ALTITUDES = [30, 50, 70, 90, 110]
ALT_TOLERANCE = 5.0

ALT_MIN_FIT = 30.0
ALT_MAX_FIT = 110.0


def _clamp(h):
    return float(max(min(h, ALT_MAX_FIT), ALT_MIN_FIT))


def predict_logdist(d, alt, params):
    h_c = _clamp(alt)
    n = np.polyval(params['n_coeffs'], h_c)
    pl0 = np.polyval(params['PL0_coeffs'], h_c)
    d_safe = np.maximum(d, 1.0)
    return pl0 + 10.0 * n * np.log10(d_safe)


def predict_3gpp(d, alt, params):
    fc_ghz = params['frequency_GHz']
    offset = params['offset']
    h_c = _clamp(alt)
    exponent = max(23.9 - 1.8 * np.log10(h_c), 20.0)
    fspl_const = 20.0 * np.log10(40.0 * np.pi * fc_ghz / 3.0)
    d_safe = np.maximum(d, 1.0)
    return exponent * np.log10(d_safe) + fspl_const + offset


def predict_two_ray(d, alt, params):
    gamma = params['gamma']
    offset = params['offset']
    d_safe = np.maximum(d, 1.0)
    return 40.0 * np.log10(d_safe) + gamma * d_safe + offset


def _panel_label(ax, text):
    ax.text(0.04, 0.96, text, transform=ax.transAxes,
            ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer1',  required=True, type=Path)
    ap.add_argument('--maeng',   required=True, type=Path)
    ap.add_argument('--out-pdf', default=Path('fig_pathloss_curves.pdf'), type=Path)
    args = ap.parse_args()

    layer1 = json.loads(args.layer1.read_text())
    p_log = layer1['models']['Log-Distance']['params']
    p_2r  = layer1['models']['Two-Ray']['params']
    p_3g  = layer1['models']['3GPP TR 36.777']['params']

    df = pd.read_csv(args.maeng,
                     usecols=['rsrp_dBm', 'distance_m', 'altitude_m_nominal'])
    df['pl_proxy_dB'] = -df['rsrp_dBm']
    print(f"loaded {len(df):,} Maeng samples")

    fig, axarr = plt.subplots(1, 5, figsize=(FULL_W, 2.2),
                               sharey=True, constrained_layout=True)
    d_grid = np.logspace(np.log10(5), np.log10(2000), 200)

    handles_for_legend = None
    for i, alt in enumerate(ALTITUDES):
        ax = axarr[i]
        sub = df[np.isclose(df['altitude_m_nominal'], alt, atol=1e-3)]
        if len(sub) > 0:
            sc = ax.scatter(sub['distance_m'], sub['pl_proxy_dB'],
                            s=1.5, alpha=0.25, color='steelblue',
                            label='Maeng')
            sc.set_rasterized(True)

        ld, = ax.plot(d_grid, predict_logdist(d_grid, alt, p_log),
                      color='C1', linewidth=1.3, label='Log-distance')
        tr, = ax.plot(d_grid, predict_two_ray(d_grid, alt, p_2r),
                      color='C2', linewidth=1.1, linestyle='--', label='Two-Ray')
        g3, = ax.plot(d_grid, predict_3gpp(d_grid, alt, p_3g),
                      color='C3', linewidth=1.1, linestyle=':', label='3GPP')

        ax.set_xscale('log')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('distance')
        if i == 0:
            ax.set_ylabel(r'$-$RSRP (dB)')
            handles_for_legend = [
                plt.Line2D([0], [0], color='steelblue', marker='o',
                           linestyle='', markersize=3, label='Maeng'),
                ld, tr, g3,
            ]
        _panel_label(ax, f"alt {alt} m")
        ax.grid(True, which='both', alpha=0.25, linewidth=0.5)

        if handles_for_legend is not None:
            fig.legend(handles=handles_for_legend,
                   labels=[h.get_label() for h in handles_for_legend],
                   loc='outside lower center', ncol=4, frameon=False)

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
