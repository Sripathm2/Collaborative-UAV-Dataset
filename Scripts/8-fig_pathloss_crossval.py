#!/usr/bin/env python3
"""
fig_pathloss_crossval.py   [Fig 2 — full-width figure*, 1x3]

Cross-dataset validation of the Layer-1 path-loss models on the Gurses-
Sichitiu dataset. Layout: figsize=(7.0, 2.4). All axis ticks AND tick
labels removed; axis labels and identity line kept. Per-panel metrics
(RMSE, bias, r) placed as small corner text, NOT a title.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

ALT_MIN_FIT = 30.0
ALT_MAX_FIT = 110.0


def _clamp(h):
    return np.clip(h, ALT_MIN_FIT, ALT_MAX_FIT)


def predict_logdist(d, alt, params):
    h_c = _clamp(np.asarray(alt, dtype=np.float64))
    n   = np.polyval(params['n_coeffs'],   h_c)
    pl0 = np.polyval(params['PL0_coeffs'], h_c)
    d_safe = np.maximum(np.asarray(d, dtype=np.float64), 1.0)
    return pl0 + 10.0 * n * np.log10(d_safe)


def predict_3gpp(d, alt, params):
    fc_ghz = params['frequency_GHz']
    offset = params['offset']
    h_c = _clamp(np.asarray(alt, dtype=np.float64))
    exponent = np.maximum(23.9 - 1.8 * np.log10(h_c), 20.0)
    fspl_const = 20.0 * np.log10(40.0 * np.pi * fc_ghz / 3.0)
    d_safe = np.maximum(np.asarray(d, dtype=np.float64), 1.0)
    return exponent * np.log10(d_safe) + fspl_const + offset


def predict_two_ray(d, alt, params):
    gamma = params['gamma']
    offset = params['offset']
    d_safe = np.maximum(np.asarray(d, dtype=np.float64), 1.0)
    return 40.0 * np.log10(d_safe) + gamma * d_safe + offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer1',  required=True, type=Path)
    ap.add_argument('--gurses',  required=True, type=Path)
    ap.add_argument('--out-pdf', default=Path('fig_pathloss_crossval.pdf'), type=Path)
    ap.add_argument('--max-points', type=int, default=8000)
    args = ap.parse_args()

    layer1 = json.loads(args.layer1.read_text())
    p_log = layer1['models']['Log-Distance']['params']
    p_2r  = layer1['models']['Two-Ray']['params']
    p_3g  = layer1['models']['3GPP TR 36.777']['params']

    df = pd.read_csv(args.gurses,
                     usecols=['rx_power_dBm', 'distance_m',
                              'rx_altitude_m', 'is_flight'])
    df = df[df['is_flight'] == True].copy()
    df = df[(df['distance_m'] > 1.0) & (df['rx_altitude_m'] > 5.0)]
    df['pl_meas'] = -df['rx_power_dBm']
    print(f"Gurses flight rows after filter: {len(df):,}")

    if len(df) > args.max_points:
        df = df.sample(args.max_points, random_state=0)
        print(f"  subsampled to {len(df):,}")

    d_arr = df['distance_m'].values
    h_arr = df['rx_altitude_m'].values
    y_meas = df['pl_meas'].values

    models = [
        ('Log-Distance',   lambda d, h: predict_logdist(d, h, p_log)),
        ('Two-Ray',        lambda d, h: predict_two_ray(d, h, p_2r)),
        ('3GPP TR 36.777', lambda d, h: predict_3gpp(d, h, p_3g)),
    ]
    panel_labels = ['(a)', '(b)', '(c)']

    fig, axarr = plt.subplots(1, 3, figsize=(FULL_W, 2.4),
                               sharex=True, sharey=True,
                               constrained_layout=True)
    for i, (name, pred_fn) in enumerate(models):
        ax = axarr[i]
        y_pred = pred_fn(d_arr, h_arr)
        rmse = float(np.sqrt(np.mean((y_pred - y_meas) ** 2)))
        bias = float(np.mean(y_pred - y_meas))
        if np.std(y_pred) > 1e-9 and np.std(y_meas) > 1e-9:
            r = float(np.corrcoef(y_pred, y_meas)[0, 1])
        else:
            r = 0.0

        sc = ax.scatter(y_meas, y_pred, s=1.5, alpha=0.25,
                        color='steelblue')
        sc.set_rasterized(True)
        lo = min(y_meas.min(), y_pred.min()) - 2
        hi = max(y_meas.max(), y_pred.max()) + 2
        ax.plot([lo, hi], [lo, hi], color='red',
                linestyle='--', linewidth=1.0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(r'measured $-$RSRP (dB)')
        if i == 0:
            ax.set_ylabel(r'predicted $-$RSRP (dB)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.text(0.04, 0.96,
                f"{panel_labels[i]} {name}\n"
                f"RMSE={rmse:.2f} dB\n"
                f"bias={bias:+.2f}  r={r:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
