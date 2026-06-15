#!/usr/bin/env python3
"""
fig_velocity_validation.py   [Fig 4 — full-width figure*, 1x4]

4-panel speed-CDF comparison sim vs measured (AFAR). Layout:
figsize=(7.0, 2.0). Shared y-label "CDF" (only on leftmost panel),
one shared legend, panel identifiers as small corner text, NOT titles.
"""

import argparse
import json
import random as _rnd
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import jensenshannon

COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

LAT0 = 35.727451
LON0 = -78.695974
DEG_TO_M = 1.113195e5

GRID = 1000.0
ALT_MIN = 30.0
ALT_MAX = 110.0

MISSIONS = ['spiral', 'grid', 'hover_transit', 'random']
PANEL_LABELS = ['(a) spiral', '(b) grid', '(c) hover_transit', '(d) random']

SIM_DURATION = {
    'spiral':        600.0,
    'grid':          120.0,
    'hover_transit': 240.0,
    'random':        180.0,
}
SIM_DT = 0.2


def afar_to_xyz(df):
    x = (df['longitude'] - LON0) * DEG_TO_M
    y = (df['latitude']  - LAT0) * DEG_TO_M
    z = df['altitude_m']
    return x.values, y.values, z.values


def speed_from_xyz(t, x, y, z):
    if len(t) < 2:
        return np.array([])
    dt = np.diff(t)
    dx = np.diff(x); dy = np.diff(y); dz = np.diff(z)
    valid = dt > 0
    v = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2 + dz[valid] ** 2) / dt[valid]
    return v


def cdf(arr):
    a = np.sort(arr)
    if len(a) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(a) + 1) / len(a)
    return a, y


def jsd_speed(sim, meas, n_bins=60):
    if len(sim) == 0 or len(meas) == 0:
        return float('nan')
    lo = min(sim.min(), meas.min())
    hi = max(sim.max(), meas.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(sim,  bins=bins, density=True)
    q, _ = np.histogram(meas, bins=bins, density=True)
    p = p + 1e-12; p = p / p.sum()
    q = q + 1e-12; q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simlib-path', required=True, type=Path)
    ap.add_argument('--layer2',      required=True, type=Path)
    ap.add_argument('--afar-csv',    required=True, type=Path)
    ap.add_argument('--out-pdf',     default=Path('fig_velocity_validation.pdf'), type=Path)
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(args.simlib_path))
    from simlib.mobility import MobilityGenerator

    layer2 = json.loads(args.layer2.read_text())
    flight_cat = layer2['flight_categories']

    df = pd.read_csv(args.afar_csv,
                     usecols=['timestamp', 'longitude', 'latitude',
                              'altitude_m', 'flight_id'])
    df = df.dropna(subset=['longitude', 'latitude', 'altitude_m']).copy()
    print(f"AFAR rows after NaN drop: {len(df):,}")

    meas_speeds = {m: [] for m in MISSIONS}
    for fid, sub in df.groupby('flight_id'):
        cat = flight_cat.get(fid)
        if cat not in MISSIONS:
            continue
        sub = sub.sort_values('timestamp')
        x, y, z = afar_to_xyz(sub)
        v = speed_from_xyz(sub['timestamp'].values, x, y, z)
        v = v[(v >= 0) & (v < 50)]
        if v.size > 0:
            meas_speeds[cat].append(v)
    meas_speeds = {m: np.concatenate(v) if v else np.array([])
                   for m, v in meas_speeds.items()}

    sim_speeds = {}
    for i, mission in enumerate(MISSIONS):
        mob = MobilityGenerator(str(args.layer2),
                                 rng=_rnd.Random(args.seed + i))
        anchor_rng = _rnd.Random(args.seed + 100 + i)
        anchor = (anchor_rng.uniform(50.0, GRID - 50.0),
                  anchor_rng.uniform(50.0, GRID - 50.0))
        uav_id = f'd_{mission}'
        mob.create_uav(uav_id=uav_id, mission_type=mission,
                       origin_xy=anchor, grid_size_m=GRID,
                       alt_min=ALT_MIN, alt_max=ALT_MAX)
        T = SIM_DURATION[mission]
        ts = np.arange(0.0, T + SIM_DT, SIM_DT)
        traj = np.array([mob.position(uav_id, float(t)) for t in ts])
        v = speed_from_xyz(ts, traj[:, 0], traj[:, 1], traj[:, 2])
        sim_speeds[mission] = v

    fig, axarr = plt.subplots(1, 4, figsize=(FULL_W, 2.0),
                               sharey=True, constrained_layout=True)
    handles_for_legend = None
    for i, mission in enumerate(MISSIONS):
        ax = axarr[i]
        sim = sim_speeds[mission]
        meas = meas_speeds[mission]
        l_sim = l_afar = None
        if sim.size:
            xs, ys = cdf(sim)
            l_sim, = ax.plot(xs, ys, color='C0', linewidth=1.3,
                             label='sim')
        if meas.size:
            xm, ym = cdf(meas)
            l_afar, = ax.plot(xm, ym, color='C3', linewidth=1.3,
                              linestyle='--', label='AFAR')
        jsd = jsd_speed(sim, meas)
        ax.set_xlabel('speed (m/s)')
        if i == 0:
            ax.set_ylabel('CDF')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.text(0.04, 0.96,
                f"{PANEL_LABELS[i]}\nJSD={jsd:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))
        if i == 0 and l_sim is not None and l_afar is not None:
            handles_for_legend = [l_sim, l_afar]

        if handles_for_legend is not None:
            fig.legend(handles=handles_for_legend,
                   labels=['sim', 'AFAR'],
                   loc='outside lower center', ncol=2, frameon=False)

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
