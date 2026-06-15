#!/usr/bin/env python3
"""
fig_mobility_traces.py   [Fig 3 — full-width figure*, 1x4]

4-panel 3D trajectory figure showing one representative trace per mission
type. Layout: figsize=(7.0, 2.2). ≤4 ticks per axis on every 3D panel.
Consistent elev/azim. Panels identified by corner text, NOT a title.
"""

import argparse
import random as _rnd
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

GRID = 1000.0
ALT_MIN = 30.0
ALT_MAX = 110.0
DT = 0.5
DURATIONS = {
    'spiral':        600.0,
    'grid':          120.0,
    'hover_transit': 240.0,
    'random':        180.0,
}
MISSIONS = ['spiral', 'grid', 'hover_transit', 'random']
PANEL_LABELS = ['(a) spiral', '(b) grid', '(c) hover_transit', '(d) random']

# Consistent viewing angle across all 4 panels
ELEV = 22.0
AZIM = -60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simlib-path', required=True, type=Path)
    ap.add_argument('--layer2',      required=True, type=Path)
    ap.add_argument('--out-pdf',     default=Path('fig_mobility_traces.pdf'), type=Path)
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(args.simlib_path))
    from simlib.mobility import MobilityGenerator

    fig = plt.figure(figsize=(FULL_W, 2.2), constrained_layout=True)
    handles_legend = None

    for i, mission in enumerate(MISSIONS):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')

        mob = MobilityGenerator(str(args.layer2),
                                 rng=_rnd.Random(args.seed + i))
        anchor_rng = _rnd.Random(args.seed + 100 + i)
        anchor = (anchor_rng.uniform(50.0, GRID - 50.0),
                  anchor_rng.uniform(50.0, GRID - 50.0))
        uav_id = f'd_{mission}'
        mob.create_uav(uav_id=uav_id, mission_type=mission,
                       origin_xy=anchor, grid_size_m=GRID,
                       alt_min=ALT_MIN, alt_max=ALT_MAX)

        T = DURATIONS[mission]
        ts = np.arange(0.0, T + DT, DT)
        traj = np.array([mob.position(uav_id, float(t)) for t in ts])
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

        # trajectory colored by time
        line_segs = ax.scatter(x, y, z, c=ts, cmap='viridis',
                                s=0.5, linewidth=0)
        line_segs.set_rasterized(True)

        start = ax.scatter(x[0], y[0], z[0], c='lime', s=20, marker='o',
                            edgecolor='black', linewidth=0.5, zorder=5)
        end   = ax.scatter(x[-1], y[-1], z[-1], c='red', s=20, marker='X',
                            edgecolor='black', linewidth=0.5, zorder=5)

        # ≤4 ticks per axis
        for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis_obj.set_major_locator(MaxNLocator(nbins=4))
        # smaller 3D tick labels
        for tick in (ax.get_xticklabels() + ax.get_yticklabels()
                     + ax.get_zticklabels()):
            tick.set_fontsize(6)

        ax.set_xlabel('x (m)', fontsize=7, labelpad=-2)
        ax.set_ylabel('y (m)', fontsize=7, labelpad=-2)
        ax.set_zlabel('z (m)', fontsize=7, labelpad=-2)
        ax.set_zlim(0, ALT_MAX + 10)
        ax.view_init(elev=ELEV, azim=AZIM)

        # panel identifier as 2D text annotation (figure coords)
        ax.text2D(0.04, 0.96, PANEL_LABELS[i], transform=ax.transAxes,
                  ha="left", va="top", fontsize=7,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white",
                            ec="none", alpha=0.85))
        if i == 0:
            handles_legend = [start, end]

    if handles_legend is not None:
        fig.legend(handles=handles_legend, labels=['start', 'end'],
                   loc='lower center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
