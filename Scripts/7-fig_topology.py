#!/usr/bin/env python3
"""
fig_topology.py   [Fig 11 — full-width figure*, 2x2]

4-panel topology snapshot figure for one configuration. Per-window UAV
positions regenerated via simlib.mobility (same library Topo.py uses);
routing via simlib.routing.build_links at each window timestamp.

Layout: figsize=(7.0, 4.0). One shared altitude colorbar, one shared
legend (UAV / BS). Panel identifiers as small corner text, NOT titles.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

COL_W  = 3.33
FULL_W = 7.0
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
})

GRID_X = 1000
GRID_Y = 1000
ALT_MIN = 30
ALT_MAX = 110
WINDOW_SIZE_S = 5.0
SNAPSHOT_WINDOWS = [1, 4, 8, 12]
SEED = 42

RE_SCENARIO  = re.compile(r"scenario raw='([^']+)'")
RE_PLACED_BS = re.compile(r"placed BS (bs\d+) at \(([^)]+)\)")
RE_PLACED_UAV = re.compile(r"placed UAV (d\d+) mission=(\w+).*?start=\(([^)]+)\)")

CONFIG_AXES = ['attacks', 'num_drones', 'num_bs', 'payload',
               'pathloss', 'modulation', 'missions', 'tx_power', 'noise']
OLD_DEFAULTS = {'pathloss': 'logdist', 'modulation': 'adaptive',
                'missions': 'spiral', 'tx_power': '20', 'noise': '95'}


def parse_config_local(cfg):
    parts = cfg.strip().split('-')
    if len(parts) == 9:
        return dict(zip(CONFIG_AXES, parts))
    if len(parts) == 4:
        d = dict(zip(CONFIG_AXES[:4], parts)); d.update(OLD_DEFAULTS); return d
    raise ValueError(f"bad config: {cfg!r}")


def parse_attack_details_static(path):
    info = {'scenario': '', 'bs_positions': {}, 'uav_starts': {}}
    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith('***'):
                continue
            content = line.split(' ', 1)[1] if ' ' in line else line
            m = RE_SCENARIO.search(content)
            if m: info['scenario'] = m.group(1); continue
            m = RE_PLACED_BS.search(content)
            if m:
                xyz = [float(v.strip()) for v in m.group(2).split(',')]
                info['bs_positions'][m.group(1)] = tuple(xyz)
                continue
            m = RE_PLACED_UAV.search(content)
            if m:
                xyz = [float(v.strip()) for v in m.group(3).split(',')]
                info['uav_starts'][m.group(1)] = (m.group(2), tuple(xyz))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simlib-path',    required=True, type=Path)
    ap.add_argument('--layer1',         required=True, type=Path)
    ap.add_argument('--layer2',         required=True, type=Path)
    ap.add_argument('--config-str',     default=None)
    ap.add_argument('--attack-prefix',  default='benign')
    ap.add_argument('--attack-details', default=None, type=Path)
    ap.add_argument('--out-pdf',        default=Path('fig_topology.pdf'),
                    type=Path)
    ap.add_argument('--seed',           type=int, default=SEED)
    args = ap.parse_args()

    sys.path.insert(0, str(args.simlib_path))
    try:
        from simlib.mobility import MobilityGenerator
        from simlib.routing  import build_links
    except Exception as e:
        raise SystemExit(f"Cannot import simlib from {args.simlib_path}: {e}")

    if args.attack_details:
        info = parse_attack_details_static(args.attack_details)
        if not info['scenario']:
            raise SystemExit('attack_details has no scenario line')
        full_cfg = info['scenario']
        forced_bs = info['bs_positions']
    elif args.config_str:
        full_cfg = f"{args.attack_prefix}-{args.config_str}"
        forced_bs = {}
    else:
        raise SystemExit('need either --attack-details or --config-str')

    print(f"config: {full_cfg}")
    cfg = parse_config_local(full_cfg)
    num_drones = int(cfg['num_drones']); num_bs = int(cfg['num_bs'])
    mission_axis = cfg['missions']

    import random as _rnd
    mob = MobilityGenerator(str(args.layer2), rng=_rnd.Random(args.seed))
    placement_rng = _rnd.Random(args.seed + 1)

    bs_positions = {}
    for i in range(num_bs):
        name = f"bs{i+1}"
        if name in forced_bs:
            bs_positions[name] = tuple(forced_bs[name])
        else:
            bs_positions[name] = (
                placement_rng.uniform(0, GRID_X),
                placement_rng.uniform(0, GRID_Y),
                0.0,
            )
    bs_names = list(bs_positions.keys())

    if mission_axis == 'mixed':
        mission_per = [placement_rng.choice(
            ['spiral', 'grid', 'hover_transit', 'random']
        ) for _ in range(num_drones)]
    else:
        mission_per = [mission_axis] * num_drones

    uav_names = []
    for i in range(num_drones):
        name = f"d{i+1}"
        anchor = (
            placement_rng.uniform(50, GRID_X - 50),
            placement_rng.uniform(50, GRID_Y - 50),
        )
        mob.create_uav(
            uav_id=name,
            mission_type=mission_per[i],
            origin_xy=anchor,
            grid_size_m=float(GRID_X),
            alt_min=float(ALT_MIN),
            alt_max=float(ALT_MAX),
        )
        uav_names.append(name)

    snapshot_positions = {}
    snapshot_links = {}
    for w in SNAPSHOT_WINDOWS:
        t_abs = w * WINDOW_SIZE_S
        pos = {n: mob.position(n, t_abs) for n in uav_names}
        info = build_links(
            uav_names=uav_names,
            uav_positions=pos,
            bs_names=bs_names,
            bs_positions=bs_positions,
        )
        snapshot_positions[w] = pos
        snapshot_links[w] = info.get('links', [])
        print(f"  window {w}  t={t_abs:.0f}s  links={len(snapshot_links[w])}")

    # ---- render ----
    fig, axarr = plt.subplots(2, 2, figsize=(FULL_W, 4.0),
                               sharex=True, sharey=True,
                               constrained_layout=True)
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    sc = None
    for i, w in enumerate(SNAPSHOT_WINDOWS):
        ax = axarr[i // 2, i % 2]
        pos = snapshot_positions[w]; links = snapshot_links[w]

        for a, b in links:
            pa = pos.get(a) or bs_positions.get(a)
            pb = pos.get(b) or bs_positions.get(b)
            if pa is None or pb is None:
                continue
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color='gray', linewidth=0.5, alpha=0.55, zorder=1)

        ux = [p[0] for p in pos.values()]
        uy = [p[1] for p in pos.values()]
        uz = [p[2] for p in pos.values()]
        sc = ax.scatter(ux, uy, c=uz, cmap='viridis',
                         s=30, edgecolor='black', linewidth=0.3,
                         vmin=ALT_MIN, vmax=ALT_MAX, zorder=2)

        bx = [p[0] for p in bs_positions.values()]
        by = [p[1] for p in bs_positions.values()]
        ax.scatter(bx, by, marker='s', c='red', s=50,
                    edgecolor='black', linewidth=0.6, zorder=3)

        ax.set_xlim(0, GRID_X); ax.set_ylim(0, GRID_Y)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if i // 2 == 1:
            ax.set_xlabel('x (m)')
        if i % 2 == 0:
            ax.set_ylabel('y (m)')
        ax.text(0.04, 0.96,
                f"{panel_labels[i]} t={w * WINDOW_SIZE_S:.0f}s "
                f"({len(links)} links)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))

    # shared altitude colorbar
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axarr.ravel().tolist(),
                             shrink=0.7, label='altitude (m)',
                             pad=0.02)
        cbar.locator = MaxNLocator(5); cbar.update_ticks()

    # shared UAV / BS legend
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='', color='steelblue',
               markeredgecolor='black', markersize=6, label='UAV'),
        Line2D([0], [0], marker='s', linestyle='', color='red',
               markeredgecolor='black', markersize=6, label='BS'),
        Line2D([0], [0], linestyle='-', color='gray', linewidth=0.8,
               label='link'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
                ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(args.out_pdf)
    fig.savefig(str(args.out_pdf).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"wrote {args.out_pdf}")


if __name__ == '__main__':
    main()
