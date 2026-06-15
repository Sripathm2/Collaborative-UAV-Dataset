#!/usr/bin/env python3
"""
fill_eval_tables.py

ONE pass over server_*/ raw folders, emits three eval tables:

  - tab_perclass_stats.tex     (T-new-5)  per-class descriptive flow stats
  - tab_separability_stats.tex (T-new-7)  per-class mean +/- std on key features
  - tab_internal_diversity.tex (T-new-6)  per-axis Hellinger / JSD
                                          (matches diversity_table.py metric)

Why direct from raw folders, not UAV-CAS_ts.csv?
The CSV strips config-axis metadata (num_drones, pathloss, mission, tx_power)
that T-new-6 needs. Re-walking the source is cheaper than rebuilding the CSV
with extra columns.

Metric conventions:
  - Hellinger  = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))   on density histograms
  - "JSD"      = KL(p||q) + KL(q||p)                       on re-normalised hist
  These are the SAME formulas used in diversity_table.py / tab:diversity, so
  T-new-6 numbers are directly comparable to the existing T8 row scale.

Usage:
  python3 5-fill_eval_tables.py --root ../UAV-cas-dataset
"""

import argparse
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import entropy

# ============================================================================
# parsers (lifted verbatim from build_ts_csv.py - keep in sync)
# ============================================================================

CONFIG_AXES = ['attacks', 'num_drones', 'num_bs', 'payload',
               'pathloss', 'modulation', 'missions', 'tx_power', 'noise']
OLD_DEFAULTS = {'pathloss': 'logdist', 'modulation': 'adaptive',
                'missions': 'spiral', 'tx_power': '20', 'noise': '95'}

RE_SCENARIO  = re.compile(r"scenario raw='([^']+)'")
RE_REPLAY    = re.compile(r"Replay\s+attacker=(d\d+)\s+victim=(d\d+)\s+length=(\d+)")
RE_WORMHOLE  = re.compile(r"Wormhole\s+attacker=(d\d+)\s+victims?=(\[[^\]]*\])")
RE_DOS       = re.compile(r"(?<![Dd])DoS\s+attacker=(d\d+)\s+victim=(d\d+)")
RE_DDOS_VIC  = re.compile(r"\bDDoS.*?victim=(d\d+)")
RE_DDOS_ATKS = re.compile(r"\bDDoS.*?attackers?=(\[[^\]]+\])")
RE_BH_VIC    = re.compile(r"\bBlackhole.*?victim[s\(\)]*\s*=\s*(\[[^\]]+\])")
RE_FLOWS_NAME = re.compile(r'^flows-(\d{8})-(\d{4})-(\d+)\.txt$')

LABEL_MAP = {'benign':'Benign', 'dos':'DoS', 'ddos':'DDoS',
             'blackhole':'Blackhole', 'wormhole':'Wormhole', 'replay':'Replay'}


def parse_config_string(cfg):
    parts = cfg.strip().split('-')
    if len(parts) == 9:
        return dict(zip(CONFIG_AXES, parts))
    if len(parts) == 4:
        d = dict(zip(CONFIG_AXES[:4], parts)); d.update(OLD_DEFAULTS); return d
    raise ValueError(f"bad config: {cfg!r}")


def parse_attack_details(path: Path):
    info = {'scenario': '', 'attacks': []}
    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith('***'):
                continue
            content = line.split(' ', 1)[1] if ' ' in line else line
            m = RE_SCENARIO.search(content)
            if m: info['scenario'] = m.group(1); continue
            m = RE_REPLAY.search(content)
            if m:
                info['attacks'].append({'type':'replay',
                                        'attacker':m.group(1),
                                        'victim':m.group(2)})
                continue
            m = RE_WORMHOLE.search(content)
            if m:
                info['attacks'].append({'type':'wormhole',
                                        'attacker':m.group(1),
                                        'victims':ast.literal_eval(m.group(2))})
                continue
            m = RE_DOS.search(content)
            if m:
                info['attacks'].append({'type':'dos',
                                        'attacker':m.group(1),
                                        'victim':m.group(2)})
                continue
            if 'DDoS' in content:
                mv = RE_DDOS_VIC.search(content); ma = RE_DDOS_ATKS.search(content)
                if mv:
                    info['attacks'].append({
                        'type':'ddos',
                        'attackers':ast.literal_eval(ma.group(1)) if ma else [],
                        'victim':mv.group(1)})
                    continue
            if 'Blackhole' in content:
                mv = RE_BH_VIC.search(content)
                if mv:
                    info['attacks'].append({'type':'blackhole',
                                            'victims':ast.literal_eval(mv.group(1))})
                    continue
    return info


def drone_to_ip(name):
    return f"10.0.0.{int(name[1:])}"


def label_components(ip_a, ip_b, attacks):
    pair = {ip_a, ip_b}
    out = []
    for atk in attacks:
        t = atk['type']
        if t == 'dos':
            if drone_to_ip(atk['attacker']) in pair and drone_to_ip(atk['victim']) in pair:
                out.append('dos')
        elif t == 'ddos':
            atks_ip = {drone_to_ip(d) for d in atk.get('attackers', [])}
            if drone_to_ip(atk['victim']) in pair and (pair & atks_ip):
                out.append('ddos')
        elif t == 'blackhole':
            if pair & {drone_to_ip(d) for d in atk.get('victims', [])}:
                out.append('blackhole')
        elif t == 'wormhole':
            if pair & {drone_to_ip(d) for d in atk.get('victims', [])}:
                out.append('wormhole')
        elif t == 'replay':
            if drone_to_ip(atk['victim']) in pair:
                out.append('replay')
    seen = set(); uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


def canon_label(comps):
    if not comps:
        return 'Benign'
    return '+'.join(LABEL_MAP[c] for c in comps)


def iter_flows(path: Path):
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                key, pkts_str = line.split('\t', 1)
            except ValueError:
                continue
            ips = key.split(' <-> ')
            if len(ips) != 2:
                continue
            try:
                pkts = ast.literal_eval(pkts_str.strip())
            except Exception:
                continue
            if not pkts:
                continue
            ts = [p[0] for p in pkts]
            sz = [p[1] for p in pkts]
            yield ips[0].strip(), ips[1].strip(), ts, sz


def find_pairs(server_dir: Path):
    out = []
    for fp in sorted(server_dir.glob('flows-*.txt')):
        m = RE_FLOWS_NAME.match(fp.name)
        if not m:
            continue
        date, time, cfg = m.groups()
        ap = server_dir / f'attack_details_{date}-{time}-{cfg}.txt'
        if ap.exists():
            out.append((fp, ap, int(cfg), f"{date}-{time}"))
    return out


# ============================================================================
# stats accumulator (T-new-5 + T-new-7)
# ============================================================================

class PerClassStats:
    """Per-class lists of flow scalars for descriptive stats."""

    def __init__(self):
        self.count    = Counter()
        self.n_pkts   = defaultdict(list)
        self.duration = defaultdict(list)
        self.bytes_   = defaultdict(list)
        self.mean_iat = defaultdict(list)   # ms
        self.pkt_rate = defaultdict(list)   # pps
        self.pkt_size = defaultdict(list)   # bytes/pkt

    def add(self, label, ts, sz):
        n = len(ts)
        if n < 2:
            return
        dur = float(ts[-1] - ts[0])
        if dur <= 0:
            return
        b = int(sum(sz))
        iat = np.diff(np.asarray(ts, dtype=np.float64))
        iat = iat[iat > 0]
        if iat.size == 0:
            return
        self.count[label] += 1
        self.n_pkts[label].append(int(n))
        self.duration[label].append(dur)
        self.bytes_[label].append(b)
        self.mean_iat[label].append(float(np.mean(iat)) * 1000.0)
        self.pkt_rate[label].append(float(n) / dur)
        self.pkt_size[label].append(float(b) / float(n))


# ============================================================================
# IAT density histogram + divergence (matches diversity_table.py)
# ============================================================================

N_BINS = 50

def iat_density_hist(ts):
    pt = np.asarray(ts, dtype=np.float64)
    if len(pt) < 2:
        return None
    iat = np.diff(pt)
    iat = iat[iat > 0]
    if iat.size < 2:
        return None
    h, _ = np.histogram(iat, bins=N_BINS, density=True)
    return h.astype(np.float64)


def hellinger_density(p, q):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2)))


def sym_kl(p, q, eps=1e-10):
    a = p + eps; a = a / a.sum()
    b = q + eps; b = b / b.sum()
    return float(entropy(a, b) + entropy(b, a))


# ============================================================================
# axis definitions for T-new-6
# ============================================================================

AXES = [
    # (display label, config key, low value, high value)
    (r"5 vs.\ 20 drones",                  'num_drones',  '5',       '20'),
    (r"Log-distance vs.\ 3GPP path loss",  'pathloss',    'logdist', '3gpp'),
    (r"Spiral vs.\ random mission",        'missions',    'spiral',  'random'),
    (r"TX power 10 vs.\ 30\,dBm",          'tx_power',    '10',      '30'),
]

N_PER_GROUP = 200       # reservoir size per (axis, value)


# ============================================================================
# table emitters
# ============================================================================

def fmt_mean_median(arr):
    if not arr:
        return "--"
    m = float(np.mean(arr))
    md = float(np.median(arr))
    if abs(m) >= 1000 or abs(md) >= 1000:
        return f"{m:,.0f} ({md:,.0f})"
    if abs(m) >= 10:
        return f"{m:,.1f} ({md:,.1f})"
    return f"{m:,.2f} ({md:,.2f})"


def fmt_mean_std(arr):
    if not arr:
        return "--"
    m = float(np.mean(arr))
    s = float(np.std(arr))
    if abs(m) >= 1000 or s >= 1000:
        return f"{m:,.0f} $\\pm$ {s:,.0f}"
    if abs(m) >= 10:
        return f"{m:,.1f} $\\pm$ {s:,.1f}"
    return f"{m:,.2f} $\\pm$ {s:,.2f}"


def write_table_perclass(stats: PerClassStats, out_path: Path):
    canon = ['Benign', 'DoS', 'DDoS', 'Blackhole', 'Wormhole', 'Replay']
    collab_labels = [k for k in stats.count if '+' in k]

    lines = []
    lines.append(r"% Table T-new-5: tab:perclass_stats")
    lines.append(r"% Per-class flow-level descriptive statistics on UAV-CAS.")
    lines.append(r"% Mean (median in parentheses).")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class flow-level descriptive statistics on \datasetname. "
                 r"Reported as mean (median in parentheses).}")
    lines.append(r"\label{tab:perclass_stats}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Class} & \textbf{\#flows} & \textbf{\#pkts} & "
                 r"\textbf{Dur (s)} & \textbf{Bytes} & \textbf{IAT (ms)} \\")
    lines.append(r"\hline")

    for c in canon:
        if c in stats.count and stats.count[c] > 0:
            lines.append(f"{c:<13s} & {stats.count[c]:,} & "
                         f"{fmt_mean_median(stats.n_pkts[c])} & "
                         f"{fmt_mean_median(stats.duration[c])} & "
                         f"{fmt_mean_median(stats.bytes_[c])} & "
                         f"{fmt_mean_median(stats.mean_iat[c])} \\\\")
        else:
            lines.append(f"{c:<13s} & 0 & -- & -- & -- & -- \\\\")

    if collab_labels:
        n_pkts_c   = [v for k in collab_labels for v in stats.n_pkts[k]]
        duration_c = [v for k in collab_labels for v in stats.duration[k]]
        bytes_c    = [v for k in collab_labels for v in stats.bytes_[k]]
        iat_c      = [v for k in collab_labels for v in stats.mean_iat[k]]
        count_c    = sum(stats.count[k] for k in collab_labels)
        lines.append(f"Collaborative & {count_c:,} & "
                     f"{fmt_mean_median(n_pkts_c)} & "
                     f"{fmt_mean_median(duration_c)} & "
                     f"{fmt_mean_median(bytes_c)} & "
                     f"{fmt_mean_median(iat_c)} \\\\")
    else:
        lines.append(r"Collaborative & 0 & -- & -- & -- & -- \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_path}")


def write_table_separability(stats: PerClassStats, out_path: Path):
    canon = ['Benign', 'DoS', 'DDoS', 'Blackhole', 'Wormhole', 'Replay']

    lines = []
    lines.append(r"% Table T-new-7: tab:separability_stats")
    lines.append(r"% Per-class summary statistics on key separability features.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class summary statistics on key separability features. "
                 r"Mean $\pm$ standard deviation across all flows in \datasetname.}")
    lines.append(r"\label{tab:separability_stats}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Class} & \textbf{IAT (ms)} & \textbf{Pkt rate (pps)} & "
                 r"\textbf{Duration (s)} & \textbf{Pkt size (B)} \\")
    lines.append(r"\hline")
    for c in canon:
        if c in stats.count and stats.count[c] > 0:
            lines.append(f"{c:<10s} & {fmt_mean_std(stats.mean_iat[c])} & "
                         f"{fmt_mean_std(stats.pkt_rate[c])} & "
                         f"{fmt_mean_std(stats.duration[c])} & "
                         f"{fmt_mean_std(stats.pkt_size[c])} \\\\")
        else:
            lines.append(f"{c:<10s} & -- & -- & -- & -- \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_path}")


def write_table_internal_diversity(reservoirs, seen_per_group, out_path: Path, axis_class: str):
    if axis_class == 'all':
        class_note = "all classes mixed"
    else:
        class_note = f"{axis_class}-only"
    lines = []
    lines.append(r"% Table T-new-6: tab:internal_diversity")
    lines.append(f"% Restricted to {class_note} flows to isolate the axis effect from")
    lines.append(r"% class-mix confounding. Metric: 'JSD' = symmetric KL on re-normalised")
    lines.append(r"% density histograms (matches tab:diversity convention).")
    lines.append(r"% Hellinger column dropped: density-histogram Hellinger is dominated by")
    lines.append(r"% gross IAT-magnitude differences (attack class), not by axis-induced")
    lines.append(r"% within-class spread, so it gives near-zero values here and is uninformative.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Internal diversity by configuration axis: symmetric Jensen--Shannon "
                 r"divergence between flow inter-arrival-time distributions induced by extreme "
                 r"values of each axis (other axes mixed). Computed on \textbf{" + axis_class.lower() +
                 r"} flows to isolate the axis effect from class-mix confounding. Higher values "
                 r"indicate the axis produces measurably different flow distributions.}")
    lines.append(r"\label{tab:internal_diversity}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Axis (low vs.\ high)} & \textbf{JSD} \\")
    lines.append(r"\hline")

    for axis_label, axis_key, low_val, high_val in AXES:
        H_lo = reservoirs.get((axis_key, low_val), [])
        H_hi = reservoirs.get((axis_key, high_val), [])
        n_lo = seen_per_group.get((axis_key, low_val), 0)
        n_hi = seen_per_group.get((axis_key, high_val), 0)
        if len(H_lo) < 2 or len(H_hi) < 2:
            print(f"  [skip axis] {axis_key}: low n={n_lo} (kept={len(H_lo)}), "
                  f"high n={n_hi} (kept={len(H_hi)}) -- not enough")
            lines.append(f"{axis_label:<35s} & -- \\\\")
            continue

        Lo = np.stack(H_lo); Hi = np.stack(H_hi)
        Js = []
        for p in Lo:
            for q in Hi:
                Js.append(sym_kl(p, q))
        j_mean = float(np.mean(Js))

        print(f"  [{axis_key}] {low_val} (n={n_lo}, kept={len(H_lo)})  vs  "
              f"{high_val} (n={n_hi}, kept={len(H_hi)})  JSD={j_mean:.2f}")
        lines.append(f"{axis_label:<35s} & {j_mean:.2f} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_path}")


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path,
                    help='Path to UAV-cas-dataset root with server_*/ folders')
    ap.add_argument('--out-perclass',     default=Path('../results/Step_5/tab_perclass_stats.tex'),     type=Path)
    ap.add_argument('--out-separability', default=Path('../results/Step_5/tab_separability_stats.tex'), type=Path)
    ap.add_argument('--out-internal',     default=Path('../results/Step_5/tab_internal_diversity.tex'), type=Path)
    ap.add_argument('--min-packets', type=int, default=10,
                    help='Drop flows shorter than this (default 10, matches process_csvs.py)')
    ap.add_argument('--axis-class', default='Benign',
                    help="Restrict T-new-6 axis-divergence comparison to flows of this class "
                         "(default 'Benign'). Use 'all' to mix all classes (legacy behaviour, "
                         "values washed out by class confounding).")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    server_dirs = sorted([d for d in args.root.iterdir()
                          if d.is_dir() and d.name.startswith('server_')])
    if not server_dirs:
        raise SystemExit(f"no server_*/ folders under {args.root}")
    print(f"found {len(server_dirs)} server dirs")

    stats          = PerClassStats()
    reservoirs     = defaultdict(list)
    seen_per_group = defaultdict(int)
    n_pairs_ok     = 0
    n_pairs_bad    = 0
    n_flows        = 0

    for sd in server_dirs:
        pairs = find_pairs(sd)
        print(f"  {sd.name}: {len(pairs)} pairs")
        for fp, ap_path, cfg_idx, run_id in pairs:
            try:
                info = parse_attack_details(ap_path)
                if not info['scenario']:
                    n_pairs_bad += 1
                    continue
                cfg = parse_config_string(info['scenario'])
            except Exception as e:
                print(f"    SKIP {ap_path.name}: {e}")
                n_pairs_bad += 1
                continue

            n_pairs_ok += 1
            for ip_a, ip_b, ts, sz in iter_flows(fp):
                if len(ts) < args.min_packets:
                    continue

                comps = label_components(ip_a, ip_b, info['attacks'])
                lab = canon_label(comps)

                # T-new-5 / T-new-7 stats (all classes)
                stats.add(lab, ts, sz)
                n_flows += 1

                # T-new-6 axis reservoirs (restricted to a single class to isolate
                # the axis effect from class-mix confounding; default 'Benign')
                if args.axis_class != 'all' and lab != args.axis_class:
                    continue
                hist = iat_density_hist(ts)
                if hist is None:
                    continue
                for _, axis_key, low_val, high_val in AXES:
                    val = cfg.get(axis_key)
                    if val not in (low_val, high_val):
                        continue
                    key = (axis_key, val)
                    seen_per_group[key] += 1
                    n = seen_per_group[key]
                    if n <= N_PER_GROUP:
                        reservoirs[key].append(hist)
                    else:
                        j = int(rng.integers(0, n))
                        if j < N_PER_GROUP:
                            reservoirs[key][j] = hist

    print(f"\nprocessed {n_pairs_ok} pairs OK, {n_pairs_bad} bad/skipped")
    print(f"kept {n_flows:,} flows (>= {args.min_packets} packets, IAT > 0)")
    print("\nlabel counts:")
    for lab, c in sorted(stats.count.items(), key=lambda kv: -kv[1]):
        print(f"  {lab:<35s} {c:,}")

    print("\naxis-group sample sizes:")
    for k, v in sorted(seen_per_group.items()):
        print(f"  {k}: seen={v:,}  kept={len(reservoirs[k])}")

    print()
    write_table_perclass(stats, args.out_perclass)
    write_table_separability(stats, args.out_separability)
    write_table_internal_diversity(reservoirs, seen_per_group,
                                   args.out_internal, args.axis_class)


if __name__ == '__main__':
    main()