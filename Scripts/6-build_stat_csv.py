#!/usr/bin/env python3
"""
build_stat_csv.py  (v2: direction-aware + optional config embedding)

Walks server_*/ folders, parses every (flows-*.txt, attack_details_*.txt)
pair, and emits UAV-CAS_stat.csv with 47 per-flow features (25 flow + 22
Fwd/Bwd) plus 11 meta columns plus Label.

Direction is recovered from the 4-tuple (ts, size, flag, dir) emitted by
process_csvs.py v2. dir=0 means the packet was sent from the alphabetically-
smaller IP of the sorted pair (we call this Fwd by convention); dir=1 is Bwd.

If --embed_config is set, the Label column becomes
        "<canonical_label>|<config_string>"
exactly as in build_ts_csv.py.

Backward compat: 3-tuple (legacy v1) flow files are accepted, with all
packets treated as dir=0 and a one-time warning printed. In that case all
Bwd features will be zero/nan — re-run process_csvs.py v2 to recover them.

Schema (59 cols total):
  meta(11)     : config_idx, num_drones, num_bs, payload, pathloss,
                 modulation, mission, tx_power, noise, src_ip, dst_ip
  flow (25)    : Flow Duration, Total Packets, Total Length of Packets,
                 Flow Bytes/s, Flow Packets/s,
                 Flow IAT Total/Mean/Std/Max/Min,
                 Min/Max Packet Length,
                 Packet Length Mean/Std/Variance,
                 FIN/SYN/RST/PSH/ACK/URG/CWE/ECE Flag Count,
                 Header Length, Average Packet Size
  fwd/bwd (22) : Total Fwd Packets, Total Bwd Packets,
                 Total Length of Fwd Packets, Total Length of Bwd Packets,
                 Fwd Packets/s, Bwd Packets/s,
                 Fwd/Bwd Packet Length Max/Min/Mean/Std,
                 Fwd/Bwd IAT Total/Mean/Std,
                 Fwd Header Length, Bwd Header Length
  Label(1)

Usage:
  python3 6-build_stat_csv.py --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_stat.csv
  python3 6-build_stat_csv.py --root ../UAV-cas-dataset --out-csv ../Datasets/UAV-CAS_stat_cfg.csv --embed_config
"""

import argparse
import ast
import csv
import re
from pathlib import Path

import numpy as np


# ---------------- config + parsing (kept in sync with build_ts_csv.py) -----

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


def parse_attack_details(path):
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
                info['attacks'].append({'type':'replay','attacker':m.group(1),
                                        'victim':m.group(2)})
                continue
            m = RE_WORMHOLE.search(content)
            if m:
                info['attacks'].append({'type':'wormhole','attacker':m.group(1),
                                        'victims':ast.literal_eval(m.group(2))})
                continue
            m = RE_DOS.search(content)
            if m:
                info['attacks'].append({'type':'dos','attacker':m.group(1),
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
    return 'Benign' if not comps else '+'.join(LABEL_MAP[c] for c in comps)


# ------------------------------ flows parser -------------------------------
_warned_legacy = False


def iter_flows_with_flags_and_dir(path):
    """Yield (ip_a, ip_b, ts, sz, fl, dr).
    Backward compat: 3-tuple files -> dr = all zeros + one-time warning.
    """
    global _warned_legacy
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
            fl = [p[2] if len(p) > 2 else '0x00' for p in pkts]
            if len(pkts[0]) >= 4:
                dr = [int(p[3]) for p in pkts]
            else:
                if not _warned_legacy:
                    print("WARNING: legacy 3-tuple flow file detected "
                          f"({path.name}); direction unavailable, "
                          "all packets marked dir=0. "
                          "Re-run process_csvs.py (v2) to recover direction.")
                    _warned_legacy = True
                dr = [0] * len(pkts)
            yield ips[0].strip(), ips[1].strip(), ts, sz, fl, dr


def find_pairs(server_dir):
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


# ---------- features ----------

FEATURE_NAMES = [
    # flow-level (direction-agnostic): 25
    'Flow Duration', 'Total Packets', 'Total Length of Packets',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Total', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Header Length', 'Average Packet Size',
    # direction-aware: 22
    'Total Fwd Packets', 'Total Bwd Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Fwd Header Length', 'Bwd Header Length',
]

META_NAMES = ['config_idx', 'num_drones', 'num_bs', 'payload', 'pathloss',
              'modulation', 'mission', 'tx_power', 'noise', 'src_ip', 'dst_ip']


def hex_to_int(fl):
    try:
        return int(fl, 16)
    except (ValueError, TypeError):
        return 0


def _safe_stats(arr):
    """Return (max, min, mean, std) with zeros on empty array."""
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (float(arr.max()), float(arr.min()),
            float(arr.mean()), float(arr.std()))


def _iat_stats(times):
    """Return (total, mean, std). times is a numpy array sorted ascending."""
    if times.size < 2:
        return 0.0, 0.0, 0.0
    iat = np.diff(times)
    return float(iat.sum()), float(iat.mean()), float(iat.std())


def compute_features(ts, sz, fl, dr):
    n = len(ts)
    if n < 2:
        return None
    pt    = np.asarray(ts, dtype=np.float64)
    sizes = np.asarray(sz, dtype=np.int64)
    flags = np.asarray([hex_to_int(f) for f in fl], dtype=np.uint16)
    dirs  = np.asarray(dr, dtype=np.int8)

    duration = float(pt[-1] - pt[0])
    total_bytes = int(sizes.sum())
    iat = np.diff(pt)

    f = {}
    # ---- flow-level (direction-agnostic) ----
    f['Flow Duration']            = duration
    f['Total Packets']            = int(n)
    f['Total Length of Packets']  = total_bytes
    f['Flow Bytes/s']             = total_bytes / duration if duration > 0 else 0.0
    f['Flow Packets/s']           = n / duration if duration > 0 else 0.0
    if iat.size > 0:
        f['Flow IAT Total'] = float(iat.sum())
        f['Flow IAT Mean']  = float(iat.mean())
        f['Flow IAT Std']   = float(iat.std())
        f['Flow IAT Max']   = float(iat.max())
        f['Flow IAT Min']   = float(iat.min())
    else:
        for k in ['Flow IAT Total','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min']:
            f[k] = 0.0
    f['Min Packet Length']        = int(sizes.min())
    f['Max Packet Length']        = int(sizes.max())
    f['Packet Length Mean']       = float(sizes.mean())
    f['Packet Length Std']        = float(sizes.std())
    f['Packet Length Variance']   = float(sizes.var())
    f['FIN Flag Count']           = int(np.sum((flags & 0x01) != 0))
    f['SYN Flag Count']           = int(np.sum((flags & 0x02) != 0))
    f['RST Flag Count']           = int(np.sum((flags & 0x04) != 0))
    f['PSH Flag Count']           = int(np.sum((flags & 0x08) != 0))
    f['ACK Flag Count']           = int(np.sum((flags & 0x10) != 0))
    f['URG Flag Count']           = int(np.sum((flags & 0x20) != 0))
    f['CWE Flag Count']           = int(np.sum((flags & 0x40) != 0))
    f['ECE Flag Count']           = int(np.sum((flags & 0x80) != 0))
    f['Header Length']            = 20 * n
    f['Average Packet Size']      = total_bytes / n

    # ---- direction-aware ----
    fwd_mask = (dirs == 0)
    bwd_mask = ~fwd_mask
    n_fwd = int(fwd_mask.sum())
    n_bwd = int(bwd_mask.sum())
    sz_fwd = sizes[fwd_mask]
    sz_bwd = sizes[bwd_mask]
    pt_fwd = pt[fwd_mask]
    pt_bwd = pt[bwd_mask]
    f['Total Fwd Packets'] = n_fwd
    f['Total Bwd Packets'] = n_bwd
    f['Total Length of Fwd Packets'] = int(sz_fwd.sum()) if n_fwd else 0
    f['Total Length of Bwd Packets'] = int(sz_bwd.sum()) if n_bwd else 0
    f['Fwd Packets/s'] = (n_fwd / duration) if duration > 0 else 0.0
    f['Bwd Packets/s'] = (n_bwd / duration) if duration > 0 else 0.0
    mx, mn, me, sd = _safe_stats(sz_fwd)
    f['Fwd Packet Length Max']  = mx
    f['Fwd Packet Length Min']  = mn
    f['Fwd Packet Length Mean'] = me
    f['Fwd Packet Length Std']  = sd
    mx, mn, me, sd = _safe_stats(sz_bwd)
    f['Bwd Packet Length Max']  = mx
    f['Bwd Packet Length Min']  = mn
    f['Bwd Packet Length Mean'] = me
    f['Bwd Packet Length Std']  = sd
    tot, me, sd = _iat_stats(pt_fwd)
    f['Fwd IAT Total'] = tot
    f['Fwd IAT Mean']  = me
    f['Fwd IAT Std']   = sd
    tot, me, sd = _iat_stats(pt_bwd)
    f['Bwd IAT Total'] = tot
    f['Bwd IAT Mean']  = me
    f['Bwd IAT Std']   = sd
    f['Fwd Header Length'] = 20 * n_fwd
    f['Bwd Header Length'] = 20 * n_bwd
    return f


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--out-csv', required=True, type=Path)
    ap.add_argument('--embed_config', action='store_true',
                    help='Append "|<config_string>" to every Label '
                         '(for FABLE paper; downstream split on "|" to recover)')
    ap.add_argument('--min-packets', type=int, default=10)
    args = ap.parse_args()

    server_dirs = sorted([d for d in args.root.iterdir()
                          if d.is_dir() and d.name.startswith('server_')])
    if not server_dirs:
        raise SystemExit(f"no server_*/ folders under {args.root}")
    print(f"found {len(server_dirs)} server dirs")
    if args.embed_config:
        print("  --embed_config: Label = '<canonical>|<config_string>'")

    cols = META_NAMES + FEATURE_NAMES + ['Label']
    n_rows = 0; n_short = 0; n_pairs_ok = 0

    with open(args.out_csv, 'w', newline='') as fout:
        w = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        w.writerow(cols)

        for sd in server_dirs:
            pairs = find_pairs(sd)
            print(f"  {sd.name}: {len(pairs)} pairs")
            for fp, ap_path, cfg_idx, run_id in pairs:
                try:
                    info = parse_attack_details(ap_path)
                    if not info['scenario']:
                        continue
                    cfg = parse_config_string(info['scenario'])
                except Exception as e:
                    print(f"    SKIP {ap_path.name}: {e}")
                    continue
                n_pairs_ok += 1

                meta = [cfg_idx,
                        cfg['num_drones'], cfg['num_bs'], cfg['payload'],
                        cfg['pathloss'], cfg['modulation'], cfg['missions'],
                        cfg['tx_power'], cfg['noise']]

                for ip_a, ip_b, ts, sz, fl, dr in iter_flows_with_flags_and_dir(fp):
                    if len(ts) < args.min_packets:
                        n_short += 1
                        continue
                    feat = compute_features(ts, sz, fl, dr)
                    if feat is None:
                        continue
                    label = canon_label(label_components(ip_a, ip_b, info['attacks']))
                    if args.embed_config:
                        label = f"{label}|{info['scenario']}"
                    row = (list(meta) + [ip_a, ip_b]
                           + [feat[k] for k in FEATURE_NAMES] + [label])
                    w.writerow(row)
                    n_rows += 1

    print(f"\nwrote {args.out_csv}  ({args.out_csv.stat().st_size/1e6:.1f} MB)")
    print(f"  pairs OK: {n_pairs_ok}  rows: {n_rows:,}  dropped(<{args.min_packets}): {n_short:,}")


if __name__ == '__main__':
    main()