#!/usr/bin/env python3
"""
build_ts_csv.py  (v3: run_id + IP pair columns for multi-flow context)

Walks server_*/ folders containing raw flows-*.txt + attack_details_*.txt and
emits UAV-CAS_ts.csv with list columns plus per-flow scalars:

    packet_time, packet_size, [packet_flag,] packet_dir, Label, run_id, ip_a, ip_b

run_id = "<server_dir>:<date>-<time>-<cfg>" — one value per (flows, attack_details)
pair, i.e. per simulation run. It is the grouping key for run-context features
in build_features.py, the unit for group-based train/test splitting, and the
natural batch for Level-2 collaborative detection. Scalar column: the run's
flows are NOT duplicated into each row.

ip_a, ip_b = the two endpoints of the flow as they appear in flows-*.txt
("a <-> b"). Used downstream for destination-convergence context features.

packet_dir[i] is the direction bit emitted by process_csvs.py:
    0 = packet sent from the alphabetically-smaller IP in the sorted pair
    1 = packet sent from the alphabetically-larger IP in the sorted pair

If --embed_config is set, the Label column becomes:
        "<canonical_label>|<config_string>"
Downstream code can split on '|': index 0 is the canonical IDS label,
index 1 is the verbatim 9-axis config string from attack_details.

Backward compat: if flows-*.txt has 3-tuples (legacy v1 data), all packets are
treated as dir=0 and a one-time warning is printed. Older downstream readers
that map columns by keyword (time/size/flag/dir/label) ignore the new columns.

Usage:
  python3 build_ts_csv.py \
      --root /path/to/UAV-cas-dataset \
      --out-csv UAV-CAS_ts.csv

  # FABLE-paper version: keep config string per row
  python3 build_ts_csv.py --root ... --out-csv UAV-CAS_ts_cfg.csv --embed_config

  # drop collaborative rows (single-class only)
  python3 build_ts_csv.py --root ... --out-csv UAV-CAS_ts_single.csv --single-only

  # also emit Table 7 (dataset stats) tex from the same pass
  python3 build_ts_csv.py --root ... --out-csv UAV-CAS_ts.csv \
      --out-tex table7_dataset_stats.tex
"""

import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path

# ----------------------------- config parsing ------------------------------

CONFIG_AXES = ['attacks', 'num_drones', 'num_bs', 'payload',
               'pathloss', 'modulation', 'missions', 'tx_power', 'noise']
OLD_DEFAULTS = {'pathloss': 'logdist', 'modulation': 'adaptive',
                'missions': 'spiral', 'tx_power': '20', 'noise': '95'}


def parse_config_string(cfg: str) -> dict:
    parts = cfg.strip().split('-')
    if len(parts) == 9:
        return dict(zip(CONFIG_AXES, parts))
    if len(parts) == 4:
        d = dict(zip(CONFIG_AXES[:4], parts))
        d.update(OLD_DEFAULTS)
        return d
    raise ValueError(f"bad config string: {cfg!r}")


# --------------------------- attack_details parser -------------------------

RE_SCENARIO  = re.compile(r"scenario raw='([^']+)'")
RE_NET_START = re.compile(r"network started drones=(\d+) bs=(\d+)")

RE_REPLAY    = re.compile(r"Replay\s+attacker=(d\d+)\s+victim=(d\d+)\s+length=(\d+)")
RE_WORMHOLE  = re.compile(r"Wormhole\s+attacker=(d\d+)\s+victims?=(\[[^\]]*\])")
RE_DOS       = re.compile(r"(?<![Dd])DoS\s+attacker=(d\d+)\s+victim=(d\d+)")
RE_DDOS_VIC  = re.compile(r"\bDDoS.*?victim=(d\d+)")
RE_DDOS_ATKS = re.compile(r"\bDDoS.*?attackers?=(\[[^\]]+\])")
RE_BH_VIC    = re.compile(r"\bBlackhole.*?victim[s\(\)]*\s*=\s*(\[[^\]]+\])")


def parse_attack_details(path: Path) -> dict:
    info = {'scenario': '', 'num_drones': None, 'num_bs': None, 'attacks': []}
    with open(path, 'r', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith('***'):
                continue
            content = line.split(' ', 1)[1] if ' ' in line else line

            m = RE_SCENARIO.search(content)
            if m: info['scenario'] = m.group(1); continue
            m = RE_NET_START.search(content)
            if m:
                info['num_drones'] = int(m.group(1))
                info['num_bs']     = int(m.group(2))
                continue
            m = RE_REPLAY.search(content)
            if m:
                info['attacks'].append({'type': 'replay',
                                        'attacker': m.group(1),
                                        'victim':   m.group(2)})
                continue
            m = RE_WORMHOLE.search(content)
            if m:
                info['attacks'].append({'type': 'wormhole',
                                        'attacker': m.group(1),
                                        'victims':  ast.literal_eval(m.group(2))})
                continue
            m = RE_DOS.search(content)
            if m:
                info['attacks'].append({'type': 'dos',
                                        'attacker': m.group(1),
                                        'victim':   m.group(2)})
                continue
            if 'DDoS' in content:
                mv = RE_DDOS_VIC.search(content)
                ma = RE_DDOS_ATKS.search(content)
                if mv:
                    info['attacks'].append({
                        'type': 'ddos',
                        'attackers': ast.literal_eval(ma.group(1)) if ma else [],
                        'victim':    mv.group(1),
                    })
                    continue
            if 'Blackhole' in content:
                mv = RE_BH_VIC.search(content)
                if mv:
                    info['attacks'].append({'type': 'blackhole',
                                            'victims': ast.literal_eval(mv.group(1))})
                    continue
    return info


# ------------------------------- labeling ----------------------------------

LABEL_MAP = {'benign':'Benign', 'dos':'DoS', 'ddos':'DDoS',
             'blackhole':'Blackhole', 'wormhole':'Wormhole', 'replay':'Replay'}


def drone_to_ip(name: str) -> str:
    return f"10.0.0.{int(name[1:])}"


def label_flow(ip_a: str, ip_b: str, attacks: list) -> list:
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


def canon_label_from_components(comps: list) -> str:
    if not comps:
        return 'Benign'
    return '+'.join(LABEL_MAP[c] for c in comps)


# ------------------------------ flows parser -------------------------------
# Backward-compat: handles 3-tuple (legacy) and 4-tuple (v2 with direction).
_warned_legacy = False


def iter_flows_with_dir(path: Path):
    """Yield (ip_a, ip_b, timestamps, sizes, flags, dirs) per flow.

    flags is a list of hex-string TCP flags (e.g. '0x10') aligned with
    timestamps/sizes; '0x00' if a packet has no flag field.
    dirs is a list of 0/1 ints aligned with timestamps/sizes.
    If the source file used 3-tuples (no direction field), dirs is all zeros
    and a one-time warning is printed.
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
            has_dir = (len(pkts[0]) >= 4)
            if has_dir:
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


# ----------------------------- folder walker -------------------------------

RE_FLOWS_NAME = re.compile(r'^flows-(\d{8})-(\d{4})-(\d+)\.txt$')


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


# -------------------------------- list_str ---------------------------------

def list_str_floats(vals):
    return '[' + ', '.join(repr(float(v)) for v in vals) + ']'


def list_str_ints(vals):
    return '[' + ', '.join(str(int(v)) for v in vals) + ']'


def list_str_strs(vals):
    """Render a list of strings as a Python-list-style literal.
    Uses repr() so each element keeps its quoting (round-trips through
    ast.literal_eval). E.g. ['0x10', '0x18'] -> "['0x10', '0x18']".
    """
    return '[' + ', '.join(repr(str(v)) for v in vals) + ']'


# ------------------------------ table 7 emit -------------------------------

CANON_ORDER = ['Benign', 'DoS', 'DDoS', 'Blackhole', 'Wormhole', 'Replay']


def _strip_config_suffix(label: str) -> str:
    """For Table 7 counts when --embed_config is on, strip the |<config> tail."""
    return label.split('|', 1)[0]


def write_table7(label_counter: Counter, out_path: Path, embedded: bool):
    canonical = {c: 0 for c in CANON_ORDER}
    collab = 0
    collab_breakdown = Counter()
    for k, v in label_counter.items():
        base = _strip_config_suffix(k) if embedded else k
        if base in canonical:
            canonical[base] += v
        else:
            collab += v
            collab_breakdown[base] += v
    total = sum(canonical.values()) + collab

    lines = [
        "% --- Table 7: tab:dataset_stats — flow counts per class ---",
        f"Benign        & {canonical['Benign']:,} \\\\",
        f"DoS           & {canonical['DoS']:,} \\\\",
        f"DDoS          & {canonical['DDoS']:,} \\\\",
        f"Blackhole     & {canonical['Blackhole']:,} \\\\",
        f"Wormhole      & {canonical['Wormhole']:,} \\\\",
        f"Replay        & {canonical['Replay']:,} \\\\",
        f"Collaborative & {collab:,} \\\\",
        r"\hline",
        f"\\textbf{{Total}} & \\textbf{{{total:,}}} \\\\",
        "",
        "% --- Collaborative composition breakdown ---",
    ]
    for k, v in sorted(collab_breakdown.items(), key=lambda kv: -kv[1]):
        lines.append(f"%   {k:<40s} {v:,}")
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_path}")


# ---------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',    required=True, type=Path)
    ap.add_argument('--out-csv', required=True, type=Path)
    ap.add_argument('--out-tex', type=Path, default=None,
                    help='Also emit Table 7 (dataset stats) tex')
    ap.add_argument('--single-only', action='store_true',
                    help='Drop collaborative (multi-attack) rows')
    ap.add_argument('--embed_config', action='store_true',
                    help='Append "|<config_string>" to every Label '
                         '(for FABLE paper; downstream split on "|" to recover)')
    ap.add_argument('--include_flags', action='store_true',
                    help='Add a third list column packet_flag (hex TCP flag '
                         'strings per packet) to the output.')
    ap.add_argument('--min-packets', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0,
                    help='Debug: stop after N (flows, attack_details) pairs')
    args = ap.parse_args()

    server_dirs = sorted([d for d in args.root.iterdir()
                          if d.is_dir() and d.name.startswith('server_')])
    if not server_dirs:
        raise SystemExit(f"no server_*/ folders under {args.root}")
    print(f"found {len(server_dirs)} server dirs")
    if args.embed_config:
        print("  --embed_config: Label = '<canonical>|<config_string>'")
    if args.include_flags:
        print("  --include_flags: emitting packet_flag column")

    label_counter = Counter()
    attack_type_counter = Counter()
    run_counter = Counter()              # flows per run (sanity stats)
    n_rows_written = 0
    n_rows_dropped_short = 0
    n_pairs_processed = 0
    n_pairs_failed = 0

    # header: list columns first, then scalars (run_id, ip_a, ip_b at the end
    # so keyword-based downstream column mapping is unaffected)
    if args.include_flags:
        header = ['packet_time', 'packet_size', 'packet_flag', 'packet_dir',
                  'Label', 'run_id', 'ip_a', 'ip_b']
    else:
        header = ['packet_time', 'packet_size', 'packet_dir',
                  'Label', 'run_id', 'ip_a', 'ip_b']

    with open(args.out_csv, 'w', newline='') as fout:
        w = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)

        for sd in server_dirs:
            pairs = find_pairs(sd)
            print(f"  {sd.name}: {len(pairs)} pairs")
            for fp, ap_path, cfg_idx, datetime_id in pairs:
                try:
                    info = parse_attack_details(ap_path)
                    if not info['scenario']:
                        n_pairs_failed += 1
                        continue
                    parse_config_string(info['scenario'])  # validate
                except Exception as e:
                    print(f"    SKIP {ap_path.name}: {e}")
                    n_pairs_failed += 1
                    continue

                run_id = f"{sd.name}:{datetime_id}-{cfg_idx}"
                attacks = info['attacks']
                for atk in attacks:
                    attack_type_counter[atk['type']] += 1
                for ip_a, ip_b, ts, sz, fl, dr in iter_flows_with_dir(fp):
                    if len(ts) < args.min_packets:
                        n_rows_dropped_short += 1
                        continue
                    comps = label_flow(ip_a, ip_b, attacks)
                    if args.single_only and len(comps) > 1:
                        continue
                    label = canon_label_from_components(comps)
                    if args.embed_config:
                        label = f"{label}|{info['scenario']}"
                    row = [list_str_floats(ts), list_str_ints(sz)]
                    if args.include_flags:
                        row.append(list_str_strs(fl))
                    row += [list_str_ints(dr), label, run_id, ip_a, ip_b]
                    w.writerow(row)
                    label_counter[label] += 1
                    run_counter[run_id] += 1
                    n_rows_written += 1

                n_pairs_processed += 1
                if args.limit and n_pairs_processed >= args.limit:
                    break
            if args.limit and n_pairs_processed >= args.limit:
                break

    sz_mb = args.out_csv.stat().st_size / 1e6
    print(f"\nwrote {args.out_csv}  ({sz_mb:.1f} MB)")
    print(f"  pairs processed: {n_pairs_processed}  failed: {n_pairs_failed}")
    print(f"  rows written:    {n_rows_written:,}")
    print(f"  rows dropped (<{args.min_packets} pkts): {n_rows_dropped_short:,}")
    if run_counter:
        import statistics
        fc = list(run_counter.values())
        print(f"  runs: {len(run_counter):,}  flows/run: "
              f"min={min(fc)} median={statistics.median(fc):.0f} "
              f"mean={statistics.mean(fc):.1f} max={max(fc)}")
    print("\nlabel counts:")
    for k, v in sorted(label_counter.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {k:<60s} {v:,}")
    if len(label_counter) > 30:
        print(f"  ... and {len(label_counter)-30} more")

    print("\nattack dispatches found in attack_details (across all configs):")
    for k, v in sorted(attack_type_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<10s} {v:,}")

    if args.out_tex:
        write_table7(label_counter, args.out_tex, embedded=args.embed_config)


if __name__ == '__main__':
    main()