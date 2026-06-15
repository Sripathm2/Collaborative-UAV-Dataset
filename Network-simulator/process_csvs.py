#!/usr/bin/env python3
"""
process_csvs.py  (v2: direction-aware)

Reads tshark-converted CSVs under ./pcaps/, groups by (date, time, config),
aggregates per-flow packet lists, and writes flows-*.txt.

CHANGED in v2: each packet stored as a 4-tuple
        (timestamp, packet_size, tcp_flag_hex_str, direction_bit)
where direction_bit is:
    0  if packet's src_ip == the alphabetically smaller IP in the sorted pair
    1  otherwise

The IP-pair KEY is still sorted alphabetically (so (A<->B) and (B<->A) collapse),
preserving v1 file layout. Direction information lives entirely in the
per-packet 4-tuple's 4th field. Downstream build_ts_csv.py / build_stat_csv.py
will read this bit; if the field is missing (legacy v1 files), they treat all
packets as dir=0 and warn once.
"""
import os
import re
import csv
from collections import defaultdict


current_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir     = os.path.join(current_dir, 'pcaps')
max_packets = 100000

RE_CSV_NAME = re.compile(
    r'^mn\.[a-z0-9]+-(\d{8})-(\d{4})-(\d+)\.pcap\.csv$'
)


def normalize_ip_pair(ip1, ip2):
    """Ensure (src, dst) and (dst, src) are treated the same."""
    return tuple(sorted([ip1, ip2]))


def process_single_csv(filepath):
    flows = defaultdict(list)
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 9:
                continue
            try:
                timestamp = float(row[0])
                pkt_size  = int(row[1])
                # tshark with -i any may output multiple IP layers as
                # comma-separated values (e.g. '10.0.0.6,10.0.0.4').
                # The last value is the innermost (actual endpoint) IP.
                src_ip = row[2].split(',')[-1].strip()
                dst_ip = row[3].split(',')[-1].strip()
                if not src_ip or not dst_ip:
                    continue
                if (len(row[4]) == 0 and len(row[5]) == 0
                        and len(row[6]) == 0 and len(row[7]) == 0):
                    continue
                tcp_flag = row[8].strip()
                key = normalize_ip_pair(src_ip, dst_ip)
                # direction bit: 0 if sender is the alphabetically-smaller
                # endpoint, 1 if sender is the alphabetically-larger one.
                direction = 0 if src_ip == key[0] else 1
                flows[key].append((timestamp, pkt_size, tcp_flag, direction))
            except ValueError:
                print(f"Skipping malformed row {row}")
    return flows


def append_flows_to_output(flows, out_f):
    for (ip1, ip2), packet_list in flows.items():
        packet_list.sort()
        if len(packet_list) > max_packets:
            packet_list = packet_list[:max_packets]
        if len(packet_list) < 10:
            continue
        out_f.write(f"{ip1} <-> {ip2}\t{packet_list}\n")


if __name__ == "__main__":
    if not os.path.isdir(csv_dir):
        print(f"ERROR: {csv_dir} is not a directory")
        raise SystemExit(1)

    # Group CSVs by (yyyymmdd, hhmm, config_idx)
    groups = defaultdict(list)
    skipped_names = []
    for fn in sorted(os.listdir(csv_dir)):
        if not fn.endswith('.csv'):
            continue
        m = RE_CSV_NAME.match(fn)
        if m is None:
            skipped_names.append(fn)
            continue
        groups[(m.group(1), m.group(2), m.group(3))].append(
            os.path.join(csv_dir, fn)
        )

    if not groups:
        print(f"No matching CSV files in {csv_dir}; nothing to do.")
        raise SystemExit(0)

    total_csvs = sum(len(v) for v in groups.values())
    print(f"Found {total_csvs} CSVs in {len(groups)} group(s)")
    if skipped_names:
        print(f"  Skipped {len(skipped_names)} unrecognized filenames")

    processed = 0
    already_done = 0
    for (yyyymmdd, hhmm, cfg), csv_paths in sorted(groups.items()):
        tag = f"{yyyymmdd}-{hhmm}-{cfg}"
        output_file = os.path.join(csv_dir, f"flows-{tag}.txt")
        if os.path.exists(output_file):
            already_done += 1
            continue
        with open(output_file, "w") as out_f:
            for cp in sorted(csv_paths):
                per_file_flows = process_single_csv(cp)
                append_flows_to_output(per_file_flows, out_f)
        processed += 1
        print(f"  -> flows-{tag}.txt  ({len(csv_paths)} csvs)")

    print(f"Done: {processed} new, {already_done} skipped (already exist)")