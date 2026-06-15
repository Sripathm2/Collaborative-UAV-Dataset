"""
generate_configs.py

Generates configurations.txt as the Cartesian product of axes you define
below. Run it once, point your bash script's line index at the resulting
file, and re-run this script whenever the campaign design changes.

Usage:
    python3 generate_configs.py                    # writes configurations.txt
    python3 generate_configs.py -o custom.txt      # custom path
    python3 generate_configs.py --validate         # also parse every line
                                                    through simlib.config_parser

Edit AXES below to shape your campaign. The 'attacks' axis is any mixture of
single attacks and '+'-combined attacks. Keep in mind: total line count =
product of all axis lengths, so watch your multipliers.
"""

import argparse
import itertools
import os
import sys

# ---------------- EDIT THIS BLOCK TO SHAPE YOUR CAMPAIGN ----------------

AXES = {
    # -- campaign variants --
    #
    # Attack combos use '+' separator. By convention we list combo components
    # in alphabetical order so logs and pcap filenames sort consistently. The
    # parser accepts either order, but consistency makes downstream analysis
    # easier.
    #
    # Replay parameters: <buffer_packets>,<delay_ms>,<rate_pps>,<ttl_dec>,<seq_mode>
    #   inc = monotonic increment, rand = uniform random, zero = always 0
    'attacks': [
        # ---- benign baseline ----
        'benign',

        # ---- single attacks ----
        'dos=1000',                  # 1 pps DoS  (hping3 -i u1000)
        # 'dos=100',                   # 10 pps DoS (hping3 -i u100)
        'ddos=1000',
        # 'ddos=100',
        'blackhole',
        'wormhole',
        'replay=50,200,10,5,inc',    # buf=50, 200ms delay, 10 pps, ttl-=5, seq++
        'replay=100,100,20,3,rand',  # buf=100, 100ms delay, 20 pps, ttl-=3, seq=random

        # ---- flood + stealth (DoS/DDoS with blackhole or wormhole) ----
        'blackhole+dos=1000',
        # 'blackhole+dos=100',
        'blackhole+ddos=1000',
        # 'blackhole+ddos=100',
        'dos=1000+wormhole',
        # 'dos=100+wormhole',
        'ddos=1000+wormhole',
        # 'ddos=100+wormhole',

        # ---- stealth + stealth ----
        'blackhole+wormhole',

        # ---- replay + X (both replay variants for each combo) ----
        # 'blackhole+replay=100,100,20,3,rand',
        'blackhole+replay=50,200,10,5,inc',
        # 'ddos=100+replay=100,100,20,3,rand',
        # 'ddos=100+replay=50,200,10,5,inc',
        # 'ddos=1000+replay=100,100,20,3,rand',
        'ddos=1000+replay=50,200,10,5,inc',
        # 'dos=100+replay=100,100,20,3,rand',
        # 'dos=100+replay=50,200,10,5,inc',
        # 'dos=1000+replay=100,100,20,3,rand',
        'dos=1000+replay=50,200,10,5,inc',
        # 'replay=100,100,20,3,rand+wormhole',
        'replay=50,200,10,5,inc+wormhole',
    ],
    'num_drones': [5, 10, 15, 20],
    'num_basestations': [2],
    'payload': ['image'],

    # -- physical layer --
    'pathloss': ['logdist', '3gpp'],
    # adaptive = real-radio rate adaptation; bpsk12 = robust extreme;
    # qam64_34 = high-rate extreme. Three points span the SNR sensitivity
    # curve. Add intermediate schemes if you want a finer sweep.
    'modulation': ['adaptive'],
    'missions': ['spiral', 'grid', 'hover_transit', 'random'],     # or 'spiral', 'grid', ..., or 'spiral,grid'
    'tx_power_dBm': [10, 30],      # add more values for a power scan
    'noise_pos_dBm': [95],     # positive int -> -95 dBm
}

# -----------------------------------------------------------------------


def build_lines(axes):
    keys = ['attacks', 'num_drones', 'num_basestations', 'payload',
            'pathloss', 'modulation', 'missions', 'tx_power_dBm', 'noise_pos_dBm']
    for k in keys:
        if k not in axes or not axes[k]:
            raise ValueError(f"Missing/empty axis: {k!r}")
    lines = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        lines.append('-'.join(str(x) for x in combo))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', default='configurations.txt',
                    help='Output path (default: configurations.txt in cwd)')
    ap.add_argument('--validate', action='store_true',
                    help='Parse each generated line through simlib.config_parser '
                         'to confirm it is well-formed.')
    ap.add_argument('--dry-run', action='store_true',
                    help="Don't write; print summary only.")
    ap.add_argument('--max-lines', type=int, default=1000000,
                    help='Refuse to write if line count exceeds this (default '
                         '10000). Guards against accidental axis explosion. '
                         'Use --max-lines 0 to disable.')
    args = ap.parse_args()

    lines = build_lines(AXES)
    n = len(lines)
    print(f"Generating {n} configuration lines")
    print(f"Axes: " + ", ".join(f"{k}={len(AXES[k])}" for k in AXES))

    if args.max_lines > 0 and n > args.max_lines:
        print(f"\nERROR: {n} lines exceeds --max-lines={args.max_lines}.")
        print("If this is intentional, re-run with --max-lines <larger value>")
        print("or --max-lines 0 to disable the check.")
        sys.exit(3)

    if args.validate:
        # Import from sibling Network-simulator/simlib/ if present
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, 'simlib'))
        sys.path.insert(0, os.path.join(here, 'Network-simulator', 'simlib'))
        try:
            from config_parser import parse_line
        except ImportError as e:
            print(f"Validation requested but cannot import config_parser: {e}")
            sys.exit(2)
        bad = 0
        for i, ln in enumerate(lines):
            try:
                parse_line(ln)
            except Exception as e:
                print(f"  line {i}: {ln!r}  -> {e}")
                bad += 1
        if bad:
            print(f"VALIDATION FAILED: {bad} bad lines.")
            sys.exit(1)
        print(f"Validation OK: all {n} lines parse.")

    if args.dry_run:
        print("First 5 lines:")
        for ln in lines[:5]:
            print(f"  {ln}")
        print("Last 3 lines:")
        for ln in lines[-3:]:
            print(f"  {ln}")
        return

    with open(args.output, 'w') as f:
        for ln in lines:
            f.write(ln + '\n')
    print(f"Wrote {n} lines to {args.output}")


if __name__ == '__main__':
    main()