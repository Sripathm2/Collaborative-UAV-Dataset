"""
step1_process_aadm.py — Process AADM dataset (USRP component).

Extracts UAV GPS traces + power/SNR measurements from testbed and
development flights. Merges with base station power logs.

Usage:
    python 0-step1_process_aadm.py --aadm_dir ../Datasets/Finetuning/aadm/AADM2025Dryad/USRP

    
Outputs:
    processed/aadm_testbed.csv    — all testbed flights merged
    processed/aadm_development.csv — all DT flights merged
    processed/aadm_summary.csv    — per-flight stats
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime, timezone


# ============================================================
# Base station locations at AERPAW Lake Wheeler
# ============================================================
BS_LOCATIONS = {
    'LW1': {'lat': 35.727451, 'lon': -78.695974},
    'LW2': {'lat': 35.729118, 'lon': -78.699181},
    'LW3': {'lat': 35.729851, 'lon': -78.697110},
    'LW4': {'lat': 35.728067, 'lon': -78.697304},
}

DEG_TO_M = 1.113195e5


# ============================================================
# Parsers
# ============================================================

def parse_vehicleout(filepath):
    """
    Parse vehicleOut.txt GPS log.
    Format: index,lon,lat,alt,"(orientation)","(velocity)",value,datetime,int,int
    Some lines may have 11+ fields due to tuple parsing.
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract fields carefully — tuples in quotes complicate CSV parsing
            # Strategy: find the datetime pattern, then parse around it
            dt_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
            if not dt_match:
                continue

            dt_str = dt_match.group(1)

            # Parse the prefix (before the first quoted tuple)
            prefix = line.split('"')[0].rstrip(',')
            parts = prefix.split(',')

            if len(parts) < 4:
                continue

            try:
                lon = float(parts[1])
                lat = float(parts[2])
                alt = float(parts[3])

                # Parse datetime to unix timestamp
                dt = datetime.strptime(dt_str[:26], '%Y-%m-%d %H:%M:%S.%f')
                ts = dt.replace(tzinfo=timezone.utc).timestamp()

                rows.append({
                    'timestamp': ts,
                    'longitude': lon,
                    'latitude': lat,
                    'altitude_m': alt,
                    'datetime_str': dt_str,
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)


def parse_power_log(filepath):
    """
    Parse power_log.txt or snr_log.txt.
    Format: [2025-09-10 10:58:57.725711] 0000000       -788.5956
    Lines with '*' are skipped.
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '*' in line:
                continue

            # Extract timestamp and value
            match = re.match(
                r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s+\d+\s+([-\d.]+)',
                line
            )
            if not match:
                continue

            dt_str = match.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                continue

            # Parse timestamp
            dt = datetime.strptime(dt_str[:26], '%Y-%m-%d %H:%M:%S.%f')
            ts = dt.replace(tzinfo=timezone.utc).timestamp()

            rows.append({'timestamp': ts, 'value': value})

    return pd.DataFrame(rows)


# ============================================================
# Process a single flight
# ============================================================

def process_flight(flight_dir, flight_id):
    """
    Process one flight directory containing UAV/ and LW1-4/ subfolders.
    UAV folder has GPS (vehicleOut.txt).
    LW1-LW4 folders have power_log and snr_log (BS receives UAV signal).
    Returns a merged DataFrame.
    """
    flight_path = Path(flight_dir)
    uav_dir = flight_path / 'UAV'

    if not uav_dir.exists():
        return None

    # --- Find and load UAV GPS ---
    gps_files = list(uav_dir.glob('*vehicleOut*'))
    if not gps_files:
        return None

    gps_df = parse_vehicleout(gps_files[0])
    if gps_df.empty or len(gps_df) < 2:
        return None

    gps_sorted = gps_df.sort_values('timestamp')
    ts_gps = gps_sorted['timestamp'].values
    gps_start = ts_gps[0]
    gps_end = ts_gps[-1]

    # Build GPS interpolators
    interp_lon = interp1d(ts_gps, gps_sorted['longitude'].values,
                          kind='linear', fill_value='extrapolate')
    interp_lat = interp1d(ts_gps, gps_sorted['latitude'].values,
                          kind='linear', fill_value='extrapolate')
    interp_alt = interp1d(ts_gps, gps_sorted['altitude_m'].values,
                          kind='linear', fill_value='extrapolate')

    # --- Load power/SNR from each base station ---
    all_bs_frames = []

    for bs_name in ['LW1', 'LW2', 'LW3', 'LW4']:
        bs_dir = flight_path / bs_name
        if not bs_dir.exists():
            continue

        power_files = list(bs_dir.glob('*power_log*'))
        snr_files = list(bs_dir.glob('*snr_log*'))

        if not power_files:
            continue

        # Load power
        power_df = parse_power_log(power_files[0])
        if power_df.empty:
            continue
        power_df.rename(columns={'value': 'power_dBm'}, inplace=True)
        power_df = power_df[power_df['power_dBm'] > -200].copy()

        if power_df.empty:
            continue

        # Load SNR if available
        if snr_files:
            snr_df = parse_power_log(snr_files[0])
            if not snr_df.empty:
                snr_df.rename(columns={'value': 'snr_dB'}, inplace=True)
                snr_df = snr_df[snr_df['snr_dB'] > -200].copy()
                power_df = pd.merge_asof(
                    power_df.sort_values('timestamp'),
                    snr_df.sort_values('timestamp'),
                    on='timestamp', tolerance=0.1, direction='nearest'
                )

        # Filter to GPS time range
        power_df = power_df[
            (power_df['timestamp'] >= gps_start) &
            (power_df['timestamp'] <= gps_end)
        ].copy()

        if power_df.empty:
            continue

        # Interpolate UAV GPS position at BS measurement timestamps
        ts = power_df['timestamp'].values
        power_df['longitude'] = interp_lon(ts)
        power_df['latitude'] = interp_lat(ts)
        power_df['altitude_m'] = interp_alt(ts)

        # Distance from UAV to this BS
        bs_loc = BS_LOCATIONS[bs_name]
        dx = (power_df['longitude'] - bs_loc['lon']) * DEG_TO_M
        dy = (power_df['latitude'] - bs_loc['lat']) * DEG_TO_M
        power_df['distance_to_bs_m'] = np.sqrt(dx**2 + dy**2)
        power_df['bs_id'] = bs_name

        all_bs_frames.append(power_df)

    if not all_bs_frames:
        return None

    merged = pd.concat(all_bs_frames, ignore_index=True)
    merged['time_rel_s'] = merged['timestamp'] - merged['timestamp'].min()
    merged['flight_id'] = flight_id

    return merged


# ============================================================
# Process all flights in a stage directory
# ============================================================

def process_stage(stage_dir, stage_name):
    """Process all flights in a testbed or development directory."""
    stage_path = Path(stage_dir)
    if not stage_path.exists():
        print(f"  {stage_name}: directory not found")
        return None

    flight_dirs = sorted([d for d in stage_path.iterdir() if d.is_dir()])
    print(f"  {stage_name}: found {len(flight_dirs)} flight directories")

    all_frames = []
    for flight_dir in flight_dirs:
        flight_id = f"{stage_name}_{flight_dir.name}"
        result = process_flight(flight_dir, flight_id)

        if result is not None and len(result) > 0:
            result['stage'] = stage_name
            all_frames.append(result)
            print(f"    {flight_dir.name}: {len(result)} points, "
                  f"power={result['power_dBm'].mean():.1f} dBm")
        else:
            print(f"    {flight_dir.name}: no valid data")

    if all_frames:
        return pd.concat(all_frames, ignore_index=True)
    return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process AADM USRP dataset')
    parser.add_argument('--aadm_dir', type=str, required=True,
                        help='Path to AADM2025Dryad/USRP directory')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("AADM Dataset Processing (USRP)")
    print("=" * 60)

    base = Path(args.aadm_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        return

    # Find stage directories
    stage_dirs = {}
    for d in base.iterdir():
        if d.is_dir():
            name_lower = d.name.lower()
            if 'testbed' in name_lower and '33' in name_lower:
                stage_dirs['testbed'] = d
            elif 'development' in name_lower:
                stage_dirs['development'] = d
            elif 'testbed' in name_lower:
                # Additional testbed batches
                stage_dirs[f'testbed_{d.name[:20]}'] = d

    print(f"Found stages: {list(stage_dirs.keys())}")

    all_frames = []
    summaries = []

    for stage_name, stage_dir in sorted(stage_dirs.items()):
        print(f"\n{'='*40}")
        print(f"Processing: {stage_name}")
        print(f"{'='*40}")

        df = process_stage(stage_dir, stage_name)
        if df is not None:
            all_frames.append(df)

            # Per-flight summary
            for fid, group in df.groupby('flight_id'):
                summaries.append({
                    'flight_id': fid,
                    'stage': stage_name,
                    'n_measurements': len(group),
                    'duration_s': group['time_rel_s'].max(),
                    'mean_power': group['power_dBm'].mean(),
                    'mean_dist_to_bs': group['distance_to_bs_m'].mean(),
                    'mean_altitude': group['altitude_m'].mean(),
                })

    if not all_frames:
        print("\nERROR: No data processed")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Save
    out_dir = Path('../Datasets/Finetuning-processed/')
    out_dir.mkdir(exist_ok=True)

    # Split by stage type
    testbed_mask = combined['stage'].str.contains('testbed')
    dev_mask = combined['stage'].str.contains('development')

    testbed = combined[testbed_mask]
    development = combined[dev_mask]

    if not testbed.empty:
        testbed.to_csv(out_dir / 'aadm_testbed.csv', index=False)
        print(f"\nSaved: {out_dir / 'aadm_testbed.csv'} ({len(testbed)} rows)")

    if not development.empty:
        development.to_csv(out_dir / 'aadm_development.csv', index=False)
        print(f"\nSaved: {out_dir / 'aadm_development.csv'} ({len(development)} rows)")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / 'aadm_summary.csv', index=False)
    print(f"Saved: {out_dir / 'aadm_summary.csv'}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total measurements: {len(combined)}")
    print(f"  Testbed: {len(testbed)}")
    print(f"  Development: {len(development)}")
    print(f"  Flights: {combined['flight_id'].nunique()}")

    if not testbed.empty:
        print(f"\nTestbed stats:")
        print(f"  Power range: {testbed['power_dBm'].min():.1f} to {testbed['power_dBm'].max():.1f} dBm")
        print(f"  Altitude range: {testbed['altitude_m'].min():.1f} to {testbed['altitude_m'].max():.1f} m")
        if 'snr_dB' in testbed.columns:
            valid_snr = testbed['snr_dB'].dropna()
            if len(valid_snr) > 0:
                print(f"  SNR range: {valid_snr.min():.1f} to {valid_snr.max():.1f} dB")

    print(f"\n{'='*60}")
    print("DONE — This dataset provides:")
    print(f"{'='*60}")
    print("• Layer 2: UAV mobility traces (GPS from 33+ testbed flights)")
    print("• Layer 3: Power/SNR vs distance for 4 base stations")
    print("• Layer 4: Testbed vs Development (DT) comparison")


if __name__ == '__main__':
    main()