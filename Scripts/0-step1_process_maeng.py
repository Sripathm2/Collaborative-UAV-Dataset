"""
step1_process_maeng.py — Process Maeng et al. LTE I/Q dataset.

Reads raw IQ samples (cf64_le) and GPS logs (rf64_le) from the five
altitude archives (30, 50, 70, 90, 110m), computes received power,
interpolates GPS positions, and computes distance to the base station.

Usage:
    # First extract all archives:
    cd ~/datasets/maeng_rsrp
    mkdir -p extracted/{30m,50m,70m,90m,110m}
    7z x NRDZ_30m_dataset.7z -o./extracted/30m
    7z x NRDZ_50m_dataset.7z -o./extracted/50m
    7z x NRDZ_70m_dataset.7z -o./extracted/70m
    7z x NRDZ_90m_dataset.7z -o./extracted/90m
    7z x NRDZ_110m_dataset.7z -o./extracted/110m

    # Then run:
    python 0-step1_process_maeng.py --maeng_dir ../Datasets/Finetuning-raw/maeng_rsrp/extracted

Outputs:
    processed/maeng_rsrp.csv — one row per IQ measurement with:
        timestamp, longitude, latitude, altitude_m, rx_power_dBm,
        distance_to_bs_m, altitude_target_m
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ============================================================
# AERPAW Lake Wheeler BS tower location
# (from AFAR main.m and AERPAW documentation)
# ============================================================
BS_LAT = 35.727451
BS_LON = -78.695974
BS_ALT = 12.0  # approximate tower height in meters

DEG_TO_M = 1.113195e5  # degrees to meters at this latitude

ALTITUDE_DIRS = {
    '30m': 30,
    '50m': 50,
    '70m': 70,
    '90m': 90,
    '110m': 110,
}


# ============================================================
# Read GPS binary (rf64_le format)
# ============================================================

def read_gps_binary(gps_data_path, gps_meta_path):
    """
    Read GPS binary file. Format: rf64_le (real float64, little endian).
    Annotations define sections: longitude, latitude, altitude, timestamp.

    Returns DataFrame with columns: longitude, latitude, altitude_m, timestamp
    """
    # Read metadata to get section offsets and lengths
    with open(gps_meta_path, 'r') as f:
        meta = json.load(f)

    annotations = meta.get('annotations', [])

    # Build section map from annotations
    sections = {}
    for ann in annotations:
        comment = ann.get('core:comment', '')
        start = ann.get('core:sample_start', 0)
        count = ann.get('core:sample_count', 0)
        sections[comment] = (start, count)

    # Read the full binary file as float64
    data = np.fromfile(str(gps_data_path), dtype=np.float64)

    result = {}
    for name, (start, count) in sections.items():
        result[name] = data[start:start + count]

    df = pd.DataFrame({
        'longitude': result.get('longitude', []),
        'latitude': result.get('latitude', []),
        'altitude_m': result.get('altitude', []),
        'timestamp': result.get('timestamp', []),
    })

    return df


# ============================================================
# Extract timestamp from IQ filename
# ============================================================

def parse_iq_timestamp(filename):
    """
    Extract Unix timestamp from IQ filename.
    Format: results_2022_03_11_10_36_26_993.sigmf-data
    → 2022-03-11 10:36:26.993
    """
    # Extract the timestamp part
    match = re.search(
        r'results_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})',
        filename
    )
    if not match:
        return None

    year, month, day, hour, minute, sec, ms = match.groups()

    from datetime import datetime
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo('America/New_York')  # auto-handles EST vs EDT
    dt = datetime(int(year), int(month), int(day),
                  int(hour), int(minute), int(sec),
                  int(ms) * 1000,
                  tzinfo=eastern)
    return dt.timestamp()


# ============================================================
# Compute received power from IQ samples
# ============================================================

def compute_power_from_iq(data_path, n_iq_samples=40800):
    """
    Read cf64_le IQ samples and compute received power in dBm.

    cf64_le = complex128 (each sample = 2x float64 = 16 bytes)
    The file has: IQ samples (40800 complex) + timestamp (20 complex)

    Returns power in dB (relative, not absolute dBm without calibration)
    """
    try:
        # Read as complex128
        samples = np.fromfile(str(data_path), dtype=np.complex128)

        # Take only the IQ portion (first n_iq_samples)
        iq = samples[:n_iq_samples]

        if len(iq) == 0:
            return None

        # Received power = mean(|s|^2)
        power_linear = np.mean(np.abs(iq) ** 2)

        # Convert to dB
        power_dB = 10 * np.log10(power_linear + 1e-30)

        return float(power_dB)

    except Exception as e:
        return None


# ============================================================
# Process one altitude directory
# ============================================================

def process_altitude(alt_dir, target_altitude):
    """
    Process all IQ files and GPS for one altitude.

    alt_dir: Path to extracted altitude directory (e.g., extracted/30m/)
    target_altitude: nominal altitude in meters (30, 50, 70, 90, 110)
    """
    gps_dir = alt_dir / 'GPS_logs'
    iq_dir = alt_dir / 'IQ_samples'

    if not gps_dir.exists() or not iq_dir.exists():
        print(f"  Missing GPS_logs or IQ_samples in {alt_dir}")
        return None

    # --- Read GPS ---
    gps_data_files = sorted(gps_dir.glob('*vehicleOut.sigmf-data'))
    gps_meta_files = sorted(gps_dir.glob('*vehicleOut.sigmf-meta'))

    if not gps_data_files:
        print(f"  No GPS data found in {gps_dir}")
        return None

    gps_df = read_gps_binary(gps_data_files[0], gps_meta_files[0])
    print(f"  GPS: {len(gps_df)} points, "
          f"alt={gps_df['altitude_m'].mean():.1f}m (target: {target_altitude}m), "
          f"duration={gps_df['timestamp'].max() - gps_df['timestamp'].min():.0f}s")

    # --- Find all IQ files ---
    iq_data_files = sorted(iq_dir.glob('results_*.sigmf-data'))
    iq_meta_files = sorted(iq_dir.glob('results_*.sigmf-meta'))
    print(f"  IQ files: {len(iq_data_files)}")

    if not iq_data_files:
        return None

    # Read one meta to get IQ sample count
    n_iq_samples = 40800  # default
    if iq_meta_files:
        with open(iq_meta_files[0], 'r') as f:
            meta = json.load(f)
        for ann in meta.get('annotations', []):
            if ann.get('core:comment') == 'IQ':
                n_iq_samples = ann['core:sample_count']
                break

    # --- Process each IQ file ---
    rows = []
    n_success = 0
    n_fail = 0

    # Process in batches for memory efficiency
    batch_size = 1000
    for batch_start in range(0, len(iq_data_files), batch_size):
        batch_files = iq_data_files[batch_start:batch_start + batch_size]

        for data_file in batch_files:
            # Extract timestamp from filename
            ts = parse_iq_timestamp(data_file.name)
            if ts is None:
                n_fail += 1
                continue

            # Compute power
            power_dB = compute_power_from_iq(data_file, n_iq_samples)
            if power_dB is None:
                n_fail += 1
                continue

            rows.append({
                'timestamp': ts,
                'rx_power_dB': power_dB,
            })
            n_success += 1

        # Progress
        done = min(batch_start + batch_size, len(iq_data_files))
        print(f"    Processed {done}/{len(iq_data_files)} "
              f"({n_success} ok, {n_fail} failed)", end='\r')

    print(f"    Processed {len(iq_data_files)}/{len(iq_data_files)} "
          f"({n_success} ok, {n_fail} failed)")

    if not rows:
        return None

    iq_df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)

    # --- Interpolate GPS onto IQ timestamps ---
    gps_sorted = gps_df.sort_values('timestamp')
    ts_gps = gps_sorted['timestamp'].values

    # Only keep IQ measurements within GPS time range
    gps_start = ts_gps[0]
    gps_end = ts_gps[-1]
    iq_df = iq_df[(iq_df['timestamp'] >= gps_start) &
                  (iq_df['timestamp'] <= gps_end)].copy()

    if len(iq_df) == 0:
        print(f"  No IQ measurements within GPS time range")
        return None

    interp_lon = interp1d(ts_gps, gps_sorted['longitude'].values,
                          kind='linear', fill_value='extrapolate')
    interp_lat = interp1d(ts_gps, gps_sorted['latitude'].values,
                          kind='linear', fill_value='extrapolate')
    interp_alt = interp1d(ts_gps, gps_sorted['altitude_m'].values,
                          kind='linear', fill_value='extrapolate')

    ts_iq = iq_df['timestamp'].values
    iq_df['longitude'] = interp_lon(ts_iq)
    iq_df['latitude'] = interp_lat(ts_iq)
    iq_df['altitude_m'] = interp_alt(ts_iq)

    # --- Compute distance to BS ---
    dx = (iq_df['longitude'] - BS_LON) * DEG_TO_M
    dy = (iq_df['latitude'] - BS_LAT) * DEG_TO_M
    dz = iq_df['altitude_m'] - BS_ALT
    iq_df['horizontal_distance_m'] = np.sqrt(dx**2 + dy**2)
    iq_df['distance_to_bs_m'] = np.sqrt(dx**2 + dy**2 + dz**2)

    # Target altitude (nominal)
    iq_df['altitude_target_m'] = target_altitude

    print(f"  Result: {len(iq_df)} measurements, "
          f"distance range: {iq_df['distance_to_bs_m'].min():.0f}–"
          f"{iq_df['distance_to_bs_m'].max():.0f} m, "
          f"power range: {iq_df['rx_power_dB'].min():.1f} to "
          f"{iq_df['rx_power_dB'].max():.1f} dB")

    return iq_df


# ============================================================
# Plotting
# ============================================================

def plot_results(df, output_dir='../results/Step_0/maeng'):
    """Generate summary plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Power vs distance per altitude
    fig, ax = plt.subplots(figsize=(10, 6))
    for alt, group in df.groupby('altitude_target_m'):
        ax.scatter(group['distance_to_bs_m'], group['rsrp_dBm'],
                   s=3, alpha=0.2, label=f'{alt:.0f} m')
    ax.set_xlabel('Distance to BS (m)')
    ax.set_ylabel('Received Power (dB)')
    ax.set_title('Maeng et al.: Power vs Distance by Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/maeng_power_vs_distance.png', dpi=150)
    plt.close()

    # Power vs distance with log-distance x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    for alt, group in df.groupby('altitude_target_m'):
        ax.scatter(group['distance_to_bs_m'], group['rsrp_dBm'],
                   s=3, alpha=0.2, label=f'{alt:.0f} m')
    ax.set_xscale('log')
    ax.set_xlabel('Distance to BS (m, log scale)')
    ax.set_ylabel('Received Power (dB)')
    ax.set_title('Maeng et al.: Power vs log(Distance) — should be linear')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/maeng_power_vs_log_distance.png', dpi=150)
    plt.close()

    # Altitude distribution per target
    fig, ax = plt.subplots(figsize=(10, 5))
    for alt in sorted(df['altitude_target_m'].unique()):
        group = df[df['altitude_target_m'] == alt]
        ax.hist(group['altitude_m'], bins=50, alpha=0.5,
                label=f'Target {alt:.0f}m (mean={group["altitude_m"].mean():.1f})')
    ax.set_xlabel('Measured Altitude (m)')
    ax.set_ylabel('Count')
    ax.set_title('Altitude Distribution per Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/maeng_altitude_dist.png', dpi=150)
    plt.close()

    print(f"  Plots saved to {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process Maeng et al. LTE I/Q dataset')
    parser.add_argument('--maeng_dir', type=str, required=True,
                        help='Path to extracted directory (containing 30m/, 50m/, etc.)')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Maeng et al. RSRP Dataset Processing")
    print("=" * 60)

    base = Path(args.maeng_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        print("Extract archives first:")
        for alt in ALTITUDE_DIRS:
            print(f"  7z x NRDZ_{alt}_dataset.7z -o./extracted/{alt}")
        return

    all_frames = []

    for dir_name, target_alt in sorted(ALTITUDE_DIRS.items(),
                                        key=lambda x: x[1]):
        alt_dir = base / dir_name
        if not alt_dir.exists():
            print(f"\n{dir_name}: directory not found, skipping")
            continue

        print(f"\n{'='*40}")
        print(f"Processing {dir_name} (target: {target_alt}m)")
        print(f"{'='*40}")

        df = process_altitude(alt_dir, target_alt)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        print("\nERROR: No data processed")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # Save
    out_dir = Path('../Datasets/Finetuning-processed/')
    out_dir.mkdir(exist_ok=True)

    # Rename rx_power_dB to match what step2 expects
    combined.rename(columns={'rx_power_dB': 'rsrp_dBm'}, inplace=True)
    # Note: this is total received power, not true RSRP. But it's proportional
    # and the path loss exponent n(h) will be the same.

    # Also add distance_m alias for step2 compatibility
    combined['distance_m'] = combined['distance_to_bs_m']
    combined['altitude_m_nominal'] = combined['altitude_target_m']

    combined.to_csv(out_dir / 'maeng_rsrp.csv', index=False)
    print(f"\nSaved: {out_dir / 'maeng_rsrp.csv'} ({len(combined)} rows)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total measurements: {len(combined)}")

    print(f"\nPer-altitude breakdown:")
    summary = combined.groupby('altitude_target_m').agg(
        n=('rsrp_dBm', 'count'),
        mean_alt=('altitude_m', 'mean'),
        mean_power=('rsrp_dBm', 'mean'),
        std_power=('rsrp_dBm', 'std'),
        mean_dist=('distance_to_bs_m', 'mean'),
        dist_range=('distance_to_bs_m',
                     lambda x: f"{x.min():.0f}–{x.max():.0f}"),
    )
    print(summary.to_string())

    print(f"\nBS location: lat={BS_LAT}, lon={BS_LON}")
    print(f"Frequency: 3.51 GHz")

    # Plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        plot_results(combined)

    print(f"\n{'='*60}")
    print("DONE — This is the PRIMARY dataset for Layer 1 fitting")
    print(f"{'='*60}")
    print("• Column 'rsrp_dBm' is total received power (proportional to RSRP)")
    print("• Column 'distance_m' = 3D distance to BS")
    print("• Column 'altitude_target_m' = nominal flight altitude")
    print("• Feed this into step2_layer1_pathloss.py")
    print("• Note: power is relative (not calibrated to absolute dBm)")
    print("  → n(h) and sigma(h) will be correct; PL0 will need offset calibration")


if __name__ == '__main__':
    main()