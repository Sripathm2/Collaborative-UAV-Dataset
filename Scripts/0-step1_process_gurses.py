"""
step1_process_gurses.py — Process Gürses–Sichitiu channel sounding dataset.

The .sigmf-meta files are JSON with all metadata we need (position, distance,
altitude, velocity, gains). The .sigmf-data files are f32_le binary IQ samples
from which we compute received power.

Usage:
    python 0-step1_process_gurses.py --gurses_dir ../Datasets/Finetuning-raw/gurses_channel

Outputs:
    processed/gurses_channel.csv — one row per measurement with:
        timestamp, rx_lat, rx_lon, rx_altitude_m, tx_lat, tx_lon, tx_altitude_m,
        distance_m, speed_m_s, vx, vy, vz, pitch, yaw, roll,
        rx_power_dBm, flight_stage, folder_name
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Gürses dataset metadata from the paper / README
# ============================================================

# From the .sigmf-meta: core:frequency is 3.564 GHz (CBRS band)
# The paper says 3.3 GHz — the actual carrier is in the meta files
# Altitudes from paper: 40, 70, 100 m — we'll read actual from GPS

# Mapping of folder timestamps to known experiment parameters
# (from the Gürses VTC2024-Fall paper and AERPAW dataset page)
# The paper mentions: 3 altitudes (40, 70, 100m), 3 bandwidths at 40m
# 9 total measurement sessions → 9 folders
FOLDER_COUNT = 9  # expected


def process_single_measurement(meta_path, data_path):
    """
    Process a single .sigmf-meta / .sigmf-data pair.

    Returns a dict with all extracted fields, or None if invalid.
    """
    # Load metadata (JSON)
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    glob = meta.get('global', {})
    captures = meta.get('captures', [])

    if not captures:
        return None

    cap = captures[0]  # each file has one capture

    # Extract metadata fields
    rx_loc = cap.get('core:rx_location', {})
    tx_loc = cap.get('core:tx_location', {})
    rotation = cap.get('core:rotation', {})
    velocity = cap.get('core:velocity', {})

    row = {
        'timestamp': cap.get('core:timestamp'),
        'rx_lat': rx_loc.get('latitude'),
        'rx_lon': rx_loc.get('longitude'),
        'rx_altitude_m': rx_loc.get('altitude'),
        'tx_lat': tx_loc.get('latitude'),
        'tx_lon': tx_loc.get('longitude'),
        'tx_altitude_m': tx_loc.get('altitude'),
        'distance_m': cap.get('core:dist'),
        'speed_m_s': cap.get('core:speed'),
        'vx': velocity.get('velocity_x'),
        'vy': velocity.get('velocity_y'),
        'vz': velocity.get('velocity_z'),
        'pitch': rotation.get('pitch'),
        'yaw': rotation.get('yaw'),
        'roll': rotation.get('roll'),
        'flight_stage': cap.get('core:flight_stage'),
        'frequency_Hz': cap.get('core:frequency'),
        'sample_rate': glob.get('core:sample_rate'),
        'tx_gain_dBm': glob.get('core:tx_gain_ref'),
        'rx_gain_dBm': glob.get('core:rx_gain_ref'),
        'flight_time_s': cap.get('core:time'),
    }

    # Compute received power from IQ data
    if data_path and data_path.exists():
        try:
            # f32_le = 32-bit float, little endian
            # Channel sounder IQ: interleaved I, Q samples
            samples = np.fromfile(str(data_path), dtype=np.float32)

            if len(samples) >= 2:
                # Interpret as complex: I + jQ
                I = samples[0::2]
                Q = samples[1::2]
                complex_samples = I + 1j * Q

                # Received power = mean(|s|^2), in linear
                power_linear = np.mean(np.abs(complex_samples) ** 2)

                # Convert to dBm, accounting for gains
                # P_rx = 10*log10(power_linear) + tx_gain - rx_gain_ref
                # rx_gain_ref is negative (attenuation), so subtracting it adds
                power_dBm = 10 * np.log10(power_linear + 1e-30)

                # Adjust with gain references if available
                tx_gain = row['tx_gain_dBm'] or 0
                rx_gain = row['rx_gain_dBm'] or 0
                # The raw power + gain calibration gives received signal power
                row['rx_power_raw_dB'] = float(power_dBm)
                row['rx_power_dBm'] = float(power_dBm + tx_gain + rx_gain)
                row['n_samples'] = len(complex_samples)
        except Exception as e:
            row['rx_power_dBm'] = None
            row['rx_power_raw_dB'] = None
            row['n_samples'] = 0
    else:
        row['rx_power_dBm'] = None
        row['rx_power_raw_dB'] = None
        row['n_samples'] = 0

    return row


def process_all_folders(gurses_dir):
    """
    Walk all timestamped folders in the Gürses dataset.
    """
    base = Path(gurses_dir)
    all_rows = []

    # Find all timestamped folders (format: 2023-12-15_HH_MM)
    folders = sorted([d for d in base.iterdir()
                      if d.is_dir() and d.name.startswith('2023')])

    print(f"Found {len(folders)} measurement folders")

    for folder in folders:
        meta_files = sorted(folder.glob('*.sigmf-meta'))
        print(f"\n  {folder.name}: {len(meta_files)} measurements")

        folder_rows = 0
        for meta_path in meta_files:
            # Corresponding data file
            data_path = meta_path.with_suffix('.sigmf-data')

            row = process_single_measurement(meta_path, data_path)
            if row is None:
                continue

            row['folder_name'] = folder.name
            all_rows.append(row)
            folder_rows += 1

        if folder_rows > 0:
            # Quick stats for this folder
            folder_df = pd.DataFrame([r for r in all_rows if r['folder_name'] == folder.name])
            alt = folder_df['rx_altitude_m']
            dist = folder_df['distance_m']
            print(f"    Altitude range: {alt.min():.1f}–{alt.max():.1f} m "
                  f"(mean {alt.mean():.1f})")
            print(f"    Distance range: {dist.min():.0f}–{dist.max():.0f} m")
            if folder_df['rx_power_dBm'].notna().any():
                pwr = folder_df['rx_power_dBm'].dropna()
                print(f"    Power range: {pwr.min():.1f} to {pwr.max():.1f} dBm")
            print(f"    Flight stages: "
                  f"{folder_df['flight_stage'].value_counts().to_dict()}")

    return pd.DataFrame(all_rows)


# ============================================================
# Post-processing
# ============================================================

def add_derived_columns(df):
    """Add useful derived columns."""
    # Altitude bins (round to nearest known altitude)
    known_alts = [40, 70, 100]
    df['altitude_bin_m'] = df['rx_altitude_m'].apply(
        lambda h: min(known_alts, key=lambda a: abs(a - h))
        if pd.notna(h) else None
    )

    # Filter to flight stage only (exclude takeoff/landing)
    df['is_flight'] = df['flight_stage'] == 'Flight'

    # Horizontal distance (excluding altitude difference)
    # The core:dist in metadata might be 3D — let's also compute 2D
    if df['rx_lat'].notna().any() and df['tx_lat'].notna().any():
        DEG_TO_M = 1.113195e5
        dx = (df['rx_lon'] - df['tx_lon']) * DEG_TO_M
        dy = (df['rx_lat'] - df['tx_lat']) * DEG_TO_M
        df['horizontal_distance_m'] = np.sqrt(dx**2 + dy**2)

    return df


# ============================================================
# Plotting
# ============================================================

def plot_results(df, output_dir='../results/Step_0/gurses'):
    """Generate summary plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    flight_df = df[df['is_flight']].copy()

    if flight_df.empty:
        print("  No flight-stage data to plot")
        return

    # Power vs distance per altitude bin
    fig, ax = plt.subplots(figsize=(10, 6))
    for alt_bin, group in flight_df.groupby('altitude_bin_m'):
        if group['rx_power_dBm'].notna().any():
            ax.scatter(group['distance_m'], group['rx_power_dBm'],
                       s=8, alpha=0.4, label=f'{alt_bin:.0f} m')
    ax.set_xlabel('Distance to transmitter (m)')
    ax.set_ylabel('Received Power (dBm)')
    ax.set_title('Gürses–Sichitiu: Power vs Distance by Altitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gurses_power_vs_distance.png', dpi=150)
    plt.close()

    # Altitude distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(flight_df['rx_altitude_m'].dropna(), bins=50, alpha=0.7)
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Count')
    ax.set_title('Flight Altitude Distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gurses_altitude_dist.png', dpi=150)
    plt.close()

    # 3D scatter: position colored by power
    if flight_df['rx_power_dBm'].notna().any():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        valid = flight_df.dropna(subset=['rx_power_dBm'])
        sc = ax.scatter(valid['rx_lon'], valid['rx_lat'],
                        valid['rx_altitude_m'],
                        c=valid['rx_power_dBm'], s=5, cmap='jet', alpha=0.6)
        # Mark TX location
        tx_lat = valid['tx_lat'].iloc[0]
        tx_lon = valid['tx_lon'].iloc[0]
        tx_alt = valid['tx_altitude_m'].iloc[0]
        ax.scatter([tx_lon], [tx_lat], [tx_alt], c='black', s=100, marker='^',
                   label='Transmitter')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('3D Flight Path (color = power)')
        plt.colorbar(sc, label='Power (dBm)', shrink=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gurses_3d_scatter.png', dpi=150)
        plt.close()

    print(f"  Plots saved to {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process Gürses–Sichitiu channel sounding dataset')
    parser.add_argument('--gurses_dir', type=str, required=True,
                        help='Path to gurses_channel directory')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Gürses–Sichitiu Channel Sounding Dataset Processing")
    print("=" * 60)

    if not Path(args.gurses_dir).exists():
        print(f"ERROR: {args.gurses_dir} not found")
        return

    # Process all measurements
    df = process_all_folders(args.gurses_dir)

    if df.empty:
        print("ERROR: No data extracted")
        return

    # Add derived columns
    df = add_derived_columns(df)

    # Save
    out_dir = Path('../Datasets/Finetuning-processed/')
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'gurses_channel.csv', index=False)
    print(f"\nSaved: {out_dir / 'gurses_channel.csv'} ({len(df)} rows)")

    # Flight-only subset
    flight_df = df[df['is_flight']]
    flight_df.to_csv(out_dir / 'gurses_channel_flight.csv', index=False)
    print(f"Saved: {out_dir / 'gurses_channel_flight.csv'} ({len(flight_df)} rows)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total measurements: {len(df)}")
    print(f"Flight-stage only: {len(flight_df)}")
    print(f"Folders: {df['folder_name'].nunique()}")

    print(f"\nPer-altitude-bin breakdown (flight stage only):")
    if not flight_df.empty:
        alt_summary = flight_df.groupby('altitude_bin_m').agg(
            n=('timestamp', 'count'),
            mean_alt=('rx_altitude_m', 'mean'),
            mean_dist=('distance_m', 'mean'),
            dist_range=('distance_m', lambda x: f"{x.min():.0f}–{x.max():.0f}"),
            mean_power=('rx_power_dBm', 'mean'),
        )
        print(alt_summary.to_string())

    print(f"\nFrequency: {df['frequency_Hz'].iloc[0]/1e9:.3f} GHz")
    print(f"TX gain ref: {df['tx_gain_dBm'].iloc[0]} dBm")
    print(f"RX gain ref: {df['rx_gain_dBm'].iloc[0]} dBm")
    print(f"Sample rate: {df['sample_rate'].iloc[0]/1e6:.0f} MHz")

    # Plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        plot_results(df)

    print(f"\n{'='*60}")
    print("DONE — This dataset is used for:")
    print(f"{'='*60}")
    print("• Layer 1 cross-validation (power vs distance at 40, 70, 100m)")
    print("• Comparing against Maeng et al. fitted n(h) at overlapping altitudes")
    print("• The CSV column 'rx_power_dBm' = calibrated received power")
    print("• The CSV column 'distance_m' = 3D distance to transmitter")
    print("• Filter to 'is_flight == True' for clean in-flight measurements")


if __name__ == '__main__':
    main()