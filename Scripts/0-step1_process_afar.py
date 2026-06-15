"""
step1_process_afar.py — Process the AFAR dataset into analysis-ready CSVs.

This is a Python port of main.m and Real_vs_DT.m from the AFAR dataset.
It processes all 5 teams × 3 locations × 2 stages (testbed + development).

Usage:
    python step1_process_afar.py --afar_dir /path/to/AFAR\ \ 2023_SigMF

    Example:
    python 0-step1_process_afar.py --afar_dir ../Datasets/Finetuning-raw/afar/AFAR\ \ 2023_SigMF

Outputs:
    processed/afar_all_flights.csv          — every measurement point, all teams/locs/stages
    processed/afar_testbed.csv              — testbed (real-world) only
    processed/afar_development.csv          — development (digital twin) only
    processed/afar_summary.csv              — per-flight summary statistics
    outputs/plots/afar_power_vs_distance/   — power vs distance plots per flight
    outputs/plots/afar_real_vs_dt/          — Real vs DT comparison plots
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ============================================================
# Ground truth transmitter locations per loc (from main.m)
# These are the UGV (rover) locations that each team was trying to find
# ============================================================

# Per-location origin coordinates (transmitter position)
# Extracted from main.m — these are the same across all teams
LOCATION_ORIGINS = {
    1: {'lat': 35.72806709, 'lon': -78.69730398},
    2: {'lat': 35.72911779, 'lon': -78.69918128},
    3: {'lat': 35.72985129, 'lon': -78.69711002},
}

# Teams in the dataset
TEAMS = [288, 300, 301, 309, 328]
LOCATIONS = [1, 2, 3]
STAGES = ['testbed', 'development']

# Conversion factor from degrees to meters (at AERPAW latitude ~35.7°N)
# This matches the factor used in main.m: sqrt(dx^2 + dy^2) * 1.113195e5
DEG_TO_M = 1.113195e5


# ============================================================
# Data loading functions
# ============================================================

def load_power_csv(filepath):
    """
    Load power_log.csv: unix_timestamp, power_dBm
    No header row.
    """
    df = pd.read_csv(filepath, header=None, names=['timestamp', 'power_dBm'])
    return df


def load_quality_csv(filepath):
    """
    Load quality_log.csv: unix_timestamp, quality
    No header row.
    """
    df = pd.read_csv(filepath, header=None, names=['timestamp', 'quality'])
    return df


def load_gps_csv(filepath):
    """
    Load log.csv (GPS): longitude, latitude, altitude_m, unix_timestamp
    No header row.
    """
    df = pd.read_csv(filepath, header=None,
                     names=['longitude', 'latitude', 'altitude_m', 'timestamp'])
    return df


# ============================================================
# Core processing (port of main.m)
# ============================================================

def process_flight(power_df, quality_df, gps_df, location_num, quality_threshold=0):
    """
    Port of main.m logic:
    1. Filter power measurements by quality threshold
    2. Interpolate GPS onto power timestamps
    3. Compute distance to transmitter
    4. Compute altitude relative to takeoff

    Returns: DataFrame with columns:
        timestamp, power_dBm, quality, longitude, latitude,
        altitude_m, altitude_rel_m, distance_to_tx_m
    """
    # Merge power and quality by closest timestamp
    # They're recorded at the same instants, so direct merge works
    merged = pd.merge_asof(
        power_df.sort_values('timestamp'),
        quality_df.sort_values('timestamp'),
        on='timestamp',
        tolerance=0.1,  # within 100ms
        direction='nearest'
    )

    # Filter by quality threshold
    if quality_threshold > 0:
        merged = merged[merged['quality'] > quality_threshold].copy()

    if len(merged) == 0:
        return None

    # Sort GPS by timestamp for interpolation
    gps_sorted = gps_df.sort_values('timestamp')

    # Only keep power measurements within GPS time range
    gps_start = gps_sorted['timestamp'].iloc[0]
    gps_end = gps_sorted['timestamp'].iloc[-1]
    merged = merged[
        (merged['timestamp'] >= gps_start) &
        (merged['timestamp'] <= gps_end)
    ].copy()

    if len(merged) == 0 or len(gps_sorted) < 2:
        return None

    # Interpolate GPS coordinates at power measurement timestamps
    ts_gps = gps_sorted['timestamp'].values

    interp_lon = interp1d(ts_gps, gps_sorted['longitude'].values,
                          kind='linear', fill_value='extrapolate')
    interp_lat = interp1d(ts_gps, gps_sorted['latitude'].values,
                          kind='linear', fill_value='extrapolate')
    interp_alt = interp1d(ts_gps, gps_sorted['altitude_m'].values,
                          kind='linear', fill_value='extrapolate')

    ts_power = merged['timestamp'].values
    merged['longitude'] = interp_lon(ts_power)
    merged['latitude'] = interp_lat(ts_power)
    merged['altitude_m'] = interp_alt(ts_power)

    # Altitude relative to takeoff (same as main.m: GPSz = GPSz - GPSz(1))
    alt_takeoff = gps_sorted['altitude_m'].iloc[0]
    merged['altitude_rel_m'] = merged['altitude_m'] - alt_takeoff

    # Compute distance to transmitter (matching main.m's formula)
    origin = LOCATION_ORIGINS[location_num]
    dx = merged['longitude'] - origin['lon']
    dy = merged['latitude'] - origin['lat']
    merged['distance_to_tx_m'] = np.sqrt(dx**2 + dy**2) * DEG_TO_M

    # Time relative to GPS start (useful for plotting)
    merged['time_rel_s'] = merged['timestamp'] - gps_start

    return merged


# ============================================================
# Process all flights
# ============================================================

def process_all_flights(afar_dir):
    """
    Walk the AFAR directory structure and process every flight.
    Returns a single DataFrame with all data plus metadata columns.
    """
    afar_path = Path(afar_dir)
    all_frames = []
    summary_rows = []

    for team in TEAMS:
        team_dir = afar_path / str(team)
        if not team_dir.exists():
            print(f"  Team {team}: directory not found, skipping")
            continue

        for stage in STAGES:
            stage_dir = team_dir / stage
            if not stage_dir.exists():
                print(f"  Team {team}/{stage}: not found, skipping")
                continue

            for loc_num in LOCATIONS:
                loc_dir = stage_dir / f'loc{loc_num}'
                if not loc_dir.exists():
                    print(f"  Team {team}/{stage}/loc{loc_num}: not found, skipping")
                    continue

                logs_dir = loc_dir / 'logs'

                # Find the CSV files
                power_file = logs_dir / 'power_log.csv'
                quality_file = logs_dir / 'quality_log.csv'
                gps_file = logs_dir / 'log.csv'

                # Check what's available
                missing = []
                if not power_file.exists():
                    missing.append('power_log.csv')
                if not quality_file.exists():
                    missing.append('quality_log.csv')
                if not gps_file.exists():
                    # Development (DT) stage may not have GPS log
                    missing.append('log.csv')

                if 'power_log.csv' in missing:
                    print(f"  Team {team}/{stage}/loc{loc_num}: "
                          f"missing {missing}, skipping")
                    continue

                # Load data
                power_df = load_power_csv(power_file)
                quality_df = load_quality_csv(quality_file) if quality_file.exists() else None
                gps_df = load_gps_csv(gps_file) if gps_file.exists() else None

                flight_id = f"{team}_{stage}_loc{loc_num}"

                if gps_df is not None and quality_df is not None:
                    result = process_flight(power_df, quality_df, gps_df, loc_num)
                elif gps_df is not None:
                    # No quality data — use power + GPS only
                    dummy_quality = pd.DataFrame({
                        'timestamp': power_df['timestamp'],
                        'quality': 100.0  # assume all valid
                    })
                    result = process_flight(power_df, dummy_quality, gps_df, loc_num)
                else:
                    # No GPS (common for development/DT) — just keep power + quality
                    result = power_df.copy()
                    if quality_df is not None:
                        result = pd.merge_asof(
                            result.sort_values('timestamp'),
                            quality_df.sort_values('timestamp'),
                            on='timestamp', tolerance=0.1, direction='nearest'
                        )
                    result['time_rel_s'] = (result['timestamp'] -
                                            result['timestamp'].iloc[0])

                if result is None or len(result) == 0:
                    print(f"  Team {team}/{stage}/loc{loc_num}: "
                          f"no valid data after processing")
                    continue

                # Add metadata columns
                result['team_id'] = team
                result['stage'] = stage
                result['location'] = loc_num
                result['flight_id'] = flight_id

                all_frames.append(result)

                # Summary statistics
                summary = {
                    'flight_id': flight_id,
                    'team_id': team,
                    'stage': stage,
                    'location': loc_num,
                    'n_measurements': len(result),
                    'duration_s': result['time_rel_s'].max(),
                    'mean_power_dBm': result['power_dBm'].mean(),
                    'std_power_dBm': result['power_dBm'].std(),
                }
                if 'distance_to_tx_m' in result.columns:
                    summary['mean_distance_m'] = result['distance_to_tx_m'].mean()
                    summary['max_distance_m'] = result['distance_to_tx_m'].max()
                if 'altitude_rel_m' in result.columns:
                    summary['mean_altitude_m'] = result['altitude_rel_m'].mean()

                summary_rows.append(summary)
                print(f"  Team {team}/{stage}/loc{loc_num}: "
                      f"{len(result)} points, "
                      f"mean power={result['power_dBm'].mean():.1f} dBm, "
                      f"duration={result['time_rel_s'].max():.0f}s")

    if not all_frames:
        print("\nERROR: No data processed.")
        return None, None

    combined = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    return combined, summary


# ============================================================
# Plotting
# ============================================================

def plot_power_vs_distance(combined_df, output_dir='../results/Step_0/afar_power_vs_distance'):
    """Power vs distance plots per flight (matching main.m Figure 3)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    testbed = combined_df[
        (combined_df['stage'] == 'testbed') &
        combined_df['distance_to_tx_m'].notna()
    ]

    for flight_id, group in testbed.groupby('flight_id'):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(group['distance_to_tx_m'], group['power_dBm'],
                   s=5, alpha=0.5, c='blue')
        ax.set_xlabel('Distance to transmitter (m)')
        ax.set_ylabel('Power (dBm)')
        ax.set_title(f'Power vs Distance: {flight_id}')
        ax.set_ylim(-30, 35)
        ax.set_xlim(0, group['distance_to_tx_m'].max() * 1.1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{flight_id}.png', dpi=100)
        plt.close()

    print(f"  Power vs distance plots saved to {output_dir}/")


def plot_real_vs_dt(combined_df, output_dir='../results/Step_0/afar_real_vs_dt'):
    """
    Port of Real_vs_DT.m — compare testbed vs development power traces.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for team in TEAMS:
        for loc_num in LOCATIONS:
            testbed = combined_df[
                (combined_df['team_id'] == team) &
                (combined_df['stage'] == 'testbed') &
                (combined_df['location'] == loc_num)
            ]
            development = combined_df[
                (combined_df['team_id'] == team) &
                (combined_df['stage'] == 'development') &
                (combined_df['location'] == loc_num)
            ]

            if len(testbed) == 0 or len(development) == 0:
                continue

            # Mean offset (matching Real_vs_DT.m)
            mean_offset = testbed['power_dBm'].mean() - development['power_dBm'].mean()

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(testbed['time_rel_s'], testbed['power_dBm'],
                       s=10, alpha=0.5, c='blue', marker='x',
                       label='Testbed (real-world)')
            ax.scatter(development['time_rel_s'],
                       development['power_dBm'] + mean_offset,
                       s=10, alpha=0.5, c='red', marker='o',
                       label=f'Development (DT) + {mean_offset:.1f} dB offset')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power (dBm)')
            ax.set_title(f'Real vs DT: Team {team}, Location {loc_num}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/team{team}_loc{loc_num}.png', dpi=100)
            plt.close()

    print(f"  Real vs DT plots saved to {output_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process AFAR dataset into analysis-ready CSVs')
    parser.add_argument('--afar_dir', type=str, required=True,
                        help='Path to AFAR 2023_SigMF directory')
    parser.add_argument('--quality_threshold', type=float, default=0,
                        help='Quality threshold for filtering (default: 0, no filter)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()

    print("=" * 60)
    print("AFAR Dataset Processing")
    print("=" * 60)
    print(f"Source: {args.afar_dir}")

    # Verify directory exists
    if not Path(args.afar_dir).exists():
        print(f"\nERROR: Directory not found: {args.afar_dir}")
        print("Make sure the path is correct (watch for spaces in folder name)")
        return

    # List what's there
    contents = sorted(os.listdir(args.afar_dir))
    print(f"Contents: {contents}")

    # Process all flights
    print(f"\nProcessing all flights...")
    combined, summary = process_all_flights(args.afar_dir)

    if combined is None:
        return

    # Save outputs
    out_dir = Path('../Datasets/Finetuning-processed/')
    out_dir.mkdir(exist_ok=True)

    # Full dataset
    combined.to_csv(out_dir / 'afar_all_flights.csv', index=False)
    print(f"\nSaved: {out_dir / 'afar_all_flights.csv'} ({len(combined)} rows)")

    # Split by stage
    testbed = combined[combined['stage'] == 'testbed']
    development = combined[combined['stage'] == 'development']

    testbed.to_csv(out_dir / 'afar_testbed.csv', index=False)
    print(f"Saved: {out_dir / 'afar_testbed.csv'} ({len(testbed)} rows)")

    development.to_csv(out_dir / 'afar_development.csv', index=False)
    print(f"Saved: {out_dir / 'afar_development.csv'} ({len(development)} rows)")

    # Summary
    summary.to_csv(out_dir / 'afar_summary.csv', index=False)
    print(f"Saved: {out_dir / 'afar_summary.csv'} ({len(summary)} rows)")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal measurements: {len(combined)}")
    print(f"  Testbed (real-world): {len(testbed)}")
    print(f"  Development (DT):     {len(development)}")
    print(f"\nFlights processed: {combined['flight_id'].nunique()}")

    print(f"\nPer-stage breakdown:")
    stage_summary = combined.groupby('stage').agg(
        n_flights=('flight_id', 'nunique'),
        n_measurements=('power_dBm', 'count'),
        mean_power=('power_dBm', 'mean'),
        std_power=('power_dBm', 'std'),
    )
    print(stage_summary.to_string())

    if 'distance_to_tx_m' in testbed.columns:
        print(f"\nTestbed distance range: "
              f"{testbed['distance_to_tx_m'].min():.0f}–"
              f"{testbed['distance_to_tx_m'].max():.0f} m")

    if 'altitude_rel_m' in testbed.columns:
        print(f"Testbed altitude range: "
              f"{testbed['altitude_rel_m'].min():.1f}–"
              f"{testbed['altitude_rel_m'].max():.1f} m")

    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        plot_power_vs_distance(combined)
        plot_real_vs_dt(combined)

    print(f"\n{'='*60}")
    print("DONE — Next steps:")
    print(f"{'='*60}")
    print("1. Inspect processed/afar_summary.csv for an overview")
    print("2. The testbed data (processed/afar_testbed.csv) has columns:")
    print("   timestamp, power_dBm, quality, longitude, latitude,")
    print("   altitude_m, altitude_rel_m, distance_to_tx_m, time_rel_s,")
    print("   team_id, stage, location, flight_id")
    print("3. Use this for Layer 1 cross-validation (power vs distance)")
    print("4. Use this for Layer 2 (velocity from GPS traces)")
    print("5. Use testbed + development for Layer 4 (real vs DT comparison)")


if __name__ == '__main__':
    main()