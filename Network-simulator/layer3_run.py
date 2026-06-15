#!/usr/bin/env python3
"""layer3_run.py
Replay a prepared trajectory through the calibrated path-loss + shadow-fading
+ modulation chain. Emits per-step physics CSV.

For each AADM sample time point t_rel in trajectory.csv:
  - pos    = (distance, 0, altitude)   from _ReplayMission
  - PL     = pathloss.mean_pathloss(d, h) + shadow_field.sample(...)
  - RSS    = tx_power - PL                 (dBm)
  - SNR    = tx_power - PL - noise_floor   (dB)
  - bw,per = modulation.compute_link(scheme, snr_db)

Output: results/layer3/sim_<flight_id>.csv with columns
  time_rel_s, d_m, h_m, mean_pl_dB, total_pl_dB, rss_dBm, snr_dB,
  bw_mbit, per, scheme

Standalone (no Containernet needed). Can also be invoked by Topo.py --layer3.
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Make simlib importable. Adjust SIMLIB_PARENT if your layout differs.
SIMLIB_PARENT = Path("../Network-simulator")
if str(SIMLIB_PARENT) not in sys.path:
    sys.path.insert(0, str(SIMLIB_PARENT))

from simlib.pathloss      import PathLossModel
from simlib.shadow_fading import ShadowFadingField
from simlib.modulation    import compute_link as compute_modulation_link
from simlib.mobility      import MobilityGenerator

# _ReplayMission lives in your patched mobility.py once the patch is applied.
# If you haven't applied the patch yet, this import will fail — see
# mobility_replay_patch.py for the class body to paste in.
from simlib.mobility      import _ReplayMission

# =========================================================================
# CONFIG
# =========================================================================
LAYER1_JSON = "../results/Step_1/layer1_comparison.json"
LAYER2_JSON = "../results/Step_1/layer2a_mobility_library.json"

OUT_DIR = Path("../results/layer3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match Topo.py defaults
SHADOW_CORRELATION_M = 50.0
DEFAULT_TX_POWER_DBM    = 30.0
DEFAULT_NOISE_FLOOR_DBM = -58.0
DEFAULT_MODULATION      = "adaptive"
DEFAULT_PATHLOSS_MODEL  = "logdist"
DEFAULT_SEED            = 41


def run_layer3(flight_id: str,
               pathloss_model_name: str = DEFAULT_PATHLOSS_MODEL,
               modulation_scheme:   str = DEFAULT_MODULATION,
               tx_power_dbm:      float = DEFAULT_TX_POWER_DBM,
               noise_floor_dbm:   float = DEFAULT_NOISE_FLOOR_DBM,
               seed:                int = DEFAULT_SEED,
               include_shadow:     bool = True,
               out_suffix:          str = ""):
    traj_path = OUT_DIR / f"trajectory_{flight_id}.csv"
    if not traj_path.exists():
        raise SystemExit(
            f"missing {traj_path}; run layer3_prepare.py --flight_id {flight_id} first"
        )

    rng = random.Random(seed)
    pl  = PathLossModel(pathloss_model_name, LAYER1_JSON)
    sf  = ShadowFadingField(pl, d_corr_m=SHADOW_CORRELATION_M, rng=rng)

    # Bypass MobilityGenerator library lookup; inject _ReplayMission directly.
    mob = MobilityGenerator(LAYER2_JSON, rng=rng)
    mob.uavs["d1"] = _ReplayMission(
        params={"trajectory_csv": str(traj_path)},
        origin_xy=(0.0, 0.0),
    )

    # Read trajectory time points so we step at the SAME instants as the
    # measurements — enables direct point-to-point comparison.
    times = []
    with open(traj_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            times.append(float(row["time_rel_s"]))
    print(f"  walking {len(times):,} time points "
          f"(span {times[0]:.1f} -> {times[-1]:.1f}s)")

    out_path = OUT_DIR / f"sim_{flight_id}{out_suffix}.csv"
    n_written = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_rel_s", "d_m", "h_m",
                    "mean_pl_dB", "total_pl_dB", "rss_dBm", "snr_dB",
                    "bw_mbit", "per", "scheme"])
        for t in times:
            x, y, z = mob.position("d1", t)
            d_m = max((x * x + y * y + z * z) ** 0.5, 1e-3)
            h_m = z
            mean_pl = pl.mean_pathloss(d_m, h_m)
            if include_shadow:
                shadow = sf.sample("d1", "bs1", (x, y, z), h_m)
            else:
                shadow = 0.0
            total_pl = mean_pl + shadow
            rss_dbm  = tx_power_dbm - total_pl
            snr_db   = tx_power_dbm - total_pl - noise_floor_dbm
            res = compute_modulation_link(modulation_scheme, snr_db,
                                          packet_bits=8000)
            # IMPORTANT: write time_rel_s at full precision (repr) so the
            # downstream merge with ground_truth_<fid>.csv (also full precision)
            # produces an exact float match for every sample. Truncating to
            # f"{t:.6f}" caused near-empty merges (n=2..8 instead of N).
            w.writerow([repr(t), f"{d_m:.4f}", f"{h_m:.4f}",
                        f"{mean_pl:.4f}", f"{total_pl:.4f}",
                        f"{rss_dbm:.4f}", f"{snr_db:.4f}",
                        f"{res['rate_mbps']:.4f}",
                        f"{res['per']:.6f}",
                        res["scheme"]])
            n_written += 1

    print(f"  wrote {out_path}  ({n_written:,} rows)")
    print(f"  pathloss clamps: {pl.diagnostics()}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight_id", required=True)
    ap.add_argument("--pathloss_model", default=DEFAULT_PATHLOSS_MODEL,
                    choices=["logdist", "3gpp"])
    ap.add_argument("--modulation", default=DEFAULT_MODULATION)
    ap.add_argument("--tx_power_dbm", type=float, default=DEFAULT_TX_POWER_DBM)
    ap.add_argument("--noise_floor_dbm", type=float,
                    default=DEFAULT_NOISE_FLOOR_DBM)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no_shadow", action="store_true",
                    help="disable stochastic shadow fading (mean-only)")
    ap.add_argument("--out_suffix", default="",
                    help="appended to sim_<fid>{suffix}.csv (e.g. "
                         "'__logdist_shadow') so sweep variants don't overwrite")
    args = ap.parse_args()

    run_layer3(
        flight_id           = args.flight_id,
        pathloss_model_name = args.pathloss_model,
        modulation_scheme   = args.modulation,
        tx_power_dbm        = args.tx_power_dbm,
        noise_floor_dbm     = args.noise_floor_dbm,
        seed                = args.seed,
        include_shadow      = not args.no_shadow,
        out_suffix          = args.out_suffix,
    )


if __name__ == "__main__":
    main()