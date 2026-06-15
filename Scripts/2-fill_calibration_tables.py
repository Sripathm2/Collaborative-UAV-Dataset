#!/usr/bin/env python3
"""
fill_calibration_tables.py
Emits LaTeX rows for:
  - tab:pathloss_params  (Table 5)  -- n(h), PL0(h), sigma(h) at 5 altitudes
  - tab:mobility_params  (Table 6)  -- per-mission fitted params

Usage:
  python3 2-fill_calibration_tables.py \
      --layer1 ../results/Step_1/layer1_comparison.json \
      --layer2 ../results/Step_1/layer2a_mobility_library.json \
      --out ../results/Step_2/tables.tex
"""

import argparse
import json
from pathlib import Path


def fmt(v, nd=2):
    if v is None:
        return "--"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def build_pathloss_table(layer1_path: Path) -> str:
    data = json.loads(layer1_path.read_text())
    per_alt = data["models"]["Log-Distance"]["params"]["per_altitude"]
    alts = sorted(per_alt.keys(), key=float)

    lines = []
    lines.append("% --- Table 5: Fitted path loss parameters ---")
    lines.append("% Source: layer1_comparison.json -> models['Log-Distance']['params']['per_altitude']")
    lines.append("% Columns: Altitude (m) | n(h) | PL0(h) (dB) | sigma(h) (dB)")
    for k in alts:
        p = per_alt[k]
        h = int(float(k))
        n = fmt(p["n"], 3)
        pl0 = fmt(p["PL0_local"], 2)
        sig = fmt(p["sigma"], 2)
        lines.append(f"{h:<3} & {n} & {pl0} & {sig} \\\\")
    return "\n".join(lines)


def build_mobility_table(layer2_path: Path) -> str:
    data = json.loads(layer2_path.read_text())
    lib = data["library"]

    # TeX table columns: Mission | alpha | v_bar (m/s) | sigma_v^2 | Altitude (m)
    # spiral/grid/hover_transit are mission-driven (speed-only), not Gauss-Markov.
    # random is the only true Gauss-Markov fit.
    rows = []

    sp = lib["spiral"]
    rows.append((
        "Spiral",
        "--",
        fmt(sp["speed"], 3),
        "--",
        f"{fmt(sp['altitude'], 2)} (clamped to 30)",
    ))

    gr = lib["grid"]
    rows.append((
        "Grid",
        "--",
        fmt(gr["speed"], 3),
        "--",
        f"{fmt(gr['altitude'], 2)} (clamped to 30)",
    ))

    hv = lib["hover_transit"]
    # mean of nonzero altitudes (the cruise legs)
    cruise_alts = [a for a in hv["altitudes"] if a > 1.0]
    mean_cruise = sum(cruise_alts) / len(cruise_alts) if cruise_alts else 50.0
    rows.append((
        "Hover-transit",
        "--",
        fmt(hv["speed"], 3),
        "--",
        f"{fmt(mean_cruise, 2)} (forced to 50)",
    ))

    rn = lib["random"]
    # report horizontal Gauss-Markov params (vertical reported separately in caption)
    alpha_h = fmt(rn["alpha_h"], 3)
    v_mean_h = fmt(rn["v_mean_h"], 3)
    v_var_h = fmt(rn["v_std_h"] ** 2, 3)
    rows.append((
        "Random",
        alpha_h,
        v_mean_h,
        v_var_h,
        "Varies",
    ))

    out = []
    out.append("")
    out.append("% --- Table 6: Fitted mobility parameters ---")
    out.append("% Source: layer2a_mobility_library.json -> library[*]")
    out.append("% spiral/grid/hover_transit are mission-driven; only 'random' is Gauss-Markov.")
    out.append("% For random, we report HORIZONTAL components (alpha_h, v_mean_h, v_std_h^2).")
    out.append(f"% Random vertical: alpha_v={fmt(rn['alpha_v'],3)}, "
               f"v_mean_v={fmt(rn['v_mean_v'],3)}, "
               f"v_var_v={fmt(rn['v_std_v']**2,3)} -- mention in caption.")
    for r in rows:
        out.append(" & ".join(r) + " \\\\")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1", required=True, type=Path)
    ap.add_argument("--layer2", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    block = []
    block.append(build_pathloss_table(args.layer1))
    block.append(build_mobility_table(args.layer2))
    text = "\n".join(block) + "\n"

    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()