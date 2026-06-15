"""
simlib/pathloss.py

Selectable path loss model for UAV-to-ground / UAV-to-UAV links.

Two models supported:

  'logdist': altitude-dependent log-distance model fitted on AERPAW data.
             PL(d, h) = PL0(h) + 10 * n(h) * log10(d / d0)
             where PL0, n are polynomials in altitude h (meters).
             Fit source: finetuing_layers/outputs/layer1_comparison.json

  '3gpp':    3GPP TR 36.777 Rural-AV LOS model with calibration offset fitted
             on the same AERPAW data.
             PL_LOS(d, h) = max(23.9 - 1.8*log10(h), 20) * log10(d)
                          + 20*log10(40*pi*f_GHz/3)
                          + offset
             Shadow sigma: 4.2 * exp(-0.0046 * h)   (3GPP Rural-AV LOS)

Altitude clamping:
  The log-distance polynomials were fitted on altitudes 30..110 m (5 anchor
  points). Outside this range the polynomial can extrapolate nonsensically.
  For path-loss PARAMETER LOOKUP we clamp h to [30, 110]. The real 3D slant
  distance is always used as-is for the distance-dependent term. A clamp
  counter is maintained for end-of-run diagnostics.

Shadow fading is NOT included here — this module returns mean path loss only.
Use simlib.shadow_fading.ShadowFadingField for the correlated stochastic term,
then add its sample to this mean.
"""

import json
import math


class PathLossModel:

    # Layer 1 fit range. Polynomials were fitted on these altitudes.
    ALT_MIN = 30.0
    ALT_MAX = 110.0

    # Reference distance (standard log-distance convention)
    D0_M = 1.0

    SUPPORTED_MODELS = ('logdist', '3gpp')

    def __init__(self, model_name, params_json_path):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown path loss model {model_name!r}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        self.model_name = model_name

        with open(params_json_path, 'r') as f:
            data = json.load(f)

        if model_name == 'logdist':
            p = data['models']['Log-Distance']['params']
            # coeffs are stored as [a2, a1, a0] meaning a2*h^2 + a1*h + a0
            self.PL0_coeffs = p['PL0_coeffs']
            self.n_coeffs = p['n_coeffs']
            self.sigma_coeffs = p['sigma_coeffs']
        else:  # '3gpp'
            p = data['models']['3GPP TR 36.777']['params']
            self.offset = p['offset']
            self.frequency_GHz = p['frequency_GHz']

        # Diagnostics: number of times altitude clamping was triggered
        self.clamp_count = 0
        self.clamp_low_count = 0
        self.clamp_high_count = 0

    # ---- internal helpers ----

    @staticmethod
    def _poly2(coeffs, x):
        """Evaluate [a2, a1, a0] at x."""
        return coeffs[0] * x * x + coeffs[1] * x + coeffs[2]

    def _clamp_altitude(self, h):
        if h < self.ALT_MIN:
            self.clamp_count += 1
            self.clamp_low_count += 1
            return self.ALT_MIN
        if h > self.ALT_MAX:
            self.clamp_count += 1
            self.clamp_high_count += 1
            return self.ALT_MAX
        return h

    # ---- public API ----

    def sigma(self, h):
        """
        Shadow fading standard deviation (dB) at UAV altitude h (m).
        Altitude is clamped to the fit range for parameter lookup.
        """
        h_c = self._clamp_altitude(h)
        if self.model_name == 'logdist':
            s = self._poly2(self.sigma_coeffs, h_c)
        else:
            # 3GPP TR 36.777 Rural-AV LOS
            s = 4.2 * math.exp(-0.0046 * h_c)
        # Guard: sigma must be positive
        return max(s, 0.1)

    def mean_pathloss(self, d_m, h):
        """
        Mean path loss (dB) for a link at 3D slant distance d_m meters and
        UAV altitude h meters. Does NOT include shadow fading — add that
        separately via ShadowFadingField.sample().

        d_m is floored at 1e-3 m to avoid log10(0).
        """
        if d_m <= 0:
            d_m = 1e-3
        h_c = self._clamp_altitude(h)

        if self.model_name == 'logdist':
            PL0 = self._poly2(self.PL0_coeffs, h_c)
            n = self._poly2(self.n_coeffs, h_c)
            return PL0 + 10.0 * n * math.log10(d_m / self.D0_M)
        else:
            # 3GPP TR 36.777 Rural-AV LOS
            exponent = max(23.9 - 1.8 * math.log10(h_c), 20.0)
            fspl_constant = 20.0 * math.log10(
                40.0 * math.pi * self.frequency_GHz / 3.0
            )
            return exponent * math.log10(max(d_m, 1.0)) + fspl_constant + self.offset

    def diagnostics(self):
        """Return a dict summarising clamp activity for logging at end of run."""
        return {
            'model': self.model_name,
            'clamp_total': self.clamp_count,
            'clamp_below_30m': self.clamp_low_count,
            'clamp_above_110m': self.clamp_high_count,
        }


# =========================================================================
# Standalone self-test
# =========================================================================
if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Default to project-relative path when run from Network-simulator/simlib/
        here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            here, '..', '..', 'finetuing_layers', 'outputs', 'layer1_comparison.json'
        )

    print(f"Loading fitted params from: {json_path}\n")

    for model_name in ('logdist', '3gpp'):
        print(f"================ Model: {model_name} ================")
        pl = PathLossModel(model_name, json_path)

        # Mean PL grid
        print("Mean path loss (dB), rows = distance, cols = altitude:")
        print(f"{'d (m)':>8} | " + "  ".join(f"h={h}m" for h in (30, 50, 70, 90, 110)))
        print("-" * 58)
        for d in (10, 50, 100, 250, 500, 1000, 2000):
            row = f"{d:>8.0f} | "
            row += "  ".join(f"{pl.mean_pathloss(d, h):>5.1f}" for h in (30, 50, 70, 90, 110))
            print(row)

        # Sigma curve
        print("\nShadow fading sigma(h):")
        for h in (30, 50, 70, 90, 110):
            print(f"  h={h:>3}m : sigma = {pl.sigma(h):.2f} dB")

        # Sanity check against the per-altitude values stored in JSON
        # (only applies to logdist)
        if model_name == 'logdist':
            with open(json_path) as f:
                ref = json.load(f)['models']['Log-Distance']['params']['per_altitude']
            print("\nSanity check — polynomial vs per-altitude fit values:")
            print(f"{'h':>5} {'n_poly':>8} {'n_fit':>8} {'PL0_poly':>10} {'PL0_fit':>10} {'sig_poly':>9} {'sig_fit':>9}")
            for h_str, vals in ref.items():
                h = float(h_str)
                n_poly = pl._poly2(pl.n_coeffs, h)
                PL0_poly = pl._poly2(pl.PL0_coeffs, h)
                sig_poly = pl._poly2(pl.sigma_coeffs, h)
                print(f"{h:>5.0f} {n_poly:>8.3f} {vals['n']:>8.3f} "
                      f"{PL0_poly:>10.3f} {vals['PL0_local']:>10.3f} "
                      f"{sig_poly:>9.3f} {vals['sigma']:>9.3f}")

        # Altitude clamping behaviour
        pl.clamp_count = pl.clamp_low_count = pl.clamp_high_count = 0
        _ = pl.mean_pathloss(100, 5)     # below
        _ = pl.mean_pathloss(100, 20)    # below
        _ = pl.mean_pathloss(100, 75)    # in range
        _ = pl.mean_pathloss(100, 150)   # above
        print(f"\nClamp diagnostics after 4 calls (2 below, 1 in, 1 above):")
        print(f"  {pl.diagnostics()}")

        # Edge case: d=0
        pl_zero = pl.mean_pathloss(0, 70)
        print(f"\nd=0 safety (should be finite, not +inf): PL = {pl_zero:.1f} dB")

        print()

    print("pathloss.py self-test complete.")
