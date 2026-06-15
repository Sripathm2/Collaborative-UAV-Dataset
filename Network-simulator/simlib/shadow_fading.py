"""
simlib/shadow_fading.py

Gudmundson correlated shadow fading for UAV links.

Shadow fading X (dB, zero-mean Gaussian with std sigma(h)) is NOT independent
between samples — it decorrelates over distance with correlation distance
d_corr. The standard Gudmundson auto-regressive update is:

    rho = exp(-dd / d_corr)
    X_new = X_old * rho + sigma(h) * sqrt(1 - rho^2) * N(0, 1)

where dd is the mobile's displacement since the previous sample. This keeps
X_new marginally N(0, sigma^2) while imposing the correlation e^{-dd/d_corr}
between consecutive samples at separation dd.

State is kept PER LINK — i.e., one independent shadowing process per
(uav_id, bs_id) pair. Fresh link (first sample) draws from N(0, sigma(h_0)).

Notes / approximations:
  - sigma depends on altitude h; when altitude changes between samples, we
    use the current sigma(h) for the noise term. Gudmundson assumes stationary
    sigma; this is a small approximation acceptable for slowly-varying h.
  - dd is UAV displacement, not UAV-BS distance change. This matches the
    physical interpretation (shadowing decorrelates as the mobile moves
    through the scattering field).
"""

import math
import random


class ShadowFadingField:

    def __init__(self, pathloss_model, d_corr_m=50.0, rng=None):
        """
        pathloss_model: instance of simlib.pathloss.PathLossModel, used for
                        sigma(h) lookups.
        d_corr_m: correlation distance in meters. Default 50 m is a typical
                  value for aerial LOS-dominant links.
        rng: optional random.Random instance for reproducibility. If None,
             uses a fresh default Random() (non-reproducible).
        """
        if d_corr_m <= 0:
            raise ValueError("d_corr_m must be positive")
        self.pathloss_model = pathloss_model
        self.d_corr_m = float(d_corr_m)
        self.rng = rng if rng is not None else random.Random()
        # state[(uav_id, bs_id)] = (last_X_dB, last_uav_position_tuple)
        self._state = {}

    def sample(self, uav_id, bs_id, uav_pos, h):
        """
        Draw the shadow fading contribution (dB) for this link at this instant.

        uav_id, bs_id: any hashable identifiers (strings, ints, ...).
        uav_pos: (x, y, z) tuple of UAV position in meters.
        h: UAV altitude (m); passed through to pathloss_model.sigma(h).

        Returns a float (dB, can be positive or negative). Add this to the
        mean path loss from PathLossModel.mean_pathloss() to get the full
        instantaneous path loss.
        """
        key = (uav_id, bs_id)
        sigma = self.pathloss_model.sigma(h)

        if key not in self._state:
            X = self.rng.gauss(0.0, sigma)
            self._state[key] = (X, tuple(uav_pos))
            return X

        X_old, pos_old = self._state[key]
        dd = math.sqrt(sum((a - b) ** 2 for a, b in zip(uav_pos, pos_old)))
        rho = math.exp(-dd / self.d_corr_m)
        # Clamp the sqrt argument to [0, 1] to guard against floating-point
        # rounding giving a tiny negative when rho ~= 1 (dd near zero).
        innov_scale = math.sqrt(max(0.0, 1.0 - rho * rho))
        X_new = X_old * rho + sigma * innov_scale * self.rng.gauss(0.0, 1.0)

        self._state[key] = (X_new, tuple(uav_pos))
        return X_new

    def reset(self, uav_id=None, bs_id=None):
        """
        Clear stored state. If both ids are None, wipe all links.
        If both are given, drop that one link only.
        """
        if uav_id is None and bs_id is None:
            self._state.clear()
        else:
            key = (uav_id, bs_id)
            self._state.pop(key, None)

    def num_tracked_links(self):
        return len(self._state)


# =========================================================================
# Standalone self-test
# =========================================================================
if __name__ == '__main__':
    import os
    import sys
    import statistics

    # Allow `python3 shadow_fading.py [json_path]` from anywhere
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            here, '..', '..', 'finetuing_layers', 'outputs', 'layer1_comparison.json'
        )

    # Import PathLossModel from sibling module. Works whether run as a script
    # inside simlib/ or from the project root.
    try:
        from pathloss import PathLossModel
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pathloss import PathLossModel

    pl = PathLossModel('logdist', json_path)

    print("=== Shadow Fading self-test ===\n")

    # Test 1: Reproducibility with a seeded RNG
    print("Test 1: Reproducibility under same seed")
    sf_a = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(123))
    sf_b = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(123))
    pos = (0.0, 0.0, 70.0)
    a1 = sf_a.sample('u1', 'b1', pos, 70.0)
    b1 = sf_b.sample('u1', 'b1', pos, 70.0)
    print(f"  sf_a first sample: {a1:.4f}")
    print(f"  sf_b first sample: {b1:.4f}")
    print(f"  match: {abs(a1 - b1) < 1e-12}\n")

    # Test 2: Near-zero displacement => rho ~= 1 => X barely changes
    print("Test 2: Near-zero movement keeps X nearly constant")
    sf = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(7))
    prev = sf.sample('u1', 'b1', (0.0, 0.0, 70.0), 70.0)
    print(f"  X_0 = {prev:.3f}")
    for step in range(5):
        curr = sf.sample('u1', 'b1', (0.001 * (step + 1), 0.0, 70.0), 70.0)
        print(f"  step {step + 1}: dd~0.001m  X = {curr:.3f}  (delta = {curr - prev:+.4f})")
        prev = curr
    print()

    # Test 3: Large displacement => rho ~= 0 => X decorrelates (new N(0, sigma))
    print("Test 3: Large displacement decorrelates X")
    sf = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(42))
    X0 = sf.sample('u1', 'b1', (0.0, 0.0, 70.0), 70.0)
    X_far = sf.sample('u1', 'b1', (500.0, 0.0, 70.0), 70.0)  # 10x d_corr
    rho_expected = math.exp(-500.0 / 50.0)
    print(f"  X_0 = {X0:.3f}")
    print(f"  X after 500m jump = {X_far:.3f}")
    print(f"  Expected rho = exp(-10) = {rho_expected:.2e} (effectively 0)\n")

    # Test 4: Empirical correlation recovers the Gudmundson kernel
    # Run many trajectories with fixed step size and measure lag-1 correlation.
    print("Test 4: Empirical lag-1 correlation vs theory")
    for step_m in (5, 25, 50, 100, 200):
        n_traj = 5000
        xs_curr, xs_prev = [], []
        for t in range(n_traj):
            sf = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(t))
            x0 = sf.sample('u1', 'b1', (0.0, 0.0, 70.0), 70.0)
            x1 = sf.sample('u1', 'b1', (step_m, 0.0, 70.0), 70.0)
            xs_prev.append(x0)
            xs_curr.append(x1)
        mean_p = sum(xs_prev) / n_traj
        mean_c = sum(xs_curr) / n_traj
        cov = sum((a - mean_p) * (b - mean_c) for a, b in zip(xs_prev, xs_curr)) / n_traj
        var_p = sum((a - mean_p) ** 2 for a in xs_prev) / n_traj
        var_c = sum((b - mean_c) ** 2 for b in xs_curr) / n_traj
        rho_emp = cov / math.sqrt(var_p * var_c)
        rho_theo = math.exp(-step_m / 50.0)
        print(f"  step={step_m:>4}m  rho_theory={rho_theo:.3f}  rho_empirical={rho_emp:.3f}")
    print()

    # Test 5: Empirical marginal std recovers sigma(h)
    print("Test 5: Empirical marginal std vs sigma(h)")
    for h in (30, 70, 110):
        n_samples = 10000
        samples = []
        for t in range(n_samples):
            sf = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(t))
            samples.append(sf.sample('u1', 'b1', (0.0, 0.0, h), h))
        s_emp = statistics.stdev(samples)
        s_theo = pl.sigma(h)
        print(f"  h={h}m  sigma_theory={s_theo:.3f}  sigma_empirical={s_emp:.3f}")
    print()

    # Test 6: Independent state per (uav, bs) pair
    print("Test 6: Distinct link keys have independent state")
    sf = ShadowFadingField(pl, d_corr_m=50.0, rng=random.Random(99))
    x_a = sf.sample('u1', 'b1', (0, 0, 70), 70)
    x_b = sf.sample('u1', 'b2', (0, 0, 70), 70)
    x_c = sf.sample('u2', 'b1', (0, 0, 70), 70)
    print(f"  (u1,b1)={x_a:.3f}  (u1,b2)={x_b:.3f}  (u2,b1)={x_c:.3f}")
    print(f"  num tracked links = {sf.num_tracked_links()} (expect 3)\n")

    print("shadow_fading.py self-test complete.")
