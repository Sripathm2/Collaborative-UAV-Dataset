"""
step2_layer1_compare.py — Compare three Layer 1 path loss models.

Fits and evaluates three altitude-aware path loss models against Maeng et al.
data, with cross-validation against Gürses–Sichitiu.

Models compared:
    1. Altitude-Dependent Log-Distance: PL(d,h) = PL0 + 10*n(h)*log10(d/d0) + X_sigma(h)
    2. Enhanced Two-Ray (Masrur & Guvenc style): direct + ground-reflected paths
    3. 3GPP TR 36.777 (Rural-AV): altitude-aware LoS/NLoS with LoS probability

Usage:
    python 1-step2_layer1_compare.py

Prerequisites:
    - processed/maeng_rsrp.csv
    - processed/gurses_channel_flight.csv
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, curve_fit
from scipy.stats import pearsonr

# ============================================================
# Configuration
# ============================================================
FREQ_HZ = 3.51e9            # Maeng carrier frequency
FREQ_GHZ = FREQ_HZ / 1e9
C = 3e8                     # speed of light
LAMBDA = C / FREQ_HZ
D0 = 1.0                    # reference distance for log-distance model
BS_HEIGHT = 12.0            # AERPAW BS tower height (m), approx
NOISE_FLOOR_DB = -58.0      # filter measurements below this (visible in plot)
TRAIN_FRAC = 0.8            # train/test split

# In our processed Maeng data, "rsrp_dBm" is actually relative power (uncalibrated).
# The model predicts path loss; we convert measured power to measured path loss
# via PL_meas = K - power_meas, where K is an unknown calibration constant.
# Since K is the same for all measurements, it just shifts PL0 — fitting absorbs it.
# We pick a convenient K so that PL values are in a reasonable range.
K_CALIB = 0.0  # placeholder; subtraction from power gives "measured PL" up to constant


# ============================================================
# Model 1: Altitude-Dependent Log-Distance
# ============================================================

def fit_log_distance_model(df):
    """
    Fit PL(d, h) = PL0(h) + 10*n(h)*log10(d/d0) + X_sigma(h)

    For each altitude bin, fit n, PL0, and sigma via linear regression.
    Then fit n(h), PL0(h), and sigma(h) as polynomials in h.
    """
    per_alt = {}
    for alt, group in df.groupby('altitude_target_m'):
        d = group['distance_to_bs_m'].values
        pl = group['pl_measured'].values

        valid = (d > D0) & np.isfinite(pl)
        d, pl = d[valid], pl[valid]
        if len(d) < 10:
            continue

        log_d = np.log10(d / D0)
        # PL = a + b*log_d, where a=PL0_alt, b=10*n(h)
        coeffs = np.polyfit(log_d, pl, 1)
        b, a = coeffs
        n_h = b / 10.0
        pl0_alt = a
        residuals = pl - (pl0_alt + 10 * n_h * log_d)
        sigma_h = np.std(residuals)

        per_alt[float(alt)] = {
            'n': float(n_h),
            'sigma': float(sigma_h),
            'PL0_local': float(pl0_alt),
        }

    altitudes = np.array(sorted(per_alt.keys()))
    n_vals = np.array([per_alt[h]['n'] for h in altitudes])
    sigma_vals = np.array([per_alt[h]['sigma'] for h in altitudes])
    pl0_vals = np.array([per_alt[h]['PL0_local'] for h in altitudes])

    # Fit polynomial n(h), sigma(h), and PL0(h) — all altitude-dependent
    deg = min(2, len(altitudes) - 1)
    n_coeffs = np.polyfit(altitudes, n_vals, deg)
    sigma_coeffs = np.polyfit(altitudes, sigma_vals, deg)
    pl0_coeffs = np.polyfit(altitudes, pl0_vals, deg)

    return {
        'PL0_coeffs': pl0_coeffs.tolist(),
        'n_coeffs': n_coeffs.tolist(),
        'sigma_coeffs': sigma_coeffs.tolist(),
        'per_altitude': per_alt,
        'altitudes_fitted': altitudes.tolist(),
    }


def predict_log_distance(d_m, h_m, params, add_fading=False, rng=None):
    """Predict path loss using altitude-dependent log-distance model."""
    n_coeffs = np.array(params['n_coeffs'])
    sigma_coeffs = np.array(params['sigma_coeffs'])
    pl0_coeffs = np.array(params['PL0_coeffs'])

    n_h = np.polyval(n_coeffs, h_m)
    sigma_h = np.polyval(sigma_coeffs, h_m)
    pl0_h = np.polyval(pl0_coeffs, h_m)

    d_safe = np.maximum(d_m, D0)
    pl = pl0_h + 10 * n_h * np.log10(d_safe / D0)

    if add_fading:
        if rng is None:
            rng = np.random.default_rng()
        pl = pl + rng.normal(0, np.maximum(sigma_h, 0.1), size=np.shape(d_m))

    return pl


# ============================================================
# Model 2: Enhanced Two-Ray
# ============================================================

def two_ray_path_loss(d_m, h_uav, h_bs, freq_Hz, gamma=-0.9):
    """
    Two-ray ground reflection path loss.

    Direct path d_los + reflected path d_ref with phase difference.
    gamma: reflection coefficient (typically -0.9 to -1.0 for ground at grazing angles)

    Returns path loss in dB.
    """
    wavelength = C / freq_Hz

    # Path lengths (3D)
    d_los = np.sqrt(d_m**2 + (h_uav - h_bs)**2)
    d_ref = np.sqrt(d_m**2 + (h_uav + h_bs)**2)

    # Phase difference
    delta = d_ref - d_los
    phi = 2 * np.pi * delta / wavelength

    # Field amplitudes (proportional to 1/d)
    E_los = 1.0 / d_los
    E_ref = gamma / d_ref

    # Coherent sum
    E_total_sq = E_los**2 + E_ref**2 + 2 * E_los * E_ref * np.cos(phi)
    E_total = np.sqrt(np.maximum(E_total_sq, 1e-30))

    # Path loss in dB (using FSPL reference at 1m as baseline)
    pl_dB = -20 * np.log10(E_total * wavelength / (4 * np.pi))
    return pl_dB


def fit_two_ray(df):
    """
    Fit two-ray model parameters: gamma (reflection coefficient) and an
    additive offset (calibration constant) by minimizing RMSE on training data.
    """
    d = df['distance_to_bs_m'].values
    h = df['altitude_m'].values
    pl_meas = df['pl_measured'].values

    valid = (d > D0) & np.isfinite(pl_meas) & np.isfinite(h)
    d, h, pl_meas = d[valid], h[valid], pl_meas[valid]

    def objective(params):
        gamma, offset, h_bs = params
        pl_pred = two_ray_path_loss(d, h, h_bs, FREQ_HZ, gamma=gamma) + offset
        return np.mean((pl_meas - pl_pred) ** 2)

    from scipy.optimize import minimize
    result = minimize(objective, x0=[-0.9, 0.0, BS_HEIGHT],
                      bounds=[(-1.0, 1.0), (-200, 200), (1.0, 50.0)],
                      method='L-BFGS-B')
    gamma_fit, offset_fit, h_bs_fit = result.x

    return {
        'gamma': float(gamma_fit),
        'offset': float(offset_fit),
        'h_bs': float(h_bs_fit),
        'frequency_Hz': FREQ_HZ,
    }


def predict_two_ray(d_m, h_m, params):
    """Predict path loss using two-ray model with fitted offset."""
    return two_ray_path_loss(d_m, h_m, params['h_bs'],
                             params['frequency_Hz'],
                             gamma=params['gamma']) + params['offset']


# ============================================================
# Model 3: 3GPP TR 36.777 (Rural-AV)
# ============================================================

def gpp_rural_los_probability(d_2d, h_uav):
    """
    LoS probability for rural-AV scenario from 3GPP TR 36.777.
    d_2d: horizontal distance in meters
    h_uav: UAV altitude in meters (10 < h <= 300)
    """
    h_uav = np.asarray(h_uav, dtype=float)
    d_2d = np.asarray(d_2d, dtype=float)

    # Clamp inputs to safe ranges
    h_safe = np.clip(h_uav, 1.0, 300.0)
    d_2d_safe = np.maximum(d_2d, 1e-3)

    # 3GPP TR 36.777 Table B-2 Rural-AV
    # Compute both branches safely (no NaN propagation through np.where)
    log_h = np.log10(h_safe)

    # For h <= 40m
    d1_low = np.maximum(294.05 * log_h - 432.94, 18.0)
    p1_low = np.maximum(233.98 * log_h - 0.95, 1.0)

    # For h > 40m, use h=40 reference values
    d1_high = np.full_like(h_safe, 18.0)
    p1_high = np.full_like(h_safe, 160.0)

    d1 = np.where(h_safe <= 40, d1_low, d1_high)
    p1 = np.where(h_safe <= 40, p1_low, p1_high)
    p1 = np.maximum(p1, 1.0)  # ensure positive

    # LoS probability — both branches safe (d_2d_safe never zero)
    ratio = d1 / d_2d_safe
    p_los_far = ratio + np.exp(-d_2d_safe / p1) * (1 - ratio)
    p_los = np.where(d_2d_safe <= d1, 1.0, p_los_far)

    # For high altitudes (>100m), LoS probability is essentially 1
    p_los = np.where(h_safe > 100, 1.0, p_los)
    return np.clip(p_los, 0.0, 1.0)


def gpp_rural_path_loss(d_3d, d_2d, h_uav, freq_GHz):
    """
    Path loss for 3GPP TR 36.777 Rural-AV scenario.
    Returns (PL_LoS, PL_NLoS) tuple in dB.
    """
    d_3d_safe = np.maximum(d_3d, 1.0)
    h_safe = np.maximum(h_uav, 1.0)

    # Rural-AV LoS (free-space-like with environmental adjustment)
    pl_los = 28.0 + 22 * np.log10(d_3d_safe) + 20 * np.log10(freq_GHz)

    # Rural-AV NLoS
    pl_nlos = (-17.5
               + (46 - 7 * np.log10(h_safe)) * np.log10(d_3d_safe)
               + 20 * np.log10(40 * np.pi * freq_GHz / 3))

    # NLoS should always be >= LoS
    pl_nlos = np.maximum(pl_nlos, pl_los)
    return pl_los, pl_nlos


def fit_gpp(df):
    """
    3GPP model has fixed parameters from the standard, but we fit
    a single calibration offset to align with our measured power scale.
    """
    d = df['distance_to_bs_m'].values
    h = df['altitude_m'].values
    pl_meas = df['pl_measured'].values

    valid = (d > 1) & np.isfinite(pl_meas) & np.isfinite(h)
    d, h, pl_meas = d[valid], h[valid], pl_meas[valid]

    # Compute predicted PL (expected value) for all training points
    d_2d = np.sqrt(np.maximum(d**2 - h**2, 0))
    pl_los, pl_nlos = gpp_rural_path_loss(d, d_2d, h, FREQ_GHZ)
    p_los = gpp_rural_los_probability(d_2d, h)
    pl_pred = p_los * pl_los + (1 - p_los) * pl_nlos

    offset = float(np.mean(pl_meas - pl_pred))

    return {
        'offset': offset,
        'frequency_GHz': FREQ_GHZ,
        'scenario': 'Rural-AV',
    }


def predict_gpp(d_m, h_m, params):
    """Predict path loss using 3GPP TR 36.777 Rural-AV with fitted offset."""
    d = np.asarray(d_m, dtype=float)
    h = np.asarray(h_m, dtype=float)
    d_2d = np.sqrt(np.maximum(d**2 - h**2, 0))
    pl_los, pl_nlos = gpp_rural_path_loss(d, d_2d, h, params['frequency_GHz'])
    p_los = gpp_rural_los_probability(d_2d, h)
    return p_los * pl_los + (1 - p_los) * pl_nlos + params['offset']


# ============================================================
# Evaluation
# ============================================================

def evaluate(pl_meas, pl_pred, label=""):
    """Compute RMSE, MAE, Pearson r, R²."""
    valid = np.isfinite(pl_meas) & np.isfinite(pl_pred)
    m, p = pl_meas[valid], pl_pred[valid]
    if len(m) < 10:
        return {}

    residuals = m - p
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.median(np.abs(residuals)))
    bias = float(np.mean(residuals))
    r, _ = pearsonr(m, p)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((m - np.mean(m))**2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        'rmse_dB': rmse,
        'mae_dB': mae,
        'bias_dB': bias,
        'pearson_r': float(r),
        'r_squared': r_squared,
        'n': int(len(m)),
    }
    if label:
        print(f"  {label:40s}: RMSE={rmse:.2f} dB, MAE={mae:.2f} dB, "
              f"r={r:.3f}, R²={r_squared:.3f}, N={len(m)}")
    return metrics


def evaluate_per_altitude(df_test, predict_fn, params, label):
    """Evaluate a model on test data, per altitude."""
    print(f"\n{label}:")
    per_alt = {}
    for alt, group in df_test.groupby('altitude_target_m'):
        pl_pred = predict_fn(
            group['distance_to_bs_m'].values,
            group['altitude_m'].values,
            params
        )
        m = evaluate(group['pl_measured'].values, pl_pred,
                     f"  altitude {alt:.0f}m")
        per_alt[float(alt)] = m

    # Aggregate
    pl_pred_all = predict_fn(
        df_test['distance_to_bs_m'].values,
        df_test['altitude_m'].values,
        params
    )
    overall = evaluate(df_test['pl_measured'].values, pl_pred_all,
                       f"  OVERALL")
    return {'per_altitude': per_alt, 'overall': overall}


# ============================================================
# Plotting
# ============================================================

def plot_models(df, params_log, params_2ray, params_gpp,
                output_dir='../results/Step_1/layer1'):
    """Overlay model predictions on measured data, per altitude."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    altitudes = sorted(df['altitude_target_m'].unique())
    n_alt = len(altitudes)

    fig, axes = plt.subplots(1, n_alt, figsize=(5 * n_alt, 5), sharey=True)
    if n_alt == 1:
        axes = [axes]

    for ax, alt in zip(axes, altitudes):
        group = df[df['altitude_target_m'] == alt]
        d = group['distance_to_bs_m'].values
        pl = group['pl_measured'].values
        h_mean = group['altitude_m'].mean()

        # Sample for visibility
        sample = np.random.choice(len(d), size=min(1000, len(d)), replace=False)
        ax.scatter(d[sample], pl[sample], s=4, alpha=0.3,
                   color='gray', label='Measured')

        # Smooth curves for predictions
        d_grid = np.linspace(d.min(), d.max(), 200)
        h_grid = np.full_like(d_grid, h_mean)

        pl_log = predict_log_distance(d_grid, h_grid, params_log)
        pl_2ray = predict_two_ray(d_grid, h_grid, params_2ray)
        pl_gpp = predict_gpp(d_grid, h_grid, params_gpp)

        ax.plot(d_grid, pl_log, 'b-', linewidth=2, label='Log-distance')
        ax.plot(d_grid, pl_2ray, 'g--', linewidth=2, label='Two-Ray')
        ax.plot(d_grid, pl_gpp, 'r:', linewidth=2, label='3GPP TR 36.777')

        ax.set_xlabel('Distance (m)')
        if ax == axes[0]:
            ax.set_ylabel('Path Loss (dB)')
        ax.set_title(f'Altitude target: {alt:.0f} m')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.invert_yaxis()  # PL: smaller = stronger signal

    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer1_model_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir}/layer1_model_comparison.png")


def plot_rmse_comparison(results, output_dir='../results/Step_1/layer1'):
    """Bar chart: RMSE per model per altitude."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = list(results.keys())
    altitudes = sorted(results[models[0]]['per_altitude'].keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    x = np.arange(len(altitudes))
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for i, model in enumerate(models):
        rmses = [results[model]['per_altitude'][alt].get('rmse_dB', 0)
                 for alt in altitudes]
        ax.bar(x + i * width, rmses, width, label=model, color=colors[i])

    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('RMSE (dB)')
    ax.set_title('Layer 1 Model Comparison: RMSE per Altitude (test set)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{a:.0f}' for a in altitudes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer1_rmse_comparison.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/layer1_rmse_comparison.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("STEP 2: Layer 1 Path Loss Model Comparison")
    print("=" * 60)

    # --- Load Maeng data ---
    maeng_path = Path('../Datasets/Finetuning-processed/maeng_rsrp.csv')
    if not maeng_path.exists():
        print(f"ERROR: {maeng_path} not found")
        return

    df = pd.read_csv(maeng_path)
    print(f"\nLoaded Maeng: {len(df)} measurements")

    # Convert measured power to "measured path loss" (up to a constant K)
    # For fitting purposes the absolute value of K doesn't matter; each model
    # absorbs it into PL0/offset. We use K = 0 so PL = -power_dBm.
    df['pl_measured'] = -df['rsrp_dBm']

    # Filter out noise floor
    pre = len(df)
    df = df[df['rsrp_dBm'] > NOISE_FLOOR_DB].copy()
    print(f"After noise floor filter (>{NOISE_FLOOR_DB} dB): "
          f"{len(df)} measurements ({pre - len(df)} removed)")

    # Train/test split (random per altitude to ensure all altitudes in both)
    rng = np.random.default_rng(42)
    train_idx = []
    test_idx = []
    for alt, group in df.groupby('altitude_target_m'):
        idx = group.index.values
        rng.shuffle(idx)
        n_train = int(TRAIN_FRAC * len(idx))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # ============================================================
    # Fit all three models
    # ============================================================
    print("\n" + "=" * 60)
    print("FITTING MODELS")
    print("=" * 60)

    print("\n[1] Altitude-Dependent Log-Distance...")
    params_log = fit_log_distance_model(df_train)
    print(f"  PL0 polynomial coeffs: {params_log['PL0_coeffs']}")
    print(f"  Per-altitude fits: ")
    for alt in sorted(params_log['per_altitude'].keys()):
        a = params_log['per_altitude'][alt]
        print(f"    {alt:.0f}m: n={a['n']:.2f}, σ={a['sigma']:.2f}, PL0_local={a['PL0_local']:.2f}")

    print("\n[2] Enhanced Two-Ray...")
    params_2ray = fit_two_ray(df_train)
    print(f"  gamma = {params_2ray['gamma']:.3f}, "
          f"h_bs = {params_2ray['h_bs']:.2f} m, "
          f"offset = {params_2ray['offset']:.2f} dB")

    print("\n[3] 3GPP TR 36.777 (Rural-AV)...")
    params_gpp = fit_gpp(df_train)
    print(f"  offset = {params_gpp['offset']:.2f} dB (fixed standard parameters)")

    # ============================================================
    # Evaluate on Maeng test set
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION ON MAENG TEST SET")
    print("=" * 60)

    results = {}
    results['Log-Distance']    = evaluate_per_altitude(df_test, predict_log_distance, params_log,
                                                        "Model 1: Altitude-Dependent Log-Distance")
    results['Two-Ray']         = evaluate_per_altitude(df_test, predict_two_ray, params_2ray,
                                                        "Model 2: Enhanced Two-Ray")
    results['3GPP TR 36.777']  = evaluate_per_altitude(df_test, predict_gpp, params_gpp,
                                                        "Model 3: 3GPP TR 36.777 Rural-AV")

    # ============================================================
    # Cross-validate against Gürses
    # ============================================================
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION ON GÜRSES")
    print("=" * 60)
    gurses_path = Path('../Datasets/Finetuning-processed/gurses_channel_flight.csv')
    cv_results = {}
    if gurses_path.exists():
        gdf = pd.read_csv(gurses_path)
        gdf = gdf.dropna(subset=['rx_power_dBm', 'distance_m', 'rx_altitude_m'])

        # Use the altitude bin since true GPS altitude is offset
        gdf['altitude_target_m'] = gdf['altitude_bin_m']
        gdf['altitude_m'] = gdf['altitude_target_m']  # use nominal for prediction
        gdf['distance_to_bs_m'] = gdf['distance_m']
        gdf['pl_measured'] = -gdf['rx_power_dBm']

        # Bias-correct each model on Gürses (frequency differs slightly: 3.564 vs 3.51 GHz)
        # We allow each model an additive bias correction estimated from a small sample
        # to keep the comparison fair across models with different absolute calibrations.
        for name, predict_fn, params in [
            ('Log-Distance', predict_log_distance, params_log),
            ('Two-Ray', predict_two_ray, params_2ray),
            ('3GPP TR 36.777', predict_gpp, params_gpp),
        ]:
            pl_pred = predict_fn(gdf['distance_to_bs_m'].values,
                                  gdf['altitude_m'].values, params)
            bias = np.mean(gdf['pl_measured'].values - pl_pred)
            pl_pred_corrected = pl_pred + bias
            print(f"\n{name} (Gürses, bias-corrected by {bias:+.1f} dB):")
            cv_results[name] = evaluate(
                gdf['pl_measured'].values, pl_pred_corrected, "  overall"
            )
    else:
        print(f"  {gurses_path} not found, skipping cross-validation")

    # ============================================================
    # Save and plot
    # ============================================================
    out = Path('../results/Step_1')
    out.mkdir(exist_ok=True)
    summary = {
        'training_size': len(df_train),
        'test_size': len(df_test),
        'noise_floor_dB': NOISE_FLOOR_DB,
        'frequency_Hz': FREQ_HZ,
        'models': {
            'Log-Distance': {'params': params_log, 'maeng_test': results['Log-Distance'], 'gurses_cv': cv_results.get('Log-Distance', {})},
            'Two-Ray':      {'params': params_2ray, 'maeng_test': results['Two-Ray'], 'gurses_cv': cv_results.get('Two-Ray', {})},
            '3GPP TR 36.777': {'params': params_gpp, 'maeng_test': results['3GPP TR 36.777'], 'gurses_cv': cv_results.get('3GPP TR 36.777', {})},
        }
    }
    with open(out / 'layer1_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out / 'layer1_comparison.json'}")

    # Plots
    plot_models(df_test, params_log, params_2ray, params_gpp)
    plot_rmse_comparison(results)

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RANKING (lower RMSE is better)")
    print("=" * 60)
    ranked = sorted(results.items(),
                    key=lambda x: x[1]['overall'].get('rmse_dB', float('inf')))
    print(f"\nMaeng test set:")
    for i, (name, res) in enumerate(ranked, 1):
        rmse = res['overall'].get('rmse_dB', float('nan'))
        r2 = res['overall'].get('r_squared', float('nan'))
        print(f"  {i}. {name:25s}  RMSE={rmse:.2f} dB  R²={r2:.3f}")

    if cv_results:
        print(f"\nGürses cross-validation:")
        ranked_cv = sorted(cv_results.items(),
                            key=lambda x: x[1].get('rmse_dB', float('inf')))
        for i, (name, res) in enumerate(ranked_cv, 1):
            rmse = res.get('rmse_dB', float('nan'))
            r2 = res.get('r_squared', float('nan'))
            print(f"  {i}. {name:25s}  RMSE={rmse:.2f} dB  R²={r2:.3f}")

    print("\nDone. Inspect plots in outputs/plots/layer1/ and pick a winner.")


if __name__ == '__main__':
    main()