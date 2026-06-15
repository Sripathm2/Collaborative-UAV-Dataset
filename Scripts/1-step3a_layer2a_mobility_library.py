"""
step3a_layer2a_mobility_library.py — Layer 2a: Mobility Pattern Library.

Builds a library of calibrated mobility generators from AFAR flights.
Uses hand-coded per-flight category assignments validated visually.

Categories:
  - spiral:        301 (×3), 309 (×2)
  - grid:          328 (×3)
  - hover_transit: 288 (×3)
  - random:        300 (×3)

Usage:
    python step3a_layer2a_mobility_library.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial.distance import jensenshannon


SAMPLE_DT = 0.2
RNG_SEED = 42
TRAIN_FRAC = 0.7
DEG_TO_M = 1.113195e5

BASE_PROCESSED = Path('../Datasets/Finetuning-processed/')
if not BASE_PROCESSED.exists():
    BASE_PROCESSED = Path('processed')

OUT_DIR = Path('../results/Step_1')
PLOT_DIR = OUT_DIR / 'plots' / 'layer2a'

# Hand-coded categories (validated from cluster_trajectories.png)
FLIGHT_CATEGORIES = {
    '301_testbed_loc1': 'spiral',
    '301_testbed_loc2': 'spiral',
    '301_testbed_loc3': 'spiral',
    '309_testbed_loc2': 'spiral',
    '309_testbed_loc3': 'spiral',
    '328_testbed_loc1': 'grid',
    '328_testbed_loc2': 'grid',
    '328_testbed_loc3': 'grid',
    '288_testbed_loc1': 'hover_transit',
    '288_testbed_loc2': 'hover_transit',
    '288_testbed_loc3': 'hover_transit',
    '300_testbed_loc1': 'random',
    '300_testbed_loc2': 'random',
    '300_testbed_loc3': 'random',
}


# ============================================================
# Trajectory utils
# ============================================================

def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * DEG_TO_M * np.cos(np.radians(lat0))
    y = (lat - lat0) * DEG_TO_M
    return x, y


def resample_flight(df, dt=SAMPLE_DT):
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    if len(df) < 50:
        return None
    t = df['timestamp'].values
    t_rel = t - t[0]
    lat0 = float(df['latitude'].iloc[0])
    lon0 = float(df['longitude'].iloc[0])
    x, y = latlon_to_xy(df['latitude'].values, df['longitude'].values, lat0, lon0)
    if 'altitude_rel_m' in df.columns and df['altitude_rel_m'].notna().any():
        z = df['altitude_rel_m'].values
    else:
        z = df['altitude_m'].values - df['altitude_m'].iloc[0]
    t_uniform = np.arange(0, t_rel[-1], dt)
    if len(t_uniform) < 100:
        return None
    x_u = np.interp(t_uniform, t_rel, x)
    y_u = np.interp(t_uniform, t_rel, y)
    z_u = np.interp(t_uniform, t_rel, z)
    vx = np.diff(x_u) / dt
    vy = np.diff(y_u) / dt
    vz = np.diff(z_u) / dt
    return pd.DataFrame({
        't': t_uniform[:-1], 'x': x_u[:-1], 'y': y_u[:-1], 'z': z_u[:-1],
        'vx': vx, 'vy': vy, 'vz': vz,
    })


# ============================================================
# Metrics
# ============================================================

def jsd_distribution(real_values, sim_values, bins=50):
    if len(real_values) < 5 or len(sim_values) < 5:
        return 1.0
    lo = float(min(real_values.min(), sim_values.min()))
    hi = float(max(real_values.max(), sim_values.max()))
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p_r, _ = np.histogram(real_values, bins=edges, density=True)
    p_s, _ = np.histogram(sim_values, bins=edges, density=True)
    p_r = p_r / (p_r.sum() + 1e-12)
    p_s = p_s / (p_s.sum() + 1e-12)
    return float(jensenshannon(p_r, p_s, base=np.e) ** 2)


def kl_divergence_2d(p, q):
    p = p.flatten() + 1e-12
    q = q.flatten() + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def spatial_hist(traj, x_edges, y_edges):
    h, _, _ = np.histogram2d(traj['x'].values, traj['y'].values,
                             bins=[x_edges, y_edges])
    return h / (h.sum() + 1e-12)


def make_grid_edges(traj, n=40):
    pad = max(traj['x'].max() - traj['x'].min(),
              traj['y'].max() - traj['y'].min(), 1.0) * 0.1
    xe = np.linspace(traj['x'].min() - pad, traj['x'].max() + pad, n + 1)
    ye = np.linspace(traj['y'].min() - pad, traj['y'].max() + pad, n + 1)
    return xe, ye


# ============================================================
# Generators
# ============================================================

def gen_spiral(cx, cy, alt, r0, pitch, speed, n_steps, dt=SAMPLE_DT):
    omega = speed / max(r0, 1.0)
    t_arr = np.arange(n_steps) * dt
    r = r0 + pitch * t_arr / (2 * np.pi)
    theta = omega * t_arr
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.full(n_steps, alt)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.zeros(n_steps)
    return pd.DataFrame({'t': t_arr, 'x': x, 'y': y, 'z': z,
                         'vx': vx, 'vy': vy, 'vz': vz})


def fit_spiral(real_traj):
    cx = float(real_traj['x'].mean())
    cy = float(real_traj['y'].mean())
    alt = float(real_traj['z'].mean())
    n_steps = len(real_traj)
    real_speed = np.sqrt(real_traj['vx']**2 + real_traj['vy']**2).values
    xe, ye = make_grid_edges(real_traj)
    real_hist = spatial_hist(real_traj, xe, ye)

    def obj(p):
        r0, pitch, speed = p
        try:
            sim = gen_spiral(cx, cy, alt, r0, pitch, speed, n_steps)
            sim_speed = np.sqrt(sim['vx']**2 + sim['vy']**2).values
            jsd_speed = jsd_distribution(real_speed, sim_speed)
            sim_hist = spatial_hist(sim, xe, ye)
            kl = kl_divergence_2d(real_hist, sim_hist)
            return jsd_speed + 0.1 * kl
        except Exception:
            return 10.0

    bounds = [(2.0, 50.0), (1.0, 50.0), (0.5, 10.0)]
    res = differential_evolution(obj, bounds, maxiter=30, seed=RNG_SEED, tol=1e-3)
    return {'type': 'spiral', 'center_x': cx, 'center_y': cy, 'altitude': alt,
            'r0': float(res.x[0]), 'pitch': float(res.x[1]),
            'speed': float(res.x[2]), 'fit_objective': float(res.fun)}


def gen_grid(x0, y0, x1, y1, alt, lane_spacing, speed, n_steps, dt=SAMPLE_DT):
    waypoints = []
    y = y0
    direction = 1
    while y <= y1:
        if direction == 1:
            waypoints += [(x0, y), (x1, y)]
        else:
            waypoints += [(x1, y), (x0, y)]
        y += lane_spacing
        direction *= -1

    px, py, ts = [], [], []
    t = 0.0
    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i]); p1 = np.array(waypoints[i + 1])
        dist = np.linalg.norm(p1 - p0)
        n_seg = max(int(dist / max(speed * dt, 1e-3)), 1)
        for k in range(n_seg):
            frac = k / n_seg
            pos = p0 + frac * (p1 - p0)
            px.append(pos[0]); py.append(pos[1]); ts.append(t)
            t += dt
            if len(ts) >= n_steps:
                break
        if len(ts) >= n_steps:
            break
    while len(ts) < n_steps:
        px.append(px[-1] if px else 0)
        py.append(py[-1] if py else 0)
        ts.append(t); t += dt

    x = np.array(px[:n_steps]); y = np.array(py[:n_steps])
    z = np.full(n_steps, alt)
    vx = np.gradient(x, dt); vy = np.gradient(y, dt)
    vz = np.zeros(n_steps)
    return pd.DataFrame({'t': np.array(ts[:n_steps]),
                         'x': x, 'y': y, 'z': z,
                         'vx': vx, 'vy': vy, 'vz': vz})


def fit_grid(real_traj):
    x0 = float(real_traj['x'].min()); x1 = float(real_traj['x'].max())
    y0 = float(real_traj['y'].min()); y1 = float(real_traj['y'].max())
    alt = float(real_traj['z'].mean())
    n_steps = len(real_traj)
    real_speed = np.sqrt(real_traj['vx']**2 + real_traj['vy']**2).values
    xe, ye = make_grid_edges(real_traj)
    real_hist = spatial_hist(real_traj, xe, ye)

    def obj(p):
        lane_spacing, speed = p
        try:
            sim = gen_grid(x0, y0, x1, y1, alt, lane_spacing, speed, n_steps)
            sim_speed = np.sqrt(sim['vx']**2 + sim['vy']**2).values
            jsd_speed = jsd_distribution(real_speed, sim_speed)
            sim_hist = spatial_hist(sim, xe, ye)
            kl = kl_divergence_2d(real_hist, sim_hist)
            return jsd_speed + 0.1 * kl
        except Exception:
            return 10.0

    bounds = [(3.0, max((y1 - y0) / 2, 5.0)), (0.5, 10.0)]
    res = differential_evolution(obj, bounds, maxiter=30, seed=RNG_SEED, tol=1e-3)
    return {'type': 'grid', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'altitude': alt, 'lane_spacing': float(res.x[0]),
            'speed': float(res.x[1]), 'fit_objective': float(res.fun)}


def gen_hover_transit(waypoints, alts, hover_dur, speed, n_steps, dt=SAMPLE_DT):
    px, py, pz, ts = [], [], [], []
    t = 0.0
    for i in range(len(waypoints)):
        n_hover = max(int(hover_dur / dt), 1)
        for _ in range(n_hover):
            px.append(waypoints[i][0]); py.append(waypoints[i][1])
            pz.append(alts[i] if i < len(alts) else alts[-1])
            ts.append(t); t += dt
            if len(ts) >= n_steps:
                break
        if len(ts) >= n_steps:
            break
        if i + 1 < len(waypoints):
            p0 = np.array(waypoints[i]); p1 = np.array(waypoints[i + 1])
            dist = np.linalg.norm(p1 - p0)
            n_seg = max(int(dist / max(speed * dt, 1e-3)), 1)
            for k in range(n_seg):
                frac = k / n_seg
                pos = p0 + frac * (p1 - p0)
                px.append(pos[0]); py.append(pos[1]); pz.append(alts[i])
                ts.append(t); t += dt
                if len(ts) >= n_steps:
                    break
            if len(ts) >= n_steps:
                break
    while len(ts) < n_steps:
        px.append(px[-1]); py.append(py[-1]); pz.append(pz[-1])
        ts.append(t); t += dt

    x = np.array(px[:n_steps]); y = np.array(py[:n_steps]); z = np.array(pz[:n_steps])
    vx = np.gradient(x, dt); vy = np.gradient(y, dt); vz = np.gradient(z, dt)
    return pd.DataFrame({'t': np.array(ts[:n_steps]),
                         'x': x, 'y': y, 'z': z,
                         'vx': vx, 'vy': vy, 'vz': vz})


def fit_hover_transit(real_traj):
    n = len(real_traj)
    n_steps = n
    idx = np.linspace(0, n - 1, 5, dtype=int)
    waypoints = [(float(real_traj['x'].iloc[i]),
                  float(real_traj['y'].iloc[i])) for i in idx]
    alts = [float(real_traj['z'].iloc[i]) for i in idx]
    real_speed = np.sqrt(real_traj['vx']**2 + real_traj['vy']**2).values
    xe, ye = make_grid_edges(real_traj)
    real_hist = spatial_hist(real_traj, xe, ye)

    def obj(p):
        hover_dur, speed = p
        try:
            sim = gen_hover_transit(waypoints, alts, hover_dur, speed, n_steps)
            sim_speed = np.sqrt(sim['vx']**2 + sim['vy']**2).values
            jsd_speed = jsd_distribution(real_speed, sim_speed)
            sim_hist = spatial_hist(sim, xe, ye)
            kl = kl_divergence_2d(real_hist, sim_hist)
            return jsd_speed + 0.1 * kl
        except Exception:
            return 10.0

    bounds = [(1.0, 30.0), (0.5, 10.0)]
    res = differential_evolution(obj, bounds, maxiter=30, seed=RNG_SEED, tol=1e-3)
    return {'type': 'hover_transit', 'waypoints': waypoints, 'altitudes': alts,
            'hover_duration': float(res.x[0]), 'speed': float(res.x[1]),
            'fit_objective': float(res.fun)}


def gen_gauss_markov(params, n_steps, dt=SAMPLE_DT, seed=0):
    rng = np.random.default_rng(seed)
    a_h = params['alpha_h']; vmh = params['v_mean_h']; vsh = params['v_std_h']
    a_v = params['alpha_v']; vmv = params['v_mean_v']; vsv = params['v_std_v']

    vx = np.zeros(n_steps); vy = np.zeros(n_steps); vz = np.zeros(n_steps)
    theta0 = rng.uniform(0, 2 * np.pi)
    vx[0] = vmh * np.cos(theta0); vy[0] = vmh * np.sin(theta0); vz[0] = vmv
    n_h = np.sqrt(max(1 - a_h**2, 0)) * vsh
    n_v = np.sqrt(max(1 - a_v**2, 0)) * vsv

    for t in range(n_steps - 1):
        speed_t = np.sqrt(vx[t]**2 + vy[t]**2)
        if speed_t > 1e-6:
            ux, uy = vx[t] / speed_t, vy[t] / speed_t
        else:
            theta = rng.uniform(0, 2 * np.pi)
            ux, uy = np.cos(theta), np.sin(theta)
        vx[t+1] = a_h * vx[t] + (1 - a_h) * vmh * ux + n_h * rng.standard_normal()
        vy[t+1] = a_h * vy[t] + (1 - a_h) * vmh * uy + n_h * rng.standard_normal()
        vz[t+1] = a_v * vz[t] + (1 - a_v) * vmv + n_v * rng.standard_normal()

    x = np.cumsum(vx) * dt; y = np.cumsum(vy) * dt; z = np.cumsum(vz) * dt
    return pd.DataFrame({'t': np.arange(n_steps) * dt,
                         'x': x, 'y': y, 'z': z,
                         'vx': vx, 'vy': vy, 'vz': vz})


def fit_gauss_markov(real_traj):
    n_steps = min(len(real_traj), 3000)
    real_speed = np.sqrt(real_traj['vx']**2 + real_traj['vy']**2).values

    def obj(p):
        params = {'alpha_h': p[0], 'v_mean_h': p[1], 'v_std_h': p[2],
                  'alpha_v': p[3], 'v_mean_v': p[4], 'v_std_v': p[5]}
        try:
            sim = gen_gauss_markov(params, n_steps, seed=0)
            sim_speed = np.sqrt(sim['vx']**2 + sim['vy']**2).values
            return jsd_distribution(real_speed, sim_speed)
        except Exception:
            return 10.0

    bounds = [(0.5, 0.99), (0.5, 10.0), (0.3, 5.0),
              (0.5, 0.99), (-1.0, 1.0), (0.05, 2.0)]
    res = differential_evolution(obj, bounds, maxiter=30, seed=RNG_SEED, tol=1e-3)
    return {'type': 'random', 'alpha_h': float(res.x[0]),
            'v_mean_h': float(res.x[1]), 'v_std_h': float(res.x[2]),
            'alpha_v': float(res.x[3]), 'v_mean_v': float(res.x[4]),
            'v_std_v': float(res.x[5]), 'dt_fitted': SAMPLE_DT,
            'fit_objective': float(res.fun)}


GENERATOR_FITTERS = {
    'spiral': fit_spiral,
    'grid': fit_grid,
    'hover_transit': fit_hover_transit,
    'random': fit_gauss_markov,
}

GENERATOR_SIMS = {
    'spiral': lambda p, n: gen_spiral(p['center_x'], p['center_y'], p['altitude'],
                                       p['r0'], p['pitch'], p['speed'], n),
    'grid': lambda p, n: gen_grid(p['x0'], p['y0'], p['x1'], p['y1'],
                                   p['altitude'], p['lane_spacing'], p['speed'], n),
    'hover_transit': lambda p, n: gen_hover_transit(p['waypoints'], p['altitudes'],
                                                      p['hover_duration'], p['speed'], n),
    'random': lambda p, n: gen_gauss_markov(p, n),
}


# ============================================================
# Per-category fitting and evaluation
# ============================================================

def fit_per_category(flights, assignments):
    library = {}
    rng = np.random.default_rng(RNG_SEED)

    by_cat = {}
    for fid, cat in assignments.items():
        if fid in flights:
            by_cat.setdefault(cat, []).append(fid)

    for category, flight_ids in by_cat.items():
        flight_ids = sorted(flight_ids)
        rng.shuffle(flight_ids)

        if len(flight_ids) == 1:
            train_ids = flight_ids
            test_ids = []
        else:
            n_train = max(1, int(TRAIN_FRAC * len(flight_ids)))
            train_ids = flight_ids[:n_train]
            test_ids = flight_ids[n_train:]

        print(f"\n[{category}] {len(flight_ids)} flights "
              f"(train={len(train_ids)}, test={len(test_ids)})")
        print(f"  Train: {train_ids}")
        print(f"  Test:  {test_ids}")

        train_lens = [(fid, len(flights[fid])) for fid in train_ids]
        train_lens.sort(key=lambda x: -x[1])
        anchor_id = train_lens[0][0]
        anchor_traj = flights[anchor_id]

        fitter = GENERATOR_FITTERS[category]
        params = fitter(anchor_traj)
        params['category'] = category
        params['anchor_flight'] = anchor_id
        params['train_flights'] = train_ids
        params['test_flights'] = test_ids
        params['n_members'] = len(flight_ids)
        print(f"  Fit on anchor {anchor_id}: "
              f"objective={params['fit_objective']:.4f}")

        test_metrics = []
        for tid in test_ids:
            real = flights[tid]
            n_steps = len(real)
            sim = GENERATOR_SIMS[category](params, n_steps)
            real_speed = np.sqrt(real['vx']**2 + real['vy']**2).values
            sim_speed = np.sqrt(sim['vx']**2 + sim['vy']**2).values
            jsd_s = jsd_distribution(real_speed, sim_speed)
            xe, ye = make_grid_edges(real)
            kl_sp = kl_divergence_2d(spatial_hist(real, xe, ye),
                                       spatial_hist(sim, xe, ye))
            test_metrics.append({'flight_id': tid, 'jsd_speed': jsd_s,
                                 'kl_spatial': kl_sp})
        params['test_metrics'] = test_metrics
        if test_metrics:
            mean_jsd = np.mean([m['jsd_speed'] for m in test_metrics])
            mean_kl = np.mean([m['kl_spatial'] for m in test_metrics])
            print(f"  Test (n={len(test_metrics)}): "
                  f"JSD speed={mean_jsd:.4f}, KL spatial={mean_kl:.3f}")

        library[category] = params

    return library


# ============================================================
# Plotting
# ============================================================

def plot_trajectories(flights, assignments):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    flight_ids = sorted(flights.keys())
    n = len(flight_ids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    cat_colors = {
        'spiral': '#1f77b4',
        'grid': '#2ca02c',
        'hover_transit': '#d62728',
        'random': '#9467bd',
    }

    for ax, fid in zip(axes, flight_ids):
        traj = flights[fid]
        cat = assignments.get(fid, 'unknown')
        color = cat_colors.get(cat, 'gray')
        ax.plot(traj['x'], traj['y'], linewidth=0.8, color=color)
        ax.scatter([traj['x'].iloc[0]], [traj['y'].iloc[0]],
                    c='green', s=20, zorder=5)
        ax.scatter([traj['x'].iloc[-1]], [traj['y'].iloc[-1]],
                    c='red', s=20, zorder=5)
        ax.set_title(f"{fid}\n{cat}", fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'cluster_trajectories.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOT_DIR / 'cluster_trajectories.png'}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("STEP 3a: Layer 2a - Mobility Pattern Library")
    print("=" * 60)

    afar_path = BASE_PROCESSED / 'afar_testbed.csv'
    if not afar_path.exists():
        print(f"ERROR: {afar_path} not found")
        return
    afar = pd.read_csv(afar_path)
    print(f"\nLoaded AFAR testbed: {len(afar)} rows, "
          f"{afar['flight_id'].nunique()} flights")

    flights = {}
    for fid, group in afar.groupby('flight_id'):
        traj = resample_flight(group)
        if traj is not None:
            flights[fid] = traj
    print(f"Usable flights: {len(flights)}")

    print("\nHand-coded category assignments:")
    by_cat = {}
    for fid, cat in FLIGHT_CATEGORIES.items():
        if fid in flights:
            by_cat.setdefault(cat, []).append(fid)
    for cat, fids in sorted(by_cat.items()):
        print(f"  {cat}: {len(fids)} flights — {sorted(fids)}")

    print("\nGenerating trajectory plot...")
    plot_trajectories(flights, FLIGHT_CATEGORIES)

    print("\n" + "=" * 60)
    print("FITTING GENERATORS PER CATEGORY")
    print("=" * 60)
    library = fit_per_category(flights, FLIGHT_CATEGORIES)

    OUT_DIR.mkdir(exist_ok=True)
    output = {
        'config': {
            'sample_dt': SAMPLE_DT,
            'train_frac': TRAIN_FRAC,
            'n_flights': len(flights),
            'method': 'hand-coded category assignment from visual inspection',
        },
        'flight_categories': FLIGHT_CATEGORIES,
        'library': library,
    }
    with open(OUT_DIR / 'layer2a_mobility_library.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'layer2a_mobility_library.json'}")

    print("\n" + "=" * 60)
    print("LIBRARY SUMMARY")
    print("=" * 60)
    for category, params in library.items():
        print(f"\n[{category}] ({params['n_members']} flights)")
        print(f"  Anchor: {params['anchor_flight']}")
        if params['test_metrics']:
            mean_jsd = np.mean([m['jsd_speed'] for m in params['test_metrics']])
            mean_kl = np.mean([m['kl_spatial'] for m in params['test_metrics']])
            print(f"  Test (n={len(params['test_metrics'])}): "
                  f"JSD speed={mean_jsd:.4f}, KL spatial={mean_kl:.3f}")
        else:
            print(f"  No test flights")

    print("\nDone. Library is ready for Layer 2b.")


if __name__ == '__main__':
    main()