"""
simlib/mobility.py

Mobility generator for UAV swarm simulation. Five mission types:

  spiral         Archimedean spiral at fitted altitude. Params: r0, pitch,
                 speed, altitude. Continuous-time queryable.

  grid           Boustrophedon (back-and-forth lanes) covering a rectangle
                 fitted corners (x0,y0)-(x1,y1). Params: lane_spacing, speed,
                 altitude. Continuous-time queryable.

  hover_transit  Sequence of waypoints. Hover hover_duration at each, then
                 transit at speed to next. Per-waypoint altitudes allowed.
                 Continuous-time queryable.

  random         Gauss-Markov random walk in 3D. Params fit at dt=0.2s; we
                 rescale alpha -> alpha^(dt/0.2s) to step at any dt. Reflects
                 on 6 grid faces. STATEFUL.

  replay         Position playback from a pre-recorded trajectory CSV.
                 Used by Layer-3 validation. Library JSON entry not required;
                 callers inject _ReplayMission instances directly into
                 MobilityGenerator.uavs.

All missions accept an `origin_xy` that re-anchors the fitted pattern into
the global coordinate system - the fitted shape is preserved, just translated.
For spiral the origin is the spiral center; for grid it is the (x0,y0) corner;
for hover_transit the first waypoint; for random the starting position; for
replay an additive offset on (x, y).

Query API: `position(uav_id, t_abs)` returns (x, y, z) in meters. For the
three deterministic missions and replay this is a pure function of t_abs.
For random, it steps state forward from the last query - you must call it in
monotonically non-decreasing time order.

Fit library is loaded from layer2a_mobility_library.json.
"""

import bisect
import csv
import json
import math
import random


# ===================== individual mission classes =====================

class _SpiralMission:
    """
    Archimedean spiral r(theta) = r0 + pitch * theta, traced at constant
    tangential speed. Origin of the simulation frame maps to spiral center.
    """

    def __init__(self, params, origin_xy, altitude_override=None):
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        self.r0 = float(params['r0'])
        self.pitch = float(params['pitch'])
        self.speed = float(params['speed'])
        self.altitude = float(
            altitude_override if altitude_override is not None else params['altitude']
        )

    def position(self, t_abs):
        if t_abs < 0:
            t_abs = 0.0
        s = self.speed * t_abs  # arc length traversed
        # Arc-length approximation s ≈ r0*θ + 0.5*pitch*θ^2 (valid for r >> pitch)
        # Inversion:  θ = (-r0 + sqrt(r0² + 2·pitch·s)) / pitch
        if abs(self.pitch) > 1e-9:
            disc = self.r0 * self.r0 + 2.0 * self.pitch * s
            disc = max(disc, 0.0)
            theta = (-self.r0 + math.sqrt(disc)) / self.pitch
        else:
            theta = s / max(self.r0, 1e-9)
        r = self.r0 + self.pitch * theta
        return (
            self.ox + r * math.cos(theta),
            self.oy + r * math.sin(theta),
            self.altitude,
        )


class _GridMission:
    """
    Boustrophedon coverage of rectangle defined by fitted corners
    (fx0,fy0)-(fx1,fy1). Lanes are parallel to the x-axis direction of the
    fit, stacking in the y direction with lane_spacing. Alternating lane
    directions. Hovers at final corner once total path length is reached.
    """

    def __init__(self, params, origin_xy, altitude_override=None):
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        self.fx0 = float(params['x0'])
        self.fy0 = float(params['y0'])
        self.fx1 = float(params['x1'])
        self.fy1 = float(params['y1'])
        self.lane_spacing = float(params['lane_spacing'])
        self.speed = float(params['speed'])
        self.altitude = float(
            altitude_override if altitude_override is not None else params['altitude']
        )

        self.dx_sign = 1.0 if self.fx1 >= self.fx0 else -1.0
        self.dy_sign = 1.0 if self.fy1 >= self.fy0 else -1.0
        self.x_span = abs(self.fx1 - self.fx0)
        self.y_span = abs(self.fy1 - self.fy0)

        if self.lane_spacing <= 0:
            raise ValueError("grid: lane_spacing must be positive")

        self.n_lanes = max(1, int(self.y_span // self.lane_spacing) + 1)
        # total path length: n_lanes covers x_span + (n_lanes-1) transits
        self.total_path = (
            self.n_lanes * self.x_span
            + max(0, self.n_lanes - 1) * self.lane_spacing
        )

    def position(self, t_abs):
        if t_abs < 0:
            t_abs = 0.0
        s = min(self.speed * t_abs, self.total_path)

        # Per-lane structure: each cycle (except last) = x_span lane + lane_spacing transit
        per_cycle = self.x_span + self.lane_spacing

        # Find lane index
        lane_idx = 0
        s_left = s
        while lane_idx < self.n_lanes - 1 and s_left > per_cycle:
            s_left -= per_cycle
            lane_idx += 1

        # s_left is now within current lane or its trailing transit
        on_transit = (lane_idx < self.n_lanes - 1) and (s_left > self.x_span)

        # y coordinate of the current lane (in fitted frame)
        y_lane = self.fy0 + self.dy_sign * lane_idx * self.lane_spacing

        # Lane direction alternates
        lane_dir = self.dx_sign if (lane_idx % 2 == 0) else -self.dx_sign

        if not on_transit:
            # Position along the lane
            if lane_dir > 0:
                x_local = self.fx0 + s_left
            else:
                x_local = self.fx0 + self.dx_sign * self.x_span - s_left
            y_local = y_lane
        else:
            # End of lane, transitioning sideways to next lane's start
            s_on_trans = s_left - self.x_span
            # End-of-lane x
            if lane_dir > 0:
                x_local = self.fx0 + self.dx_sign * self.x_span
            else:
                x_local = self.fx0
            y_next = y_lane + self.dy_sign * self.lane_spacing
            frac = s_on_trans / self.lane_spacing
            y_local = y_lane + (y_next - y_lane) * frac

        # Translate into simulation frame: (fx0, fy0) maps to (ox, oy)
        return (
            self.ox + (x_local - self.fx0),
            self.oy + (y_local - self.fy0),
            self.altitude,
        )


class _HoverTransitMission:
    """
    Sequence of waypoints. At each waypoint, hover for hover_duration; then
    transit at speed to next waypoint. Per-waypoint altitude is supported
    (absolute). After the last waypoint, hover there indefinitely.
    First waypoint is treated as the origin for translation.
    """

    def __init__(self, params, origin_xy, altitude_override=None):
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        wps = params['waypoints']
        alts = params['altitudes']
        if len(wps) != len(alts):
            raise ValueError("hover_transit: waypoints and altitudes length mismatch")
        self.hover_dur = float(params['hover_duration'])
        self.speed = float(params['speed'])

        # If altitude override given, apply uniformly to all waypoints
        if altitude_override is not None:
            alts = [float(altitude_override)] * len(wps)

        # First waypoint is the anchor: shift so (wp[0].x, wp[0].y) -> (ox, oy)
        wp0x, wp0y = float(wps[0][0]), float(wps[0][1])
        self.points = [
            (self.ox + (float(wp[0]) - wp0x),
             self.oy + (float(wp[1]) - wp0y),
             float(alt))
            for wp, alt in zip(wps, alts)
        ]

        # Precompute segments: list of (type, t_start, t_end, p_start, p_end)
        self.segments = []
        t = 0.0
        for k in range(len(self.points)):
            p = self.points[k]
            self.segments.append(('hover', t, t + self.hover_dur, p, p))
            t += self.hover_dur
            if k < len(self.points) - 1:
                q = self.points[k + 1]
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
                transit_t = dist / max(self.speed, 1e-9)
                self.segments.append(('transit', t, t + transit_t, p, q))
                t += transit_t
        self.total_duration = t

    def position(self, t_abs):
        if t_abs < 0:
            t_abs = 0.0
        if t_abs >= self.total_duration:
            return self.segments[-1][4]
        # linear scan (few segments, not a hot loop concern)
        for seg_type, t0, t1, p0, p1 in self.segments:
            if t0 <= t_abs <= t1:
                if seg_type == 'hover':
                    return p0
                frac = (t_abs - t0) / max(t1 - t0, 1e-9)
                return tuple(p0[i] + frac * (p1[i] - p0[i]) for i in range(3))
        return self.segments[-1][4]  # unreachable guard


class _RandomWalkMission:
    """
    3D Gauss-Markov random walk.
    Fitted at dt=0.2s; rescales alpha to any query dt via alpha^(dt/dt_fitted).

    Horizontal: v_x and v_y are independent GM(alpha_h, 0, v_std_h) — zero
                mean drift, consistent with E[||v_h||] = v_std_h*sqrt(pi/2)
                matching fitted v_mean_h for (u1,b1)=1.93*sqrt(pi/2)≈2.42 ~ 2.50.
                Documented approximation.
    Vertical:   v_z is GM(alpha_v, v_mean_v, v_std_v). v_mean_v < 0 means
                a statistical drift downward, bounded by altitude floor.

    Reflects on 6 grid faces (x, y in [0, grid], z in [alt_min, alt_max]),
    flipping the relevant velocity component on impact.
    """

    def __init__(self, params, origin_xy, grid_size_m, alt_min, alt_max,
                 initial_z=None, rng=None):
        self.ox = float(origin_xy[0])
        self.oy = float(origin_xy[1])
        self.grid = float(grid_size_m)
        self.alt_min = float(alt_min)
        self.alt_max = float(alt_max)
        self.rng = rng if rng is not None else random.Random()

        self.alpha_h = float(params['alpha_h'])
        self.v_std_h = float(params['v_std_h'])
        self.alpha_v = float(params['alpha_v'])
        self.v_mean_v = float(params['v_mean_v'])
        self.v_std_v = float(params['v_std_v'])
        self.dt_fit = float(params['dt_fitted'])

        z0 = float(initial_z) if initial_z is not None else (self.alt_min + self.alt_max) / 2.0
        z0 = min(max(z0, self.alt_min), self.alt_max)
        self.pos = [self.ox, self.oy, z0]
        self.vel = [
            self.rng.gauss(0.0, self.v_std_h),
            self.rng.gauss(0.0, self.v_std_h),
            self.rng.gauss(self.v_mean_v, self.v_std_v),
        ]
        self.last_t = 0.0

    def _alpha_scaled(self, alpha_base, dt):
        # alpha_dt = alpha_base^(dt/dt_fit). Guards on edges.
        alpha_base = max(min(alpha_base, 0.999999), 0.0)
        n = dt / self.dt_fit
        return alpha_base ** n

    def position(self, t_abs):
        if t_abs <= self.last_t:
            return tuple(self.pos)
        dt = t_abs - self.last_t

        a_h = self._alpha_scaled(self.alpha_h, dt)
        a_v = self._alpha_scaled(self.alpha_v, dt)
        innov_h = math.sqrt(max(0.0, 1.0 - a_h * a_h)) * self.v_std_h
        innov_v = math.sqrt(max(0.0, 1.0 - a_v * a_v)) * self.v_std_v

        # GM update for velocity
        self.vel[0] = a_h * self.vel[0] + innov_h * self.rng.gauss(0.0, 1.0)
        self.vel[1] = a_h * self.vel[1] + innov_h * self.rng.gauss(0.0, 1.0)
        self.vel[2] = (a_v * self.vel[2]
                       + (1.0 - a_v) * self.v_mean_v
                       + innov_v * self.rng.gauss(0.0, 1.0))

        # Advance position
        new_pos = [self.pos[i] + self.vel[i] * dt for i in range(3)]

        # Reflect horizontal on [0, grid]
        for i in (0, 1):
            if new_pos[i] < 0.0:
                new_pos[i] = -new_pos[i]
                self.vel[i] = -self.vel[i]
            elif new_pos[i] > self.grid:
                new_pos[i] = 2.0 * self.grid - new_pos[i]
                self.vel[i] = -self.vel[i]
        # Reflect vertical on [alt_min, alt_max]
        if new_pos[2] < self.alt_min:
            new_pos[2] = 2.0 * self.alt_min - new_pos[2]
            self.vel[2] = -self.vel[2]
        elif new_pos[2] > self.alt_max:
            new_pos[2] = 2.0 * self.alt_max - new_pos[2]
            self.vel[2] = -self.vel[2]

        self.pos = new_pos
        self.last_t = t_abs
        return tuple(new_pos)


class _ReplayMission:
    """
    Position playback from a pre-recorded trajectory CSV.
    Used by Layer-3 validation: drive the sim physics with measured AERPAW
    / AADM flight paths so the resulting SNR/RSS can be compared
    point-for-point to measurements.

    Trajectory CSV expected columns (header row required):
        time_rel_s, x_m, y_m, z_m
    Rows MUST be sorted by time_rel_s ascending. Linear interpolation
    between samples. Before first sample -> first row. After last -> last.
    """

    def __init__(self, params, origin_xy):
        path = params['trajectory_csv']
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])

        ts, xs, ys, zs = [], [], [], []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts.append(float(row['time_rel_s']))
                xs.append(float(row['x_m']))
                ys.append(float(row['y_m']))
                zs.append(float(row['z_m']))
        if not ts:
            raise RuntimeError(f"empty trajectory CSV: {path}")
        for i in range(1, len(ts)):
            if ts[i] < ts[i - 1]:
                raise ValueError(
                    f"trajectory not monotonic at row {i}: "
                    f"t={ts[i]} < prev {ts[i-1]}"
                )

        self.ts = ts
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.n = len(ts)

    def position(self, t_abs):
        if t_abs <= self.ts[0]:
            return (self.ox + self.xs[0], self.oy + self.ys[0], self.zs[0])
        if t_abs >= self.ts[-1]:
            return (self.ox + self.xs[-1], self.oy + self.ys[-1], self.zs[-1])
        hi = bisect.bisect_right(self.ts, t_abs)
        lo = hi - 1
        t0, t1 = self.ts[lo], self.ts[hi]
        a = (t_abs - t0) / (t1 - t0) if t1 > t0 else 0.0
        x = self.xs[lo] + a * (self.xs[hi] - self.xs[lo])
        y = self.ys[lo] + a * (self.ys[hi] - self.ys[lo])
        z = self.zs[lo] + a * (self.zs[hi] - self.zs[lo])
        return (self.ox + x, self.oy + y, z)


# ===================== facade =====================

class MobilityGenerator:
    """
    Loads Layer 2 library and creates per-UAV mission instances.
    Typical usage inside Topo.py:

        mob = MobilityGenerator('.../layer2a_mobility_library.json', rng=rng)
        mob.create_uav('d1', 'spiral',        origin_xy=(300, 300))
        mob.create_uav('d2', 'random',        origin_xy=(500, 500),
                       grid_size_m=1000, alt_min=5, alt_max=150)
        ...
        x, y, z = mob.position('d1', t_abs=15.0)

    For 'replay' missions, the library JSON entry is NOT required. Callers
    inject the mission directly:
        mob.uavs['d1'] = _ReplayMission(
            params={'trajectory_csv': '/path/to/traj.csv'},
            origin_xy=(0.0, 0.0),
        )
    """

    SUPPORTED = ('spiral', 'grid', 'hover_transit', 'random', 'replay')

    def __init__(self, library_json_path, rng=None):
        with open(library_json_path, 'r') as f:
            data = json.load(f)
        self.library = data['library']
        self.rng = rng if rng is not None else random.Random()
        self.uavs = {}  # uav_id -> mission instance

    def create_uav(self, uav_id, mission_type, origin_xy,
                   grid_size_m=1000.0, alt_min=5.0, alt_max=150.0,
                   altitude_override=None, initial_z=None,
                   trajectory_csv=None):
        if mission_type not in self.SUPPORTED:
            raise ValueError(
                f"Unknown mission_type {mission_type!r}. "
                f"Supported: {self.SUPPORTED}"
            )

        # 'replay' bypasses the library — params come from caller via
        # trajectory_csv kwarg rather than the fitted JSON.
        if mission_type == 'replay':
            if trajectory_csv is None:
                raise ValueError(
                    "mission_type='replay' requires trajectory_csv=<path>"
                )
            m = _ReplayMission({'trajectory_csv': trajectory_csv}, origin_xy)
            self.uavs[uav_id] = m
            return m

        if mission_type not in self.library:
            raise KeyError(f"Library missing entry for mission {mission_type!r}")
        params = self.library[mission_type]

        # For fixed-altitude missions (spiral, grid), the fitted altitude
        # may be outside [alt_min, alt_max]. Clamp it so UAVs stay within
        # the calibrated pathloss range. Only applies when no explicit
        # altitude_override is given by the caller.
        if altitude_override is None and mission_type in ('spiral', 'grid'):
            fitted_alt = float(params.get('altitude', (alt_min + alt_max) / 2))
            if fitted_alt < alt_min or fitted_alt > alt_max:
                altitude_override = max(alt_min, min(alt_max, fitted_alt))

        if mission_type == 'spiral':
            m = _SpiralMission(params, origin_xy, altitude_override)
        elif mission_type == 'grid':
            m = _GridMission(params, origin_xy, altitude_override)
        elif mission_type == 'hover_transit':
            m = _HoverTransitMission(params, origin_xy, altitude_override)
        elif mission_type == 'random':
            m = _RandomWalkMission(
                params, origin_xy, grid_size_m, alt_min, alt_max,
                initial_z=initial_z, rng=self.rng,
            )
        self.uavs[uav_id] = m
        return m

    def position(self, uav_id, t_abs):
        if uav_id not in self.uavs:
            raise KeyError(f"UAV {uav_id} not registered; call create_uav first")
        return self.uavs[uav_id].position(t_abs)

    def list_mission_types(self):
        return list(self.library.keys())


# ===================== self-test =====================

if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            here, '..', '..', 'finetuing_layers', 'outputs',
            'layer2a_mobility_library.json',
        )

    print(f"Loading mobility library from: {json_path}\n")
    mob = MobilityGenerator(json_path, rng=random.Random(7))

    # -------- Test 1: Spiral grows outward, starts near origin --------
    print("Test 1: Spiral radius grows with time")
    mob.create_uav('uS', 'spiral', origin_xy=(500.0, 500.0))
    for t in (0, 10, 30, 60, 120, 240):
        x, y, z = mob.position('uS', t)
        r = math.hypot(x - 500.0, y - 500.0)
        print(f"  t={t:>4}s: pos=({x:>7.2f}, {y:>7.2f}, {z:>5.2f})  r_from_center={r:>6.2f} m")
    print()

    # -------- Test 2: Grid traces boustrophedon --------
    print("Test 2: Grid boustrophedon (sampling x,y corners over time)")
    mob.create_uav('uG', 'grid', origin_xy=(200.0, 200.0))
    grid_mission = mob.uavs['uG']
    print(f"  total_path = {grid_mission.total_path:.1f} m, "
          f"n_lanes = {grid_mission.n_lanes}, "
          f"duration = {grid_mission.total_path / grid_mission.speed:.1f} s")
    prev = None
    dir_flips = 0
    samples = []
    for t in [i * 1.0 for i in range(0, int(grid_mission.total_path / grid_mission.speed) + 5)]:
        x, y, z = mob.position('uG', t)
        samples.append((t, x, y))
        if prev is not None:
            dx = x - prev[0]
            if prev[2] is not None and (dx * prev[2]) < 0:
                dir_flips += 1
            prev = (x, y, dx)
        else:
            prev = (x, y, None)
    for i in (0, 5, 10, 15, 20, 25, len(samples) - 1):
        if i < len(samples):
            t, x, y = samples[i]
            print(f"  t={t:>5.1f}s: ({x:>7.2f}, {y:>7.2f})")
    print(f"  direction flips along x: {dir_flips}  (expect ~ n_lanes-1 = {grid_mission.n_lanes - 1})")
    zs = [mob.position('uG', t)[2] for t in (0, 10, 50, 100)]
    print(f"  altitude constant: {all(abs(z - zs[0]) < 1e-9 for z in zs)} (z={zs[0]:.2f})")
    print()

    # -------- Test 3: Hover-transit respects hover duration --------
    print("Test 3: Hover_transit spends hover_duration at each waypoint")
    mob.create_uav('uH', 'hover_transit', origin_xy=(400.0, 400.0))
    ht = mob.uavs['uH']
    print(f"  n_waypoints = {len(ht.points)}, total_duration = {ht.total_duration:.2f} s")
    p0 = ht.position(0.0)
    same_count = 0
    for t in [i * 0.5 for i in range(int(ht.hover_dur / 0.5))]:
        if ht.position(t) == p0:
            same_count += 1
    print(f"  first waypoint held for {same_count * 0.5:.1f}s (expect >= {ht.hover_dur:.1f}s)")
    p_mid = ht.position(ht.hover_dur + 1.0)
    moved = (p_mid != p0)
    print(f"  after hover, position changed: {moved}")
    p_end = ht.position(ht.total_duration + 100)
    print(f"  pos at t >> duration: ({p_end[0]:.2f}, {p_end[1]:.2f}, {p_end[2]:.2f})  "
          f"last_wp = ({ht.points[-1][0]:.2f}, {ht.points[-1][1]:.2f}, {ht.points[-1][2]:.2f})")
    print()

    # -------- Test 4: Random walk stays in bounds --------
    print("Test 4: Random walk reflects on all 6 faces (5000s walk, dt=5s, seeded)")
    mob.create_uav(
        'uR', 'random', origin_xy=(500.0, 500.0),
        grid_size_m=1000.0, alt_min=5.0, alt_max=150.0, initial_z=75.0,
    )
    x_min = y_min = z_min = float('inf')
    x_max = y_max = z_max = float('-inf')
    n_hit_z_low = n_hit_z_high = 0
    n_hit_x_bound = n_hit_y_bound = 0
    for step in range(1000):
        t = (step + 1) * 5.0
        x, y, z = mob.position('uR', t)
        x_min, y_min, z_min = min(x_min, x), min(y_min, y), min(z_min, z)
        x_max, y_max, z_max = max(x_max, x), max(y_max, y), max(z_max, z)
        if z < 10: n_hit_z_low += 1
        if z > 145: n_hit_z_high += 1
        if x < 20 or x > 980: n_hit_x_bound += 1
        if y < 20 or y > 980: n_hit_y_bound += 1
    print(f"  x range : [{x_min:.1f}, {x_max:.1f}]   (grid [0, 1000])")
    print(f"  y range : [{y_min:.1f}, {y_max:.1f}]   (grid [0, 1000])")
    print(f"  z range : [{z_min:.1f}, {z_max:.1f}]   (bounds [5, 150])")
    print(f"  samples near z_low (<10m):  {n_hit_z_low}")
    print(f"  samples near z_high (>145): {n_hit_z_high}")
    print(f"  samples near x bound:       {n_hit_x_bound}")
    print(f"  samples near y bound:       {n_hit_y_bound}")
    in_bounds = (0.0 <= x_min and x_max <= 1000.0
                 and 0.0 <= y_min and y_max <= 1000.0
                 and 5.0 <= z_min and z_max <= 150.0)
    print(f"  all samples in bounds: {'OK' if in_bounds else 'FAIL'}")
    print()

    # -------- Test 5: Alpha rescaling sanity --------
    print("Test 5: Alpha rescaling gives sensible values at dt=5s")
    random_params = mob.library['random']
    a_h_base = random_params['alpha_h']
    a_v_base = random_params['alpha_v']
    for dt in (0.2, 1.0, 5.0):
        a_h = a_h_base ** (dt / 0.2)
        a_v = a_v_base ** (dt / 0.2)
        print(f"  dt={dt:>3}s: alpha_h = {a_h:.4e}   alpha_v = {a_v:.4e}")
    print()

    # -------- Test 6: Unknown mission raises --------
    print("Test 6: Unknown mission type raises ValueError")
    try:
        mob.create_uav('uX', 'figure_eight', (0, 0))
        print("  FAIL")
    except ValueError as e:
        print(f"  OK: {e}")
    print()

    print("mobility.py self-test complete.")
