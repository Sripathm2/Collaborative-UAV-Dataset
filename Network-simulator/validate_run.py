#!/usr/bin/env python3
"""
validate_run.py — Post-iteration validation for UAV swarm simulation.

Reads attack_details_*.txt, tc_diagnostics_*.log, and flows-*.txt from
the pcaps directory. For each config run found, checks:
  - tc_verify summary (missing_netem, tc_errors, mismatches)
  - Physical layer sanity (delay, bandwidth, scheme vs SNR)
  - tc actually applied to every container
  - Attack dispatch matches config expectation
  - Flow file existence, flow count, packet counts
  - DoS/DDoS: attack rate matches config; large attack flow present
  - Blackhole: victim logged
  - Wormhole: attacker + victims logged
  - Replay: params logged

Usage:
    python3 validate_run.py [pcaps_dir]
    Default pcaps_dir = <script_dir>/pcaps
"""

import os
import re
import sys
import ast
from collections import defaultdict

# ---- MCS scheme → expected bandwidth ----
SCHEME_BW = {
    'bpsk12': 6, 'qpsk12': 12, 'qpsk34': 18,
    'qam16_12': 24, 'qam16_34': 36, 'qam64_23': 48, 'qam64_34': 54,
}

# ---- Regexes for parsing attack_details lines ----
RE_SCENARIO = re.compile(r"scenario raw='([^']+)'\s+format=(\w+)")
RE_LINK = re.compile(
    r"link uav=(\S+)<->(\S+)\s+d=([\d.]+)m\s+h=([\d.]+)m\s+"
    r"PL=([\d.]+)dB\s+SNR=([\d.]+)dB\s+scheme=(\w+)\s+"
    r"BER=([\d.eE+-]+)\s+PER=([\d.]+)%\s+bw=(\d+)Mb\s+delay=([\d.]+)ms"
)
RE_TC_APPLIED = re.compile(
    r"applied tc on (\w+)/([\w-]+):\s+delay=([\d.]+)ms\s+bw=(\d+)Mb\s+loss=([\d.]+)%"
)
RE_ROUTING = re.compile(
    r"routing:\s+total_links=(\d+)\s+cluster_bridges_added=(\d+)"
)
RE_DOS = re.compile(
    r"DoS attacker=(\w+)\s+victim=(\w+)\s+rate=(\d+)\s+length=(\d+)"
)
RE_DDOS = re.compile(
    r"DDoS attackers=\[([^\]]+)\]\s+victim=(\w+)\s+rate=(\d+)\s+length=(\d+)"
)
RE_BLACKHOLE = re.compile(r"Blackhole victims=\[([^\]]+)\]\s+length=(\d+)")
RE_WORMHOLE = re.compile(r"Wormhole attacker=(\w+)\s+victims=\[([^\]]+)\]")
RE_REPLAY = re.compile(r"Replay attacker=(\w+)\s+victim=(\w+)\s+length=(\d+)")
RE_PL_DIAG = re.compile(r"pathloss diagnostics:\s+(\{.*\})")
RE_TC_SUMMARY_AD = re.compile(r"tc_verify summary:\s+(\{.*\})")

# ---- Regex for tc_diagnostics summary line ----
RE_TC_SUMMARY_LOG = re.compile(
    r"checks=(\d+)\s+mismatches=(\d+)\s+missing_netem=(\d+)\s+tc_errors=(\d+)"
)

# ---- Result levels ----
PASS = 'PASS'
WARN = 'WARN'
ERROR = 'ERROR'
INFO = 'INFO'


def color(level, text):
    """ANSI coloring for terminal output."""
    codes = {PASS: '\033[92m', WARN: '\033[93m', ERROR: '\033[91m', INFO: '\033[94m'}
    return f"{codes.get(level, '')}{text}\033[0m"


# ==================== Parsers ====================

def parse_attack_details(path):
    """Parse attack_details_*.txt into a structured dict."""
    result = {
        'scenario_raw': None, 'format': None,
        'links': [],           # list of dicts per link line
        'tc_applied': [],      # list of dicts per "applied tc" line
        'routing': [],         # list of (total_links, bridges) per window
        'attacks_logged': [],  # list of (type, details_dict)
        'pl_diag': None,
        'tc_summary': None,
        'num_windows': 0,
    }
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Strip timestamp prefix
            if line.startswith('***'):
                line = line.split(' ', 1)[-1] if ' ' in line else line

            m = RE_SCENARIO.search(line)
            if m:
                result['scenario_raw'] = m.group(1)
                result['format'] = m.group(2)
                continue

            m = RE_LINK.search(line)
            if m:
                result['links'].append({
                    'uav': m.group(1), 'peer': m.group(2),
                    'dist_m': float(m.group(3)), 'alt_m': float(m.group(4)),
                    'pl_db': float(m.group(5)), 'snr_db': float(m.group(6)),
                    'scheme': m.group(7),
                    'ber': float(m.group(8)), 'per_pct': float(m.group(9)),
                    'bw_mbit': int(m.group(10)), 'delay_ms': float(m.group(11)),
                })
                continue

            m = RE_TC_APPLIED.search(line)
            if m:
                result['tc_applied'].append({
                    'node': m.group(1), 'iface': m.group(2),
                    'delay_ms': float(m.group(3)), 'bw_mbit': int(m.group(4)),
                    'loss_pct': float(m.group(5)),
                })
                continue

            m = RE_ROUTING.search(line)
            if m:
                result['routing'].append({
                    'total_links': int(m.group(1)),
                    'bridges': int(m.group(2)),
                })
                continue

            m = RE_DOS.search(line)
            if m:
                result['attacks_logged'].append(('dos', {
                    'attacker': m.group(1), 'victim': m.group(2),
                    'rate': int(m.group(3)), 'length': int(m.group(4)),
                }))
                continue

            m = RE_DDOS.search(line)
            if m:
                result['attacks_logged'].append(('ddos', {
                    'attackers': [s.strip().strip("'") for s in m.group(1).split(',')],
                    'victim': m.group(2),
                    'rate': int(m.group(3)), 'length': int(m.group(4)),
                }))
                continue

            m = RE_BLACKHOLE.search(line)
            if m:
                result['attacks_logged'].append(('blackhole', {
                    'victims': [s.strip().strip("'") for s in m.group(1).split(',')],
                    'length': int(m.group(2)),
                }))
                continue

            m = RE_WORMHOLE.search(line)
            if m:
                result['attacks_logged'].append(('wormhole', {
                    'attacker': m.group(1),
                    'victims': [s.strip().strip("'") for s in m.group(2).split(',')],
                }))
                continue

            m = RE_REPLAY.search(line)
            if m:
                result['attacks_logged'].append(('replay', {
                    'attacker': m.group(1), 'victim': m.group(2),
                    'length': int(m.group(3)),
                }))
                continue

            m = RE_PL_DIAG.search(line)
            if m:
                try:
                    result['pl_diag'] = ast.literal_eval(m.group(1))
                except Exception:
                    pass
                continue

            m = RE_TC_SUMMARY_AD.search(line)
            if m:
                try:
                    result['tc_summary'] = ast.literal_eval(m.group(1))
                except Exception:
                    pass
                continue

            if 't_abs=' in line:
                result['num_windows'] += 1

    return result


def parse_tc_diagnostics(path):
    """Parse tc_diagnostics_*.log for the summary line."""
    result = {'checks': 0, 'mismatches': 0, 'missing_netem': 0, 'tc_errors': 0}
    with open(path) as f:
        for line in f:
            m = RE_TC_SUMMARY_LOG.search(line)
            if m:
                result = {
                    'checks': int(m.group(1)),
                    'mismatches': int(m.group(2)),
                    'missing_netem': int(m.group(3)),
                    'tc_errors': int(m.group(4)),
                }
    return result


def parse_flows_file(path):
    """
    Parse flows-*.txt. Returns list of dicts with rich per-flow data:
      {'ip_a': str, 'ip_b': str, 'packet_count': int,
       'first_ts': float, 'last_ts': float, 'duration_s': float,
       'rate_pps': float, 'total_bytes': int}

    Handles both clean IPs and comma-separated multi-layer IPs.
    """
    flows = []
    re_ip = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
    # Match (timestamp, size, ...) tuples inside the packet list
    re_pkt = re.compile(r'\(([\d.]+),\s*(\d+),')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            ip_pair = parts[0].strip()
            if '<->' not in ip_pair:
                continue
            left, right = ip_pair.split('<->', 1)
            ip_a = left.strip().split(',')[-1].strip()
            ip_b = right.strip().split(',')[-1].strip()
            if not re_ip.match(ip_a) or not re_ip.match(ip_b):
                continue

            # Extract all (timestamp, size) from packet list
            matches = re_pkt.findall(parts[1])
            if not matches:
                continue
            timestamps = [float(m[0]) for m in matches]
            sizes = [int(m[1]) for m in matches]
            pkt_count = len(timestamps)
            first_ts = timestamps[0]
            last_ts = timestamps[-1]
            duration_s = last_ts - first_ts
            rate_pps = pkt_count / duration_s if duration_s > 0.01 else 0.0
            total_bytes = sum(sizes)

            flows.append({
                'ip_a': ip_a, 'ip_b': ip_b,
                'packet_count': pkt_count,
                'first_ts': first_ts, 'last_ts': last_ts,
                'duration_s': duration_s,
                'rate_pps': rate_pps,
                'total_bytes': total_bytes,
            })
    return flows


def _node_ip(name, num_drones):
    """Map node name to IP address.
    d<n> -> 10.0.0.<n>, bs<m> -> 10.0.0.<num_drones + m>"""
    if name.startswith('d') and name[1:].isdigit():
        return f"10.0.0.{int(name[1:])}"
    if name.startswith('bs') and name[2:].isdigit():
        return f"10.0.0.{num_drones + int(name[2:])}"
    return None


def _find_flow(flows, ip_x, ip_y):
    """Find flows between two IPs (order-independent). Returns list."""
    return [f for f in flows
            if {f['ip_a'], f['ip_b']} == {ip_x, ip_y}
            or ip_x in (f['ip_a'], f['ip_b']) and ip_y in (f['ip_a'], f['ip_b'])]


def _flows_involving(flows, ip):
    """Find all flows where one endpoint is the given IP."""
    return [f for f in flows if ip in (f['ip_a'], f['ip_b'])]


def parse_config_attacks(scenario_raw):
    """
    From the raw config string, extract which attacks are expected.
    Returns list of (attack_type, params_dict).
    E.g. 'blackhole+dos=1000-5-1-...' -> [('blackhole',{}), ('dos',{'rate':1000})]
    """
    # First field before the first '-<digit>' is the attack spec
    # Split on '-' but the attack field itself can contain '-' (e.g. no,
    # actually attacks use '+' as separator, digits after '=' are params)
    config_parts = scenario_raw.split('-')
    attack_field = config_parts[0]
    attacks = []
    for atk_str in attack_field.split('+'):
        if '=' in atk_str:
            name, params_str = atk_str.split('=', 1)
        else:
            name = atk_str
            params_str = ''

        if name == 'dos' or name == 'ddos':
            attacks.append((name, {'rate': int(params_str)}))
        elif name == 'replay':
            # replay=50,200,10,5,inc
            rp = params_str.split(',')
            attacks.append(('replay', {
                'buffer': int(rp[0]) if len(rp) > 0 else 0,
                'delay_ms': int(rp[1]) if len(rp) > 1 else 0,
                'rate_pps': int(rp[2]) if len(rp) > 2 else 0,
                'ttl_dec': int(rp[3]) if len(rp) > 3 else 0,
                'seq_mode': rp[4] if len(rp) > 4 else 'inc',
            }))
        else:
            attacks.append((name, {}))
    return attacks


def parse_config_params(scenario_raw):
    """Extract num_drones, num_bs, tx_power, etc. from the raw config string."""
    parts = scenario_raw.split('-')
    # Format: <attacks>-<drones>-<bs>-<payload>-<pathloss>-<modulation>-<missions>-<tx>-<noise>
    return {
        'num_drones': int(parts[1]),
        'num_bs': int(parts[2]),
        'payload': parts[3],
        'pathloss': parts[4],
        'modulation': parts[5],
        'mission': parts[6],
        'tx_power': int(parts[7]),
        'noise': int(parts[8]),
    }


# ==================== Validators ====================

def validate_config(tag, pcaps_dir):
    """Run all checks for one config run. Returns (results_list, scenario_raw)."""
    results = []

    ad_path = None
    tc_path = None
    flows_path = os.path.join(pcaps_dir, f"flows-{tag}.txt")

    # Find attack_details and tc_diagnostics files matching this tag
    for fn in os.listdir(pcaps_dir):
        if fn.startswith('attack_details_') and fn.endswith(f'-{tag.split("-")[-1]}.txt'):
            # Match by YYYYMMDD-HHMM-CONFIG in the filename
            # attack_details_YYYYMMDD-HHMM-CONFIG.txt
            candidate_tag = fn.replace('attack_details_', '').replace('.txt', '')
            if candidate_tag == tag:
                ad_path = os.path.join(pcaps_dir, fn)
        if fn.startswith('tc_diagnostics_') and fn.endswith(f'-{tag.split("-")[-1]}.log'):
            candidate_tag = fn.replace('tc_diagnostics_', '').replace('.log', '')
            if candidate_tag == tag:
                tc_path = os.path.join(pcaps_dir, fn)

    # ---- 1. File existence ----
    if not ad_path or not os.path.exists(ad_path):
        results.append((ERROR, "attack_details file MISSING"))
        return results, None
    results.append((PASS, f"attack_details file exists"))

    if not tc_path or not os.path.exists(tc_path):
        results.append((ERROR, "tc_diagnostics file MISSING"))
    else:
        results.append((PASS, "tc_diagnostics file exists"))

    if not os.path.exists(flows_path):
        results.append((ERROR, f"flows file MISSING (expected flows-{tag}.txt)"))
    else:
        sz = os.path.getsize(flows_path)
        results.append((PASS if sz > 0 else ERROR,
                        f"flows file exists ({sz} bytes)"))

    # ---- 2. Parse files ----
    ad = parse_attack_details(ad_path)
    scenario_raw = ad['scenario_raw']
    if not scenario_raw:
        results.append((ERROR, "Could not parse scenario line from attack_details"))
        return results, None

    config_attacks = parse_config_attacks(scenario_raw)
    config_params = parse_config_params(scenario_raw)
    expected_attack_types = [a[0] for a in config_attacks]

    tc_diag = parse_tc_diagnostics(tc_path) if tc_path and os.path.exists(tc_path) else None
    flows = parse_flows_file(flows_path) if os.path.exists(flows_path) else []

    # ---- 3. tc_verify summary ----
    if tc_diag is not None:
        ne = tc_diag['missing_netem']
        te = tc_diag['tc_errors']
        mm = tc_diag['mismatches']
        ch = tc_diag['checks']

        if ne > 0:
            results.append((ERROR, f"tc_verify: missing_netem={ne} (netem not installed on {ne} interfaces)"))
        else:
            results.append((PASS, "tc_verify: missing_netem=0"))

        if te > 0:
            results.append((ERROR, f"tc_verify: tc_errors={te} (tc command failures)"))
        else:
            results.append((PASS, "tc_verify: tc_errors=0"))

        if ch == 0:
            results.append((WARN, "tc_verify: checks=0 (verifier never ran)"))
        elif mm > 0:
            pct = 100.0 * mm / ch
            level = WARN if pct < 10.0 else ERROR
            results.append((level, f"tc_verify: mismatches={mm}/{ch} ({pct:.0f}%)"))
        else:
            results.append((PASS, f"tc_verify: {ch} checks, 0 mismatches"))

    # ---- 4. Physical layer sanity ----
    if ad['links']:
        bws = set(lk['bw_mbit'] for lk in ad['links'])
        schemes = set(lk['scheme'] for lk in ad['links'])
        delays = [lk['delay_ms'] for lk in ad['links']]
        snrs = [lk['snr_db'] for lk in ad['links']]
        dists = [lk['dist_m'] for lk in ad['links']]

        # Check bw matches scheme
        bw_ok = True
        for lk in ad['links']:
            expected_bw = SCHEME_BW.get(lk['scheme'])
            if expected_bw and lk['bw_mbit'] != expected_bw:
                results.append((ERROR,
                    f"link {lk['uav']}<->{lk['peer']}: scheme={lk['scheme']} "
                    f"should give bw={expected_bw}Mb but got bw={lk['bw_mbit']}Mb"))
                bw_ok = False
                break
        if bw_ok:
            results.append((PASS,
                f"link bw matches scheme for all {len(ad['links'])} link computations "
                f"(schemes: {sorted(schemes)}, bw: {sorted(bws)}Mb)"))

        # Delay sanity
        min_d, max_d = min(delays), max(delays)
        if max_d > 50.0:
            results.append((WARN, f"delay range [{min_d:.3f}, {max_d:.3f}]ms — max>50ms is unusual"))
        else:
            results.append((PASS, f"delay range [{min_d:.3f}, {max_d:.3f}]ms"))

        # SNR sanity
        min_snr, max_snr = min(snrs), max(snrs)
        if min_snr < 0:
            results.append((WARN, f"SNR range [{min_snr:.1f}, {max_snr:.1f}]dB — negative SNR means link is failing"))
        else:
            results.append((PASS, f"SNR range [{min_snr:.1f}, {max_snr:.1f}]dB"))

        # Distance sanity
        results.append((INFO, f"link distances [{min(dists):.0f}, {max(dists):.0f}]m"))

        # Links per window
        n_routing = len(ad['routing'])
        n_links = len(ad['links'])
        if n_routing > 0:
            links_per_window = n_links / n_routing
            results.append((INFO,
                f"{links_per_window:.0f} links/window across {n_routing} windows"))
    else:
        results.append((ERROR, "No link computations found in attack_details"))

    # ---- 5. tc applied to containers ----
    if ad['tc_applied']:
        nodes_with_tc = set(tc['node'] for tc in ad['tc_applied'])
        num_containers = config_params['num_drones'] + config_params['num_bs']
        n_windows = ad['num_windows'] + 1  # +1 for initial relink before sim loop

        expected_applies = num_containers * n_windows
        actual_applies = len(ad['tc_applied'])

        if nodes_with_tc == set():
            results.append((ERROR, "tc not applied to any containers"))
        elif len(nodes_with_tc) < num_containers:
            missing = num_containers - len(nodes_with_tc)
            results.append((WARN,
                f"tc applied to {len(nodes_with_tc)}/{num_containers} containers "
                f"(missing {missing})"))
        else:
            results.append((PASS,
                f"tc applied to all {num_containers} containers "
                f"({actual_applies} applies across {n_windows} windows)"))
    else:
        results.append((ERROR, "No 'applied tc' lines found"))

    # ---- 6. Attack dispatch checks ----
    logged_types = [a[0] for a in ad['attacks_logged']]

    if 'benign' in expected_attack_types and len(expected_attack_types) == 1:
        if len(logged_types) == 0:
            results.append((PASS, "Benign config: no attacks logged (correct)"))
        else:
            results.append((WARN,
                f"Benign config but attacks logged: {logged_types}"))
    else:
        for expected_type, expected_params in config_attacks:
            if expected_type == 'benign':
                continue

            if expected_type not in logged_types:
                results.append((ERROR,
                    f"Expected {expected_type} attack but not found in logs"))
                continue

            # Find the matching logged attack
            logged = None
            for lt, ld in ad['attacks_logged']:
                if lt == expected_type:
                    logged = ld
                    break

            if expected_type == 'dos':
                rate = logged.get('rate', '?')
                expected_rate = expected_params.get('rate', '?')
                if rate == expected_rate:
                    results.append((PASS,
                        f"DoS dispatched: attacker={logged['attacker']} "
                        f"victim={logged['victim']} rate={rate}us "
                        f"length={logged['length']}s"))
                else:
                    results.append((ERROR,
                        f"DoS rate mismatch: config says {expected_rate}us "
                        f"but log says {rate}us"))

            elif expected_type == 'ddos':
                rate = logged.get('rate', '?')
                expected_rate = expected_params.get('rate', '?')
                n_attackers = len(logged.get('attackers', []))
                if rate == expected_rate:
                    results.append((PASS,
                        f"DDoS dispatched: {n_attackers} attackers -> "
                        f"victim={logged['victim']} rate={rate}us "
                        f"length={logged['length']}s"))
                else:
                    results.append((ERROR,
                        f"DDoS rate mismatch: config says {expected_rate}us "
                        f"but log says {rate}us"))
                if n_attackers < 2:
                    results.append((WARN,
                        f"DDoS has only {n_attackers} attacker(s) (expected >=2)"))

            elif expected_type == 'blackhole':
                n_victims = len(logged.get('victims', []))
                results.append((PASS,
                    f"Blackhole dispatched: {n_victims} victim(s)={logged['victims']} "
                    f"length={logged['length']}s"))

            elif expected_type == 'wormhole':
                results.append((PASS,
                    f"Wormhole dispatched: attacker={logged['attacker']} "
                    f"victims={logged['victims']}"))

            elif expected_type == 'replay':
                results.append((PASS,
                    f"Replay dispatched: attacker={logged['attacker']} "
                    f"victim={logged['victim']} length={logged['length']}s"))

    # ---- 7. Flow analysis ----
    if flows:
        total_pkts = sum(f['packet_count'] for f in flows)
        max_flow = max(flows, key=lambda f: f['packet_count'])

        results.append((INFO,
            f"Flows: {len(flows)} flows, {total_pkts} total packets, "
            f"largest={max_flow['packet_count']} pkts"))

        # Per-flow rate INFO
        for f in flows:
            rate_str = f"{f['rate_pps']:.1f} pps" if f['rate_pps'] > 0 else "N/A"
            results.append((INFO,
                f"  {f['ip_a']} <-> {f['ip_b']}: "
                f"{f['packet_count']} pkts, {f['duration_s']:.1f}s, "
                f"{rate_str}, {f['total_bytes']} bytes"))

        # Minimum flow count: at least 1 per drone
        if len(flows) < config_params['num_drones']:
            results.append((WARN,
                f"Only {len(flows)} flows for {config_params['num_drones']} drones "
                f"(expected at least {config_params['num_drones']})"))

        n_drones = config_params['num_drones']

        # ---- Attack-specific flow checks ----
        for atk_type, atk_params in config_attacks:
            if atk_type == 'benign':
                continue

            for lt, ld in ad['attacks_logged']:
                if lt != atk_type:
                    continue

                if lt == 'dos':
                    attacker_ip = _node_ip(ld['attacker'], n_drones)
                    victim_ip = _node_ip(ld['victim'], n_drones)
                    rate_us = ld['rate']
                    length_s = ld['length']

                    # Expected theoretical rate
                    if rate_us == 1:
                        expected_pps_str = "flood"
                        expected_pps = None
                    else:
                        expected_pps = 1_000_000.0 / rate_us
                        expected_pps_str = f"{expected_pps:.0f} pps"

                    # Find attack flow
                    atk_flows = _find_flow(flows, attacker_ip, victim_ip)
                    if not atk_flows:
                        results.append((ERROR,
                            f"DoS: no flow between attacker {attacker_ip} and "
                            f"victim {victim_ip}"))
                        continue

                    af = max(atk_flows, key=lambda f: f['packet_count'])
                    actual_pps = af['rate_pps']

                    results.append((INFO,
                        f"DoS attack flow: {attacker_ip}<->{victim_ip} "
                        f"{af['packet_count']} pkts, {af['duration_s']:.1f}s, "
                        f"{actual_pps:.0f} pps (expected ~{expected_pps_str})"))

                    # Rate check (generous: within 10x for network overhead)
                    if expected_pps is not None and actual_pps > 0:
                        ratio = actual_pps / expected_pps
                        if 0.1 <= ratio <= 10.0:
                            results.append((PASS,
                                f"DoS rate {actual_pps:.0f} pps is within "
                                f"reasonable range of expected {expected_pps:.0f} pps "
                                f"(ratio={ratio:.2f}x)"))
                        else:
                            results.append((WARN,
                                f"DoS rate {actual_pps:.0f} pps far from expected "
                                f"{expected_pps:.0f} pps (ratio={ratio:.2f}x)"))
                    elif af['packet_count'] >= 100:
                        results.append((PASS,
                            f"DoS flow has {af['packet_count']} pkts (attack visible)"))
                    else:
                        results.append((WARN,
                            f"DoS flow only {af['packet_count']} pkts"))

                elif lt == 'ddos':
                    attackers = ld.get('attackers', [])
                    victim_ip = _node_ip(ld['victim'], n_drones)
                    rate_us = ld['rate']
                    length_s = ld['length']

                    if rate_us == 1:
                        expected_pps_str = "flood"
                        expected_pps = None
                    else:
                        expected_pps = 1_000_000.0 / rate_us
                        expected_pps_str = f"{expected_pps:.0f} pps"

                    victim_flows = _flows_involving(flows, victim_ip)
                    if not victim_flows:
                        results.append((ERROR,
                            f"DDoS: no flows involving victim {victim_ip}"))
                        continue

                    # Check each attacker has a flow to the victim
                    attacker_ips_seen = set()
                    for aname in attackers:
                        aip = _node_ip(aname, n_drones)
                        af = _find_flow(flows, aip, victim_ip)
                        if af:
                            best = max(af, key=lambda f: f['packet_count'])
                            attacker_ips_seen.add(aip)
                            results.append((INFO,
                                f"  DDoS attacker {aname}({aip})->{victim_ip}: "
                                f"{best['packet_count']} pkts, "
                                f"{best['rate_pps']:.0f} pps "
                                f"(expected ~{expected_pps_str})"))

                            if expected_pps is not None and best['rate_pps'] > 0:
                                ratio = best['rate_pps'] / expected_pps
                                if 0.1 <= ratio <= 10.0:
                                    results.append((PASS,
                                        f"  DDoS {aname} rate OK "
                                        f"(ratio={ratio:.2f}x)"))
                                else:
                                    results.append((WARN,
                                        f"  DDoS {aname} rate off "
                                        f"(ratio={ratio:.2f}x)"))
                        else:
                            results.append((WARN,
                                f"  DDoS attacker {aname}({aip}): no flow to "
                                f"victim {victim_ip}"))

                    if len(attacker_ips_seen) >= 2:
                        results.append((PASS,
                            f"DDoS: {len(attacker_ips_seen)}/{len(attackers)} "
                            f"attackers have flows to victim"))
                    elif len(attacker_ips_seen) == 1:
                        results.append((WARN,
                            f"DDoS: only 1/{len(attackers)} attacker flows visible"))
                    else:
                        results.append((ERROR,
                            f"DDoS: no attacker flows to victim found"))

                elif lt == 'blackhole':
                    victim_names = ld.get('victims', [])
                    victim_ips = {_node_ip(v, n_drones) for v in victim_names}
                    attack_length = ld.get('length', 0)

                    # Fix 4: show attack duration vs sim duration for context
                    sim_duration = max((f['duration_s'] for f in flows), default=0)
                    results.append((INFO,
                        f"Blackhole: attack lasted {attack_length}s "
                        f"within ~{sim_duration:.0f}s simulation"))

                    # Fix 3: exclude DoS/DDoS-involved IPs from blackhole
                    # comparison so attack traffic doesn't skew averages
                    dos_involved_ips = set()
                    for olt, old in ad['attacks_logged']:
                        if olt == 'dos':
                            dos_involved_ips.add(
                                _node_ip(old['attacker'], n_drones))
                            dos_involved_ips.add(
                                _node_ip(old['victim'], n_drones))
                        elif olt == 'ddos':
                            for aname in old.get('attackers', []):
                                dos_involved_ips.add(
                                    _node_ip(aname, n_drones))
                            dos_involved_ips.add(
                                _node_ip(old['victim'], n_drones))

                    if dos_involved_ips:
                        results.append((INFO,
                            f"  Excluding DoS-involved IPs from blackhole "
                            f"comparison: {sorted(dos_involved_ips)}"))

                    victim_flow_stats = []
                    nonvictim_flow_stats = []
                    for di in range(1, n_drones + 1):
                        dip = f"10.0.0.{di}"
                        if dip in dos_involved_ips:
                            continue
                        dflows = _flows_involving(flows, dip)
                        total = sum(f['packet_count'] for f in dflows)
                        if dip in victim_ips:
                            victim_flow_stats.append((f"d{di}", dip, total))
                        else:
                            nonvictim_flow_stats.append((f"d{di}", dip, total))

                    for name, ip, total in victim_flow_stats:
                        results.append((INFO,
                            f"  Blackhole victim {name}({ip}): {total} total pkts"))
                    for name, ip, total in nonvictim_flow_stats:
                        results.append((INFO,
                            f"  Blackhole non-victim {name}({ip}): {total} total pkts"))

                    if not nonvictim_flow_stats:
                        results.append((WARN,
                            f"Blackhole: no clean non-victim drones for "
                            f"comparison (all non-victims are DoS-involved)"))
                    elif not victim_flow_stats:
                        results.append((WARN,
                            f"Blackhole: no clean victim drones for "
                            f"comparison (all victims are DoS-involved)"))
                    else:
                        avg_victim = (sum(t for _, _, t in victim_flow_stats) /
                                      len(victim_flow_stats))
                        avg_nonvictim = (sum(t for _, _, t in nonvictim_flow_stats) /
                                         len(nonvictim_flow_stats))

                        if avg_nonvictim > 0 and avg_victim < avg_nonvictim * 0.8:
                            results.append((PASS,
                                f"Blackhole: victim avg {avg_victim:.0f} pkts < "
                                f"non-victim avg {avg_nonvictim:.0f} pkts "
                                f"(drop ratio={avg_victim/avg_nonvictim:.2f}x)"))
                        elif avg_nonvictim > 0:
                            results.append((WARN,
                                f"Blackhole: victim avg {avg_victim:.0f} pkts vs "
                                f"non-victim avg {avg_nonvictim:.0f} pkts "
                                f"— effect may be weak (attack was only "
                                f"{attack_length}s of ~{sim_duration:.0f}s sim)"))
                        else:
                            results.append((WARN,
                                f"Blackhole: non-victim avg is 0 pkts — "
                                f"can't compare (FTP may have failed for all)"))

                elif lt == 'wormhole':
                    # Fix 1: Wormhole is a Layer 2 attack — victim traffic
                    # is rerouted through the attacker's switch, but IP
                    # headers are untouched. There will NOT be an
                    # attacker↔victim IP flow. Instead, check that:
                    # (a) victim still has traffic (rerouted, not dropped)
                    # (b) victim's rate may be lower than non-victims (extra hop)
                    attacker = ld.get('attacker')
                    victims = ld.get('victims', [])
                    attacker_ip = _node_ip(attacker, n_drones)
                    bs_ip = _node_ip('bs1', n_drones)

                    # Gather per-drone packet totals
                    victim_totals = {}
                    nonvictim_totals = {}
                    for di in range(1, n_drones + 1):
                        dname = f"d{di}"
                        dip = f"10.0.0.{di}"
                        if dname == attacker:
                            continue  # attacker carries forwarded traffic, skip
                        total = sum(f['packet_count']
                                    for f in _flows_involving(flows, dip))
                        if dname in victims:
                            victim_totals[dname] = total
                        else:
                            nonvictim_totals[dname] = total

                    for vname in victims:
                        victim_ip = _node_ip(vname, n_drones)
                        vtotal = victim_totals.get(vname, 0)
                        # Check victim has some traffic (reroute works)
                        if vtotal > 0:
                            results.append((PASS,
                                f"Wormhole victim {vname}({victim_ip}): "
                                f"{vtotal} pkts (traffic flowing via reroute)"))
                        else:
                            results.append((WARN,
                                f"Wormhole victim {vname}({victim_ip}): "
                                f"0 pkts (reroute may have broken connectivity)"))

                    # Compare victim avg vs non-victim avg
                    if victim_totals and nonvictim_totals:
                        avg_v = sum(victim_totals.values()) / len(victim_totals)
                        avg_nv = sum(nonvictim_totals.values()) / len(nonvictim_totals)
                        for name, total in victim_totals.items():
                            results.append((INFO,
                                f"  Wormhole victim {name}: {total} pkts"))
                        for name, total in nonvictim_totals.items():
                            results.append((INFO,
                                f"  Wormhole non-victim {name}: {total} pkts"))
                        results.append((INFO,
                            f"  Wormhole attacker {attacker}: excluded from "
                            f"comparison (carries forwarded traffic)"))
                        if avg_nv > 0:
                            ratio = avg_v / avg_nv
                            results.append((INFO,
                                f"  Victim avg {avg_v:.0f} vs non-victim avg "
                                f"{avg_nv:.0f} pkts (ratio={ratio:.2f}x)"))

                elif lt == 'replay':
                    # Fix 2: Replay attack captures packets and replays them
                    # with the ORIGINAL source/destination IPs (only TTL/seq
                    # are modified). So replayed traffic does NOT create an
                    # attacker↔victim IP flow. Instead check:
                    # (a) victim's total traffic is elevated vs baseline
                    # (b) report the elevation ratio
                    attacker = ld.get('attacker')
                    victim = ld.get('victim')
                    attacker_ip = _node_ip(attacker, n_drones)
                    victim_ip = _node_ip(victim, n_drones)

                    victim_total = sum(f['packet_count']
                                       for f in _flows_involving(flows, victim_ip))

                    # Baseline: non-attack drones
                    baseline_totals = []
                    for di in range(1, n_drones + 1):
                        dname = f"d{di}"
                        if dname == attacker or dname == victim:
                            continue
                        dip = f"10.0.0.{di}"
                        t = sum(f['packet_count']
                                for f in _flows_involving(flows, dip))
                        if t > 0:
                            baseline_totals.append((dname, t))

                    if baseline_totals:
                        avg_baseline = (sum(t for _, t in baseline_totals) /
                                        len(baseline_totals))
                        for bname, bt in baseline_totals:
                            results.append((INFO,
                                f"  Replay baseline {bname}: {bt} pkts"))
                        results.append((INFO,
                            f"  Replay victim {victim}: {victim_total} pkts, "
                            f"baseline avg: {avg_baseline:.0f} pkts"))

                        if avg_baseline > 0:
                            ratio = victim_total / avg_baseline
                            if ratio > 1.3:
                                results.append((PASS,
                                    f"Replay: victim traffic elevated "
                                    f"({ratio:.1f}x baseline) — "
                                    f"replayed packets visible"))
                            elif ratio > 0.8:
                                results.append((WARN,
                                    f"Replay: victim traffic not clearly "
                                    f"elevated ({ratio:.1f}x baseline) — "
                                    f"replay may have failed or effect is weak"))
                            else:
                                results.append((WARN,
                                    f"Replay: victim traffic LOWER than "
                                    f"baseline ({ratio:.1f}x) — "
                                    f"replay likely failed"))
                    else:
                        results.append((WARN,
                            f"Replay: no baseline drones available "
                            f"for comparison"))

    elif os.path.exists(flows_path):
        results.append((WARN, "Flows file exists but is empty (0 flows)"))

    # ---- 8. Pathloss diagnostics ----
    if ad['pl_diag']:
        clamp = ad['pl_diag'].get('clamp_total', 0)
        below = ad['pl_diag'].get('clamp_below_30m', 0)
        above = ad['pl_diag'].get('clamp_above_110m', 0)
        if clamp > 0:
            # With UAV_ALT_MIN=30 and UAV_ALT_MAX=110 matching the fitted
            # range, clamping should never happen. If it does, something
            # is wrong with the mobility module or altitude bounds.
            results.append((ERROR,
                f"pathloss altitude clamped {clamp} times "
                f"[below_30m={below}, above_110m={above}] "
                f"(should be 0 — UAV bounds match fitted range 30-110m)"))
        else:
            results.append((PASS, "pathloss: no altitude clamping (UAVs within 30-110m)"))

    return results, scenario_raw


# ==================== Results persistence ====================

RESULTS_FILENAME = 'validation_results.txt'


def _results_file_path(pcaps_dir):
    return os.path.join(pcaps_dir, RESULTS_FILENAME)


def load_saved_results(pcaps_dir):
    """Load previously saved validation results. Returns dict {tag: (cfg_idx, scenario, status, errors, warns)}."""
    path = _results_file_path(pcaps_dir)
    saved = {}
    if not os.path.exists(path):
        return saved
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) < 6:
                continue
            tag, cfg_idx, scenario, status, errors, warns = parts[:6]
            saved[tag] = (cfg_idx, scenario, status, int(errors), int(warns))
    return saved


def append_result(pcaps_dir, tag, cfg_idx, scenario, status, errors, warns):
    """Append one validation result to the results file."""
    path = _results_file_path(pcaps_dir)
    with open(path, 'a') as f:
        f.write(f"{tag}|{cfg_idx}|{scenario}|{status}|{errors}|{warns}\n")


def find_run_tags(pcaps_dir):
    """Find all run tags from attack_details_*.txt files."""
    tags = []
    for fn in sorted(os.listdir(pcaps_dir)):
        if fn.startswith('attack_details_') and fn.endswith('.txt'):
            tag = fn.replace('attack_details_', '').replace('.txt', '')
            tags.append(tag)
    return tags


def format_result_line(level, msg, use_color=True):
    """Format one check result for display."""
    if use_color:
        prefix = f"  [{color(level, level):>15s}]"
    else:
        prefix = f"  [{level:>5s}]"
    return f"{prefix} {msg}"


def format_summary_line(cfg_idx, raw, status, use_color=True):
    """Format one summary table row."""
    atk = raw.split('-')[0] if raw != '?' else '?'
    if use_color:
        indicator = color(PASS, 'OK') if status == 'CLEAN' else (
            color(ERROR, status) if 'E' in status else color(WARN, status))
    else:
        indicator = 'OK' if status == 'CLEAN' else status
    return f"  cfg {cfg_idx:>5s}  {indicator:>15s}  {atk}"


# ==================== Main ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Validate UAV simulation runs.')
    parser.add_argument('pcaps_dir', nargs='?', default=None,
                        help='Path to pcaps directory (default: <script_dir>/pcaps)')
    parser.add_argument('--sum', action='store_true', dest='summary_only',
                        help='Print summary table from saved results only; '
                             'do not validate new configs')
    args = parser.parse_args()

    if args.pcaps_dir:
        pcaps_dir = args.pcaps_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pcaps_dir = os.path.join(script_dir, 'pcaps')

    if not os.path.isdir(pcaps_dir):
        print(f"ERROR: {pcaps_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    saved = load_saved_results(pcaps_dir)

    # --sum: just print the saved summary and exit
    if args.summary_only:
        if not saved:
            print("No saved validation results found.")
            sys.exit(0)
        total_e = sum(v[3] for v in saved.values())
        total_w = sum(v[4] for v in saved.values())
        print(f"{'='*70}")
        print(f"  SUMMARY: {len(saved)} configs | {total_e} errors | {total_w} warnings")
        print(f"{'='*70}")
        for tag in sorted(saved, key=lambda t: int(t.split('-')[-1])):
            cfg_idx, scenario, status, e, w = saved[tag]
            print(format_summary_line(cfg_idx, scenario, status))
        print()
        sys.exit(1 if total_e > 0 else 0)

    # Normal mode: find new tags, validate only those
    all_tags = find_run_tags(pcaps_dir)
    new_tags = [t for t in all_tags if t not in saved]

    if not new_tags:
        n = len(all_tags)
        print(f"All {n} configs already validated. "
              f"Use --sum to see summary, or delete "
              f"{RESULTS_FILENAME} to re-validate.")
        sys.exit(0)

    print(f"Found {len(new_tags)} new config(s) to validate "
          f"({len(saved)} already done)\n")

    total_errors = 0
    total_warns = 0
    new_summary = []

    # Open a detail log file for this batch
    detail_log = os.path.join(
        pcaps_dir,
        f"validation_detail_{new_tags[0].rsplit('-',1)[0]}.txt"
    )

    with open(detail_log, 'w') as detail_f:
        for tag in new_tags:
            results, scenario_raw = validate_config(tag, pcaps_dir)

            cfg_idx = tag.split('-')[-1]
            header = f"Config {cfg_idx}"
            if scenario_raw:
                header += f" ({scenario_raw})"

            # Print to terminal
            print(f"{'='*70}")
            print(f"  {header}")
            print(f"{'='*70}")

            # Write to detail log
            detail_f.write(f"{'='*70}\n")
            detail_f.write(f"  {header}\n")
            detail_f.write(f"{'='*70}\n")

            errors = 0
            warns = 0
            for level, msg in results:
                print(format_result_line(level, msg, use_color=True))
                detail_f.write(format_result_line(level, msg, use_color=False) + '\n')
                if level == ERROR:
                    errors += 1
                elif level == WARN:
                    warns += 1

            total_errors += errors
            total_warns += warns
            status = 'CLEAN' if errors == 0 and warns == 0 else (
                f'{errors}E/{warns}W' if errors > 0 else f'{warns}W')
            new_summary.append((cfg_idx, scenario_raw or '?', status))
            print()
            detail_f.write('\n')

            # Save to persistent results
            append_result(pcaps_dir, tag, cfg_idx,
                          scenario_raw or '?', status, errors, warns)

        # Write batch summary to detail log
        detail_f.write(f"{'='*70}\n")
        detail_f.write(f"  BATCH: {len(new_tags)} configs | "
                       f"{total_errors} errors | {total_warns} warnings\n")
        detail_f.write(f"{'='*70}\n")
        for cfg_idx, raw, status in new_summary:
            detail_f.write(format_summary_line(cfg_idx, raw, status,
                                               use_color=False) + '\n')
        detail_f.write('\n')

    print(f"Validated {len(new_tags)} new configs "
          f"({total_errors} errors, {total_warns} warnings)")
    print(f"Detail log: {detail_log}")
    print(f"Run with --sum to see full summary across all runs.")

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == '__main__':
    main()