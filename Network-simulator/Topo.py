"""
Topo.py — Containernet UAV swarm simulation with calibrated physical layer,
mobility library, and configurable attacks.

Reads one line from configurations.txt at the index given on the command
line (matches bash convention: `python3 Topo.py 42` reads line index 42),
parses it via simlib.config_parser, and runs the simulation.

Pipeline (per simulation window):
  1. Advance time by window_size seconds.
  2. Query mobility.position(uav, t_abs) for every UAV's new position.
  3. Run routing.build_links() to choose active drone-drone / drone-BS links
     under the per-cluster-bridge policy (saves UAV battery).
  4. For each active link, compute mean path loss (pathloss model) +
     correlated shadow fading (Gudmundson process), derive SNR from
     tx_power and noise_floor, choose modulation scheme (fixed or
     adaptive), compute PER and Mbps rate.
  5. Apply tc params (delay/bw/loss) to the link, then call TcVerifier
     to read back and log any mismatch.
  6. Sleep window_size seconds.

Modules under simlib/ are independently unit-tested. This file is the glue.
"""

import math
import multiprocessing
import os
import random
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')   # headless backend; no display required
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mininet.net import Containernet
from mininet.node import Controller
from mininet.cli import CLI                  # noqa: F401 (kept for optional CLI)
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401 (3d projection register)

# ----- simlib (calibrated physical / mobility / routing / verification) -----
# Topo.py's directory is auto-added to sys.path when run as a script, so
# `from simlib import ...` works when simlib/ sits next to this file.
from simlib.pathloss import PathLossModel
from simlib.shadow_fading import ShadowFadingField
from simlib.modulation import compute_link as compute_modulation_link
from simlib.mobility import MobilityGenerator
from simlib.routing import build_links as build_routing_links
from simlib.tc_verify import TcVerifier
from simlib.config_parser import parse_file_line


# ----------------------------- globals -----------------------------

setLogLevel('info')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

uav_vol = current_dir + '/UAV_data/:/UAV_data/'
code_vol = current_dir + '/Python/:/codes/'
simlib_vol = current_dir + '/simlib/:/codes/simlib/'   # so docker exec finds attacks_replay.py
dataset_vol = current_dir + '/pcaps/:/Datasets/'
where_to_save_logs = current_dir + '/pcaps/'

simulation_windows = 12       # number of mobility/relink cycles
window_size = 5               # seconds per cycle (mobility query dt)

# Physical-world bounds. Path-loss polynomial was fitted at altitudes
# 30/50/70/90/110 m, so we constrain UAV flight to that same range.
# This eliminates extrapolation and the resulting clamp warnings.
GRID_SIZE_M = 1000.0
UAV_ALT_MIN = 30.0
UAV_ALT_MAX = 110.0
HOVER_TRANSIT_FORCED_ALT = 50.0   # override fitted ascent/descent to keep airborne
SHADOW_CORRELATION_M = 50.0       # Gudmundson decorrelation distance

# JSON paths (calibrated outputs from the finetuning step).
LAYER1_JSON = os.path.join(project_root, 'finetuing_layers', 'outputs',
                           'layer1_comparison.json')
LAYER2_JSON = os.path.join(project_root, 'finetuing_layers', 'outputs',
                           'layer2a_mobility_library.json')

# Run-scoped diagnostics buffer (matches the old Topo.py logging style).
logs_to_write = ''


def _log(msg):
    """Append a timestamped line to the run diagnostics log."""
    global logs_to_write
    ts = time.clock_gettime_ns(time.CLOCK_REALTIME)
    logs_to_write += f'***{ts} {msg}\n'


# ----------------------------- nodes -----------------------------

def add_nodes(net, num_nodes, prefix, node_type, image="uav_nodes:latest"):
    nodes = []
    for i in range(1, num_nodes + 1):
        node_name = f'{prefix}{i}'
        if node_type == 'docker':
            node = net.addDocker(
                node_name, dimage=image,
                volumes=[uav_vol, code_vol, simlib_vol, dataset_vol],
            )
        else:
            node = net.addSwitch(node_name)
        nodes.append(node)
    return nodes


# ----------------------------- physical layer -----------------------------

def link_params_for(distance_m, uav_altitude_m, uav_id, bs_id, uav_pos,
                    pathloss_model, shadow_field, modulation_scheme,
                    tx_power_dbm, noise_floor_dbm, packet_bits=8000):
    """
    Compute (delay_ms, bw_mbit, loss_pct) for a single link.

    distance_m       : 3D slant distance between endpoints, meters
    uav_altitude_m   : the UAV-side endpoint's altitude (used by pathloss)
    uav_id, bs_id    : identifiers used as the (uav, peer) key for the
                       shadow fading process. For UAV-UAV links, bs_id is
                       the peer UAV's name. Each (a, b) link gets its own
                       independent shadowing process.
    uav_pos          : (x, y, z) for the shadowing displacement update
    """
    mean_pl = pathloss_model.mean_pathloss(distance_m, uav_altitude_m)
    shadow = shadow_field.sample(uav_id, bs_id, uav_pos, uav_altitude_m)
    total_pl_db = mean_pl + shadow

    snr_db = tx_power_dbm - total_pl_db - noise_floor_dbm

    res = compute_modulation_link(modulation_scheme, snr_db,
                                  packet_bits=packet_bits)
    bw_mbit = res['rate_mbps']

    # Delay = propagation + transmission. Distance/c for prop, packet/bw
    # for trans. Both small (microsecond-ish), but consistent with old code.
    prop_s = distance_m / 3.0e8
    trans_s = packet_bits / (bw_mbit * 1.0e6)
    delay_ms = (prop_s + trans_s) * 1000.0

    loss_pct = 100.0 * res['per']

    _log(f"link uav={uav_id}<->{bs_id} d={distance_m:.1f}m h={uav_altitude_m:.1f}m "
         f"PL={total_pl_db:.2f}dB SNR={snr_db:.2f}dB scheme={res['scheme']} "
         f"BER={res['ber']:.2e} PER={res['per']:.2%} bw={bw_mbit}Mb "
         f"delay={delay_ms:.3f}ms")

    return delay_ms, bw_mbit, loss_pct


# ----------------------------- placement -----------------------------

def _pick_mission_origin(mission_type, rng):
    """Pick an anchor position inside the grid with margin, per mission type."""
    if mission_type == 'random':
        x = rng.uniform(0.0, GRID_SIZE_M)
        y = rng.uniform(0.0, GRID_SIZE_M)
    else:
        # Deterministic patterns: keep margin so they don't run off the grid.
        x = rng.uniform(100.0, GRID_SIZE_M - 100.0)
        y = rng.uniform(100.0, GRID_SIZE_M - 100.0)
    return (x, y)


def init_positions(drones, basestations, missions_per_drone, mob_gen, rng):
    """
    Place basestations randomly on the ground (z=0). Register every UAV
    with the mobility generator at a fresh anchor. Returns:
        locations: dict {node_name -> (x, y, z)} at t=0
    """
    locations = {}

    # Basestations: random ground placement.
    for bs in basestations:
        x = rng.uniform(0.0, GRID_SIZE_M)
        y = rng.uniform(0.0, GRID_SIZE_M)
        locations[bs.name] = (x, y, 0.0)
        _log(f"placed BS {bs.name} at ({x:.1f}, {y:.1f}, 0.0)")

    # UAVs: register with mobility generator using per-mission anchor.
    for drone, mission in zip(drones, missions_per_drone):
        origin_xy = _pick_mission_origin(mission, rng)
        kwargs = dict(
            uav_id=drone.name,
            mission_type=mission,
            origin_xy=origin_xy,
            grid_size_m=GRID_SIZE_M,
            alt_min=UAV_ALT_MIN,
            alt_max=UAV_ALT_MAX,
        )
        if mission == 'hover_transit':
            kwargs['altitude_override'] = HOVER_TRANSIT_FORCED_ALT
        elif mission == 'random':
            kwargs['initial_z'] = rng.uniform(UAV_ALT_MIN, 100.0)
        mob_gen.create_uav(**kwargs)
        x, y, z = mob_gen.position(drone.name, 0.0)
        locations[drone.name] = (x, y, z)
        _log(f"placed UAV {drone.name} mission={mission} origin=({origin_xy[0]:.1f},{origin_xy[1]:.1f}) "
             f"start=({x:.1f},{y:.1f},{z:.1f})")

    return locations


# ----------------------------- topology -----------------------------

def add_base_links(net, drones, switches, basestations, basestation_switches):
    """
    Wire the static skeleton:
      - drone <-> own switch
      - bs <-> own switch
      - all bs-switches in a chain (so any-to-any BS reachable)
      - every (drone-switch, bs-switch) pair pre-created, kept DOWN
      - every (drone-switch, drone-switch) pair pre-created, kept DOWN
    Dynamic links toggle UP/DOWN per window via configLinkStatus.
    """
    for drone, switch in zip(drones, switches):
        net.addLink(drone, switch)

    for bs, bs_switch in zip(basestations, basestation_switches):
        net.addLink(bs, bs_switch)

    for i in range(len(basestation_switches) - 1):
        net.addLink(basestation_switches[i], basestation_switches[i + 1])

    for switch in switches:
        for bs_switch in basestation_switches:
            net.addLink(switch, bs_switch)
            net.configLinkStatus(switch.name, bs_switch.name, 'down')

    for i, switch1 in enumerate(switches):
        for switch2 in switches[i + 1:]:
            net.addLink(switch1, switch2)
            net.configLinkStatus(switch1.name, switch2.name, 'down')


def _switch_for_drone(drone_name, drones, switches):
    """drone 'd3' -> switch 's3'"""
    idx = next(i for i, d in enumerate(drones) if d.name == drone_name)
    return switches[idx]


def _switch_for_bs(bs_name, basestations, basestation_switches):
    """bs 'bs1' -> switch 'bss1'"""
    idx = next(i for i, b in enumerate(basestations) if b.name == bs_name)
    return basestation_switches[idx]


def _names_to_switches(name_a, name_b, drones, switches,
                       basestations, basestation_switches):
    """Resolve a (name_a, name_b) pair from routing into the corresponding
    switch objects so configLinkStatus / link.intf can act on them."""
    def resolve(n):
        if n.startswith('bs'):
            return _switch_for_bs(n, basestations, basestation_switches)
        return _switch_for_drone(n, drones, switches)
    return resolve(name_a), resolve(name_b)


# ----------------------------- relink -----------------------------

def relink(net, prev_dynamic_links, drones, switches,
           basestations, basestation_switches, locations,
           pathloss_model, shadow_field, modulation_scheme,
           tx_power_dbm, noise_floor_dbm,
           tc_verifier,
           wormhole=False, wormhole_attacker=None, wormhole_victims=None):
    """
    Tear down previous dynamic links, ask routing for the new set, bring
    them up, and push fresh tc params on the underlying veth pairs.

    Returns the new list of (switch_a, switch_b) tuples (mirrors the old API).
    """
    # 1. tear down previous
    for sw_a, sw_b in prev_dynamic_links:
        net.configLinkStatus(sw_a.name, sw_b.name, 'down')

    # 2. ask routing for the new link set
    uav_names = [d.name for d in drones]
    bs_names = [b.name for b in basestations]
    uav_positions = {n: locations[n] for n in uav_names}
    bs_positions = {n: locations[n] for n in bs_names}

    routing_result = build_routing_links(
        uav_names=uav_names, uav_positions=uav_positions,
        bs_names=bs_names, bs_positions=bs_positions,
        wormhole=wormhole,
        wormhole_attacker=wormhole_attacker,
        wormhole_victims=wormhole_victims,
    )

    _log(f"routing: total_links={routing_result['diagnostics']['total_links']} "
         f"cluster_bridges_added={routing_result['diagnostics']['cluster_bridges_added']} "
         f"bridge_uavs={routing_result['forced_bs_links']}")

    # 3. bring new links up; capture switch-pair tuples for next teardown
    new_dynamic_links = []
    for name_a, name_b in routing_result['links']:
        sw_a, sw_b = _names_to_switches(
            name_a, name_b, drones, switches, basestations, basestation_switches,
        )
        net.configLinkStatus(sw_a.name, sw_b.name, 'up')
        new_dynamic_links.append((sw_a, sw_b))

    # 4. Per-routing-link physics → aggregate to per-container worst case
    #    → apply tc once per container interface.
    #
    # Each container (UAV or BS) has ONE eth0 interface that carries all its
    # traffic, but it may participate in multiple routing-active links per
    # window (e.g. a bridge UAV on a chain). A single eth0 can hold only one
    # netem qdisc, so we aggregate "worst case" across that container's
    # active links: max(loss_pct), max(delay_ms), min(bw_mbit). This models
    # a radio bottlenecked by its weakest active connection.
    #
    # Switch-to-switch links inside the topology cannot carry netem with
    # TCLink, so we don't try to apply anything there — the per-container
    # eth0 shaping at each endpoint covers what the data path actually sees.

    # Build container-name -> container-side TCLink interface lookup once.
    # The drone-switch and bs-switch links created in add_base_links() are
    # the ones whose UAV/BS-side interface is what we want to shape.
    container_intf = {}   # container_node_name -> intf object
    for link in net.links:
        for intf in (link.intf1, link.intf2):
            n = intf.node.name
            if _container_for(n) is not None and n not in container_intf:
                container_intf[n] = intf

    # worst[uav_or_bs_name] = {'delay_ms': float, 'bw_mbit': float, 'loss_pct': float}
    worst = {}

    def _update_worst(node_name, delay_ms, bw_mbit, loss_pct):
        cur = worst.get(node_name)
        if cur is None:
            worst[node_name] = {
                'delay_ms': delay_ms, 'bw_mbit': bw_mbit, 'loss_pct': loss_pct,
            }
        else:
            cur['delay_ms'] = max(cur['delay_ms'], delay_ms)
            cur['bw_mbit'] = min(cur['bw_mbit'], bw_mbit)
            cur['loss_pct'] = max(cur['loss_pct'], loss_pct)

    for name_a, name_b in routing_result['links']:
        if name_a not in locations or name_b not in locations:
            continue

        # For shadow-fading state we need a stable (uav_id, peer_id) key and
        # the UAV-side altitude for path loss. UAV↔BS: the UAV side gives
        # altitude. UAV↔UAV: pick alphabetical for stability.
        is_a_uav = name_a.startswith('d')
        is_b_uav = name_b.startswith('d')
        if is_a_uav and not is_b_uav:
            uav_id, peer_id = name_a, name_b
        elif is_b_uav and not is_a_uav:
            uav_id, peer_id = name_b, name_a
        else:
            uav_id, peer_id = (name_a, name_b) if name_a < name_b else (name_b, name_a)

        uav_pos = locations[uav_id]
        uav_alt = uav_pos[2]
        d_m = math.dist(locations[name_a], locations[name_b])

        delay_ms, bw_mbit, loss_pct = link_params_for(
            distance_m=d_m, uav_altitude_m=uav_alt,
            uav_id=uav_id, bs_id=peer_id, uav_pos=uav_pos,
            pathloss_model=pathloss_model, shadow_field=shadow_field,
            modulation_scheme=modulation_scheme,
            tx_power_dbm=tx_power_dbm, noise_floor_dbm=noise_floor_dbm,
        )

        # Both endpoints share the same physical link → both their eth0s
        # see the same params on this link. Update worst-case for each.
        _update_worst(name_a, delay_ms, bw_mbit, loss_pct)
        _update_worst(name_b, delay_ms, bw_mbit, loss_pct)

    # Apply once per container interface and verify.
    for node_name, params in worst.items():
        intf = container_intf.get(node_name)
        if intf is None:
            _log(f"WARN no container interface for {node_name}; skipping tc apply")
            continue
        delay_str = f"{params['delay_ms']:.3f}ms"
        try:
            intf.config(delay=delay_str, bw=params['bw_mbit'], loss=params['loss_pct'])
        except Exception as e:
            _log(f"intf.config FAILED on {intf.name} ({node_name}): {e}")
            continue

        _log(f"applied tc on {node_name}/{intf.name}: "
             f"delay={params['delay_ms']:.3f}ms "
             f"bw={params['bw_mbit']}Mb "
             f"loss={params['loss_pct']:.4f}%")

        if tc_verifier is not None:
            container = _container_for(node_name)
            tc_verifier.verify(
                intf.name,
                intended_delay_ms=params['delay_ms'],
                intended_rate_mbit=float(params['bw_mbit']),
                intended_loss_pct=params['loss_pct'],
                container=container,
            )

    return new_dynamic_links


def _container_for(node_name):
    """
    Return the docker-exec container name for a node, or None if the node
    lives on the host (i.e., is an OVS switch).

    UAVs have node names like 'd1', 'd20'.       -> 'mn.d1', 'mn.d20'
    Basestations have names like 'bs1', 'bs2'.   -> 'mn.bs1', 'mn.bs2'
    Switches have names like 's1', 'bss1'.       -> None (host namespace)
    """
    if not node_name:
        return None
    # Drone: 'd' followed by digits
    if node_name.startswith('d') and node_name[1:].isdigit():
        return 'mn.' + node_name
    # Basestation: 'bs' followed by digits (NOT 'bss', which is a switch)
    if (node_name.startswith('bs')
            and not node_name.startswith('bss')
            and node_name[2:].isdigit()):
        return 'mn.' + node_name
    return None


def _switch_to_node_name(switch_name):
    """
    Map switch name -> the data-plane node it represents:
      's3'   -> 'd3'    (drone)
      'bss2' -> 'bs2'   (basestation)
    Returns None for control-plane / unknown names.
    """
    if switch_name.startswith('bss'):
        return 'bs' + switch_name[3:]
    if switch_name.startswith('s'):
        # exclude 'bss' prefix already handled; plain 's<n>' -> 'd<n>'
        return 'd' + switch_name[1:]
    return None


# ----------------------------- diagnostic plot -----------------------------

def plot_network(net, locations, relink_num):
    """
    3D scatter of nodes + their currently-up data-plane edges. Saved to
    network<relink_num>.png in cwd. Cosmetic only; doesn't affect the
    simulation.
    """
    G = nx.Graph()
    pos = {node: locations[node] for node in locations}

    for node in locations:
        G.add_node(node, type='drone' if node.startswith('d') else 'bs')

    for link in net.links:
        if link.intf1.isUp() and link.intf2.isUp():
            a = _switch_to_node_name(link.intf1.node.name)
            b = _switch_to_node_name(link.intf2.node.name)
            if a in pos and b in pos:
                G.add_edge(a, b)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for node, (x, y, z) in pos.items():
        color = 'blue' if G.nodes[node]['type'] == 'drone' else 'red'
        ax.scatter(x, y, z, color=color, s=80)
        ax.text(x, y, z, node, fontsize=8)

    for edge in G.edges:
        x_vals, y_vals, z_vals = zip(pos[edge[0]], pos[edge[1]])
        ax.plot(x_vals, y_vals, z_vals, color='black', linewidth=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_xlim(0, GRID_SIZE_M)
    ax.set_ylim(0, GRID_SIZE_M)
    ax.set_zlim(0, UAV_ALT_MAX + 20)
    plt.savefig(f'network{relink_num}.png')
    plt.close(fig)


# ----------------------------- traffic / FTP -----------------------------

def start_tcpdump(node_name, attack_details_time):
    cmd = f"docker exec {node_name} tcpdump -i any -w /Datasets/{node_name}-{attack_details_time}.pcap"
    subprocess.Popen(cmd, shell=True)


def start_FTP_server(node_name):
    cmd = f"docker exec {node_name} nginx -c /codes/FTP/nginxmn.conf"
    subprocess.Popen(cmd, shell=True)


def start_data_transfer(node_ip, bs_name, payload):
    """Run the FTP client on the assigned BS, fetching from a UAV's FTP server."""
    cmd = (f"docker exec mn.{bs_name} "
           f"python3 /codes/FTP/FTP_client.py {node_ip} {payload}")
    subprocess.Popen(cmd, shell=True)


def stop_tcpdump(node_name):
    cmd = f"docker exec {node_name} pkill -f tcpdump"
    subprocess.run(cmd, shell=True)


# ----------------------------- attacks -----------------------------

def start_dos_attack(attack_node_name, victim_node_ip, attack_length, attack_rate):
    if attack_rate == 1:
        cmd = (f"docker exec {attack_node_name} "
               f"timeout {attack_length}s hping3 -S --flood -p 8888 {victim_node_ip}")
    else:
        cmd = (f"docker exec {attack_node_name} "
               f"timeout {attack_length}s hping3 -S -i u{attack_rate} -q -p 8888 {victim_node_ip}")
    subprocess.Popen(cmd, shell=True)


def start_blackhole_ovs(switch_name, drone_mac, attack_length):
    """
    Smarter blackhole: install a high-priority OVS flow rule on the drone's
    switch that drops anything to/from the drone's MAC. The drone's link
    stays UP (so it appears connected, ARP still resolves), but data
    silently disappears. After attack_length seconds, remove the rules.

    Runs on the host; OVS lives there, not inside the container.
    """
    cookie = '0xbeef'   # tag rules so we can clean them up surgically
    add_src = (
        f"ovs-ofctl add-flow {switch_name} "
        f"cookie={cookie},priority=65535,dl_src={drone_mac},actions=drop"
    )
    add_dst = (
        f"ovs-ofctl add-flow {switch_name} "
        f"cookie={cookie},priority=65535,dl_dst={drone_mac},actions=drop"
    )
    del_by_cookie = (
        f"ovs-ofctl --strict del-flows {switch_name} "
        f"cookie={cookie}/-1"
    )
    subprocess.run(add_src, shell=True)
    subprocess.run(add_dst, shell=True)
    time.sleep(attack_length)
    subprocess.run(del_by_cookie, shell=True)


def start_replay_attack(attacker_name, attack_length, params, victim_ip=None):
    """
    Spawn the simlib replay attacker inside the attacker's container.
    `params` is the parsed dict from config_parser:
        {'type':'replay', 'buffer_packets':..., 'delay_ms':...,
         'rate_pps':..., 'ttl_decrement':..., 'seq_mode':...}
    """
    # Scapy can't use the pseudo-interface 'any' inside Docker containers.
    # Derive the real interface name: 'mn.d4' -> 'd4-eth0'
    node_name = attacker_name.replace('mn.', '')
    iface = f"{node_name}-eth0"
    victim_arg = f"--victim-ip {victim_ip}" if victim_ip else ""
    cmd = (
        f"docker exec {attacker_name} "
        f"python3 /codes/simlib/attacks_replay.py "
        f"--iface {iface} --duration {attack_length} "
        f"--buffer {params['buffer_packets']} "
        f"--delay-ms {params['delay_ms']} "
        f"--rate-pps {params['rate_pps']} "
        f"--ttl-dec {params['ttl_decrement']} "
        f"--seq-mode {params['seq_mode']} "
        f"{victim_arg}"
    )
    subprocess.Popen(cmd, shell=True)


def dispatch_attacks(attacks, drones, switches, num_drones):
    """
    Iterate the parsed attacks list and launch each. Returns:
        (wormhole, attacker_name_or_none, victim_names_list)
    so the caller can pass these to relink() each window.
    """
    wormhole = False
    wormhole_attacker = None
    wormhole_victims = []

    for atk in attacks:
        atype = atk['type']

        if atype == 'benign':
            continue

        if atype == 'dos':
            attack_node = random.choice(drones)
            victim_node = random.choice(drones)
            while attack_node == victim_node:
                victim_node = random.choice(drones)
            attack_length = random.randint(10, 20)
            attack_rate = atk['interval_us']
            attacker_name = 'mn.' + attack_node.name
            victim_ip = victim_node.IP()
            print(f"DoS: {attack_node.name} -> {victim_node.name} "
                  f"rate(us)={attack_rate} len={attack_length}")
            multiprocessing.Process(
                target=start_dos_attack,
                args=(attacker_name, victim_ip, attack_length, attack_rate),
            ).start()
            _log(f"DoS attacker={attack_node.name} victim={victim_node.name} "
                 f"rate={attack_rate} length={attack_length}")

        elif atype == 'ddos':
            n_attackers = random.randint(2, max(2, num_drones - 1))
            attack_nodes = random.sample(drones, n_attackers)
            victim_node = random.choice(drones)
            while victim_node in attack_nodes:
                victim_node = random.choice(drones)
            attack_length = random.randint(10, 20)
            attack_rate = atk['interval_us']
            victim_ip = victim_node.IP()
            print(f"DDoS: {[n.name for n in attack_nodes]} -> {victim_node.name}")
            for n in attack_nodes:
                multiprocessing.Process(
                    target=start_dos_attack,
                    args=('mn.' + n.name, victim_ip, attack_length, attack_rate),
                ).start()
            _log(f"DDoS attackers={[n.name for n in attack_nodes]} "
                 f"victim={victim_node.name} rate={attack_rate} length={attack_length}")

        elif atype == 'blackhole':
            n_victims = random.randint(1, max(1, num_drones - 1))
            victim_drones = random.sample(drones, n_victims)
            attack_length = random.randint(10, 20)
            print(f"Blackhole victims: {[d.name for d in victim_drones]}")
            for d in victim_drones:
                sw = _switch_for_drone(d.name, drones, switches)
                drone_mac = d.MAC()
                multiprocessing.Process(
                    target=start_blackhole_ovs,
                    args=(sw.name, drone_mac, attack_length),
                ).start()
            _log(f"Blackhole victims={[d.name for d in victim_drones]} "
                 f"length={attack_length}")

        elif atype == 'wormhole':
            n_victims = random.randint(1, max(1, num_drones - 1))
            attacker = random.choice(drones)
            victims = random.sample(drones, n_victims)
            while attacker in victims:
                victims = random.sample(drones, n_victims)
            wormhole = True
            wormhole_attacker = attacker.name
            wormhole_victims = [v.name for v in victims]
            print(f"Wormhole attacker={attacker.name} victims={wormhole_victims}")
            _log(f"Wormhole attacker={attacker.name} victims={wormhole_victims}")

        elif atype == 'replay':
            attacker = random.choice(drones)
            victim = random.choice(drones)
            while victim == attacker:
                victim = random.choice(drones)
            attack_length = random.randint(10, 20)
            attacker_name = 'mn.' + attacker.name
            victim_ip = victim.IP()
            print(f"Replay attacker={attacker.name} victim={victim.name} "
                  f"len={attack_length} params={atk}")
            multiprocessing.Process(
                target=start_replay_attack,
                args=(attacker_name, attack_length, atk, victim_ip),
            ).start()
            _log(f"Replay attacker={attacker.name} victim={victim.name} "
                 f"length={attack_length} params={atk}")

        else:
            print(f"WARN: unknown attack type {atype!r}; skipping")
            _log(f"WARN unknown attack type {atype!r}")

    return wormhole, wormhole_attacker, wormhole_victims


# ----------------------------- main -----------------------------

def main():
    global logs_to_write

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config_line_index>", file=sys.stderr)
        sys.exit(2)

    config_number = int(sys.argv[1])
    attack_details_time = time.strftime("%Y%m%d-%H%M") + '-' + str(config_number)

    # Parse config line via simlib (handles old 4-field and new 9-field formats).
    config_file = os.path.join(current_dir, 'configurations.txt')
    scenario = parse_file_line(config_file, config_number, rng=random.Random())
    print(f"=== Scenario (line {config_number}) ===")
    print(f"  raw     : {scenario['raw']}")
    print(f"  format  : {scenario['format']}")
    print(f"  attacks : {scenario['attacks']}")
    print(f"  drones  : {scenario['num_drones']}  basestations: {scenario['num_basestations']}")
    print(f"  payload : {scenario['payload']}  pathloss: {scenario['pathloss_model']}  "
          f"modulation: {scenario['modulation_scheme']}")
    print(f"  missions: {scenario['missions']}")
    print(f"  tx={scenario['tx_power_dBm']}dBm  noise={scenario['noise_floor_dBm']}dBm")

    _log(f"scenario raw={scenario['raw']!r} format={scenario['format']}")

    # ------------- physical layer & mobility -------------
    pathloss_model = PathLossModel(scenario['pathloss_model'], LAYER1_JSON)
    sim_rng = random.Random()  # fresh per invocation
    shadow_field = ShadowFadingField(pathloss_model,
                                     d_corr_m=SHADOW_CORRELATION_M,
                                     rng=sim_rng)
    mob_gen = MobilityGenerator(LAYER2_JSON, rng=sim_rng)

    tc_log_path = os.path.join(where_to_save_logs,
                               f'tc_diagnostics_{attack_details_time}.log')
    tc_verifier = TcVerifier(log_path=tc_log_path)

    # ------------- topology -------------
    # link=TCLink ensures every net.addLink() returns a TCLink, which is what
    # actually carries the netem qdisc inside the container. Without this,
    # Containernet may default to plain Link with no tc support, and our
    # path-loss-driven delay/bw/loss values would never reach the wire.
    net = Containernet(controller=Controller, link=TCLink)
    info('*** Adding controller\n')
    net.addController('c0')

    drones = add_nodes(net, scenario['num_drones'], 'd', 'docker')
    switches = add_nodes(net, scenario['num_drones'], 's', 'switch')
    basestations = add_nodes(net, scenario['num_basestations'], 'bs', 'docker')
    basestation_switches = add_nodes(net, scenario['num_basestations'],
                                     'bss', 'switch')

    # placement at t=0
    locations = init_positions(drones, basestations, scenario['missions'],
                               mob_gen, sim_rng)

    # static link skeleton (all dynamic links pre-created and held DOWN)
    add_base_links(net, drones, switches, basestations, basestation_switches)

    info('*** Starting network\n')
    net.start()
    _log(f"network started drones={scenario['num_drones']} bs={scenario['num_basestations']}")
    _log(f"initial locations: {locations}")

    # initial relink — wormhole flags not known yet (attacks not dispatched);
    # we'll re-relink after attacks are launched so wormhole topology kicks in.
    dynamic_links = relink(
        net, [], drones, switches, basestations, basestation_switches,
        locations, pathloss_model, shadow_field,
        scenario['modulation_scheme'],
        scenario['tx_power_dBm'], scenario['noise_floor_dBm'],
        tc_verifier,
    )
    plot_network(net, locations, 0)

    # ------------- traffic infrastructure -------------
    info('*** Starting tcpdump on all nodes\n')
    node_names = ['mn.' + n.name for n in drones + basestations]
    with multiprocessing.Pool(processes=len(node_names)) as pool:
        pool.starmap(start_tcpdump,
                     [(n, attack_details_time) for n in node_names])
    _log(f"tcpdump started on {node_names}")

    info('*** Starting FTP servers on UAVs\n')
    drone_node_names = ['mn.' + n.name for n in drones]
    with multiprocessing.Pool(processes=len(drone_node_names)) as pool:
        pool.map(start_FTP_server, drone_node_names)
    _log(f"FTP servers started on {drone_node_names}")

    # Round-robin assign each drone to a BS for FTP fetching (fixes old bs1 hardcode)
    bs_names = [b.name for b in basestations]
    transfer_pairs = [
        (d.IP(), bs_names[i % len(bs_names)], scenario['payload'])
        for i, d in enumerate(drones)
    ]
    info('*** Starting data transfer (round-robin BS assignment)\n')
    with multiprocessing.Pool(processes=len(transfer_pairs)) as pool:
        pool.starmap(start_data_transfer, transfer_pairs)
    _log(f"data transfer pairs (drone_ip, bs, payload): {transfer_pairs}")

    info('*** pingAll to populate ARP\n')
    net.pingAll()

    # ------------- attacks -------------
    info('*** Dispatching attacks\n')
    wormhole, wh_attacker, wh_victims = dispatch_attacks(
        scenario['attacks'], drones, switches, scenario['num_drones']
    )

    # If wormhole was set, relink immediately so the wormhole topology
    # takes effect before the simulation loop begins.
    if wormhole:
        dynamic_links = relink(
            net, dynamic_links, drones, switches, basestations,
            basestation_switches, locations,
            pathloss_model, shadow_field,
            scenario['modulation_scheme'],
            scenario['tx_power_dBm'], scenario['noise_floor_dBm'],
            tc_verifier,
            wormhole=True, wormhole_attacker=wh_attacker,
            wormhole_victims=wh_victims,
        )

    # ------------- simulation loop -------------
    info('*** Running simulation\n')
    t_abs = 0.0
    for i in range(simulation_windows):
        time.sleep(window_size)
        t_abs += window_size

        # Advance every UAV via mobility generator.
        for d in drones:
            x, y, z = mob_gen.position(d.name, t_abs)
            locations[d.name] = (x, y, z)

        _log(f"t_abs={t_abs:.1f}s positions: {locations}")

        dynamic_links = relink(
            net, dynamic_links, drones, switches, basestations,
            basestation_switches, locations,
            pathloss_model, shadow_field,
            scenario['modulation_scheme'],
            scenario['tx_power_dBm'], scenario['noise_floor_dBm'],
            tc_verifier,
            wormhole=wormhole, wormhole_attacker=wh_attacker,
            wormhole_victims=wh_victims,
        )
        plot_network(net, locations, i + 1)

    # ------------- shutdown -------------
    info('*** Stopping tcpdump\n')
    with multiprocessing.Pool(processes=len(node_names)) as pool:
        pool.map(stop_tcpdump, node_names)

    # diagnostics summaries
    pl_diag = pathloss_model.diagnostics()
    tc_summary = tc_verifier.summarize()
    _log(f"pathloss diagnostics: {pl_diag}")
    _log(f"tc_verify summary: {tc_summary}")
    print(f"=== End of run ===")
    print(f"  pathloss clamp counts: {pl_diag}")
    print(f"  tc_verify             : {tc_summary}")

    info('*** Stopping network\n')
    net.stop()

    # save run log
    log_path = os.path.join(where_to_save_logs,
                            f'attack_details_{attack_details_time}.txt')
    with open(log_path, 'w') as f:
        f.write(logs_to_write)
    print(f"Run log written to {log_path}")
    print(f"tc diagnostics in   {tc_log_path}")


if __name__ == '__main__':
    # --- Layer-3 trajectory replay short-circuit ---------------------
    # Bypasses Containernet entirely. Just runs the path-loss + shadow
    # + modulation pipeline against a measured trajectory and emits a
    # per-step physics CSV. No docker / no root required.
    if '--layer3' in sys.argv:
        try:
            flight_id = sys.argv[sys.argv.index('--layer3') + 1]
        except IndexError:
            print("Usage: Topo.py --layer3 <flight_id>", file=sys.stderr)
            sys.exit(2)
        sys.path.insert(0, current_dir)
        from layer3_run import run_layer3
        run_layer3(flight_id)
        sys.exit(0)
    # -----------------------------------------------------------------
    main()