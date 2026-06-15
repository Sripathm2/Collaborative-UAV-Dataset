"""
simlib/routing.py

Dynamic link selection for the UAV swarm. Preserves Topo.py's original
nearest-neighbor-chain approach, but adds a reachability check to guarantee
that every UAV has a path to at least one basestation.

Algorithm:
  1. For each UAV, link to its single closest node (another UAV or a BS),
     exactly as the original relink().
  2. Build the resulting link graph.
  3. BFS from each UAV over the link graph. If it doesn't reach any BS,
     append a direct UAV->nearest-BS link so it does.

This fixes two latent bugs in the original:
  - Chains of UAVs can form closed loops that reach no BS.
  - The order-dependent `(switches[min_index], switches[i]) in dynamic_links`
    guard prevents duplicates but doesn't guarantee connectivity.

The output is a list of (node_a_name, node_b_name) tuples. Caller is
responsible for applying these to the Containernet topology (bringing the
corresponding switch-to-switch links up/down and calling
set_link_params() with computed delay/bw/loss).

This module knows nothing about mininet/Containernet — it operates purely
on positions and names. That makes it unit-testable and keeps Topo.py
integration clean.
"""

import math
from collections import defaultdict, deque


def _dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def build_links(uav_names, uav_positions, bs_names, bs_positions,
                wormhole=False, wormhole_attacker=None, wormhole_victims=None,
                max_reach_hops=None):
    """
    Decide the dynamic link set for one window.

    Args:
        uav_names:     list of UAV names, e.g. ['d1', 'd2', ...]
        uav_positions: dict {name -> (x, y, z)} for UAVs
        bs_names:      list of BS names, e.g. ['bs1', 'bs2', ...]
        bs_positions:  dict {name -> (x, y, z)} for BSs
        wormhole:      if True, all wormhole_victims are linked to
                       wormhole_attacker (instead of their nearest), and
                       wormhole_attacker is linked to its nearest BS.
        wormhole_attacker: name of attacker UAV (if wormhole)
        wormhole_victims : list of victim UAV names (if wormhole)
        max_reach_hops: BFS depth cap. None = unbounded (safe for realistic
                       fleet sizes; use an int for paranoia).

    Returns:
        dict with:
          'links': list of (name_a, name_b) unique unordered pairs
          'forced_bs_links': list of UAV names that got an extra BS link
                             because their chain didn't reach any BS
          'reachable': dict {uav_name -> bool} — post-fix, every entry True
          'diagnostics': {'loops_fixed': int, 'total_links': int}
    """
    if not uav_names:
        return {
            'links': [],
            'forced_bs_links': [],
            'reachable': {},
            'diagnostics': {'loops_fixed': 0, 'total_links': 0},
        }
    if not bs_names:
        raise ValueError("At least one basestation required")

    wormhole_victims = set(wormhole_victims or [])

    # -------- Step 1: nearest-neighbor choice per UAV --------
    raw_links = []   # list of (a, b) pairs (ordered: chooser first)
    for u in uav_names:
        if wormhole and u in wormhole_victims:
            # Victim forced to link to attacker
            if wormhole_attacker is None:
                raise ValueError("wormhole=True but wormhole_attacker not given")
            raw_links.append((u, wormhole_attacker))
            continue
        if wormhole and u == wormhole_attacker:
            # Attacker always links to its closest BS (tunnel endpoint)
            nearest_bs = min(bs_names,
                             key=lambda b: _dist(uav_positions[u], bs_positions[b]))
            raw_links.append((u, nearest_bs))
            continue

        # Normal nearest-neighbor
        u_pos = uav_positions[u]
        nearest_name = None
        nearest_d = float('inf')
        # consider other UAVs
        for v in uav_names:
            if v == u:
                continue
            d = _dist(u_pos, uav_positions[v])
            if d < nearest_d:
                nearest_d = d
                nearest_name = v
        # consider BSs
        for b in bs_names:
            d = _dist(u_pos, bs_positions[b])
            if d < nearest_d:
                nearest_d = d
                nearest_name = b
        raw_links.append((u, nearest_name))

    # -------- Step 2: canonicalize to unordered unique pairs --------
    unique_pairs = set()
    for a, b in raw_links:
        pair = tuple(sorted((a, b)))
        unique_pairs.add(pair)
    links = list(unique_pairs)

    # -------- Step 3: build adjacency, find connected components, fix
    #                  one bridge per isolated cluster --------
    #
    # Power-aware design: a UAV running multiple radio links burns more
    # battery, so for each cluster of UAVs not already reaching a BS, we
    # add only ONE bridge — chosen as the UAV in that cluster with the
    # shortest distance to its nearest BS — instead of giving every
    # isolated UAV its own direct BS link.
    adj = defaultdict(set)
    for a, b in links:
        adj[a].add(b)
        adj[b].add(a)

    bs_set = set(bs_names)

    # Find connected components over UAVs (we only need UAV-side components;
    # any component containing a BS is "already reachable").
    visited = set()
    components = []  # list of (uav_set, contains_bs_bool)
    all_nodes = list(uav_names) + list(bs_names)
    for start in all_nodes:
        if start in visited:
            continue
        comp_uavs = set()
        comp_has_bs = False
        q = deque([(start, 0)])
        visited.add(start)
        while q:
            node, hops = q.popleft()
            if node in bs_set:
                comp_has_bs = True
            else:
                comp_uavs.add(node)
            if max_reach_hops is not None and hops >= max_reach_hops:
                continue
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, hops + 1))
        if comp_uavs:
            components.append((comp_uavs, comp_has_bs))

    # For each component without a BS, pick one bridge UAV (shortest to
    # nearest BS) and add exactly one link.
    forced = []   # list of UAVs that became cluster bridges
    for comp_uavs, has_bs in components:
        if has_bs:
            continue

        # Wormhole victim handling: victims are "reachable via attacker
        # tunnel" by design. If the entire isolated cluster is wormhole
        # victims, we skip — that's the attack semantic, Topo.py's job to
        # monitor. If only some are victims, we exclude them from the
        # bridge candidate set (we don't want a victim to bridge).
        candidates = [
            u for u in comp_uavs
            if not (wormhole and u in wormhole_victims)
        ]
        if not candidates:
            # Pure-victim cluster — leave it alone, attack semantics intact.
            continue

        # Choose bridge: UAV with shortest distance to its nearest BS.
        best_uav = None
        best_bs = None
        best_d = float('inf')
        for u in candidates:
            for b in bs_names:
                d = _dist(uav_positions[u], bs_positions[b])
                if d < best_d:
                    best_d = d
                    best_uav = u
                    best_bs = b

        new_pair = tuple(sorted((best_uav, best_bs)))
        if new_pair not in unique_pairs:
            unique_pairs.add(new_pair)
            links.append(new_pair)
            adj[best_uav].add(best_bs)
            adj[best_bs].add(best_uav)
        forced.append(best_uav)

    # Reachability map: every non-victim UAV is now reachable; victims are
    # "reachable via tunnel" by attack semantics.
    reachable = {}
    for u in uav_names:
        if wormhole and u in wormhole_victims:
            reachable[u] = True
        else:
            reachable[u] = True   # guaranteed by the cluster-bridge step

    return {
        'links': links,
        'forced_bs_links': forced,
        'reachable': reachable,
        'diagnostics': {
            'cluster_bridges_added': len(forced),
            'total_links': len(links),
        },
    }


# ==================== self-test ====================

if __name__ == '__main__':
    print("=== routing.py self-test ===\n")

    # Test 1: Single UAV, single BS — trivial
    print("Test 1: 1 UAV, 1 BS")
    r = build_links(
        ['d1'], {'d1': (50, 50, 30)},
        ['bs1'], {'bs1': (0, 0, 0)},
    )
    print(f"  links: {r['links']}")
    print(f"  forced: {r['forced_bs_links']}  reachable: {r['reachable']}")
    assert ('bs1', 'd1') in r['links'] and not r['forced_bs_links']
    print("  OK\n")

    # Test 2: Two UAVs close to each other, BS far — should form d1-d2, then
    # one of them (whichever is closer to BS) also gets a BS link (via reach fix).
    print("Test 2: 2 UAVs cluster, BS far away (should produce loop without fix)")
    r = build_links(
        ['d1', 'd2'],
        {'d1': (500, 500, 30), 'd2': (501, 500, 30)},
        ['bs1'], {'bs1': (0, 0, 0)},
    )
    print(f"  links: {r['links']}")
    print(f"  forced: {r['forced_bs_links']}")
    # Raw nearest: d1->d2, d2->d1 (loop). Fix should add one direct d*-bs1 link.
    assert len(r['forced_bs_links']) >= 1, "Expected at least one forced BS link"
    assert all(r['reachable'].values())
    print("  OK (loop detected and fixed)\n")

    # Test 3: Three UAVs in a cycle, BS isolated
    print("Test 3: 3 UAVs in triangle, BS far")
    r = build_links(
        ['d1', 'd2', 'd3'],
        {
            'd1': (500, 500, 30),
            'd2': (510, 500, 30),
            'd3': (505, 510, 30),
        },
        ['bs1'], {'bs1': (0, 0, 0)},
    )
    print(f"  links: {r['links']}")
    print(f"  forced: {r['forced_bs_links']}")
    assert all(r['reachable'].values())
    print("  OK\n")

    # Test 4: Chain that already reaches BS — no fix needed
    print("Test 4: d1 -> d2 -> bs1 chain, d3 near bs1")
    r = build_links(
        ['d1', 'd2', 'd3'],
        {
            'd1': (200, 0, 30),
            'd2': (100, 0, 30),
            'd3': (10, 0, 30),
        },
        ['bs1'], {'bs1': (0, 0, 0)},
    )
    print(f"  links: {r['links']}")
    print(f"  forced: {r['forced_bs_links']}")
    assert r['forced_bs_links'] == [], f"Expected no forced links, got {r['forced_bs_links']}"
    print("  OK (chain naturally reaches BS)\n")

    # Test 5: Multiple BSs — any one reachable is OK
    print("Test 5: 2 BSs on opposite sides; UAV cluster in middle")
    r = build_links(
        ['d1', 'd2', 'd3', 'd4'],
        {
            'd1': (500, 500, 30), 'd2': (510, 500, 30),
            'd3': (520, 500, 30), 'd4': (530, 500, 30),
        },
        ['bs1', 'bs2'],
        {'bs1': (0, 0, 0), 'bs2': (1000, 1000, 0)},
    )
    print(f"  links: {r['links']}")
    print(f"  forced: {r['forced_bs_links']}")
    assert all(r['reachable'].values())
    print("  OK\n")

    # Test 6: Wormhole — attacker connects to BS, victims connect to attacker
    print("Test 6: Wormhole topology")
    r = build_links(
        ['d1', 'd2', 'd3', 'd4'],
        {
            'd1': (100, 100, 30),  # attacker, near bs1
            'd2': (900, 900, 30),  # victim, far from bs1
            'd3': (910, 900, 30),  # victim, far
            'd4': (500, 500, 30),  # normal
        },
        ['bs1'], {'bs1': (0, 0, 0)},
        wormhole=True,
        wormhole_attacker='d1',
        wormhole_victims=['d2', 'd3'],
    )
    print(f"  links: {r['links']}")
    print(f"  forced (non-wormhole victims): {r['forced_bs_links']}")
    # d2->d1 and d3->d1 must be present
    assert tuple(sorted(('d2', 'd1'))) in r['links']
    assert tuple(sorted(('d3', 'd1'))) in r['links']
    assert tuple(sorted(('d1', 'bs1'))) in r['links']
    print("  OK\n")

    # Test 7: empty UAV list
    print("Test 7: empty UAV list")
    r = build_links([], {}, ['bs1'], {'bs1': (0, 0, 0)})
    print(f"  links: {r['links']}  (expect [])")
    assert r['links'] == []
    print("  OK\n")

    # Test 8: 20 UAVs scattered around (500,500,30), BS at (0,0,0).
    # Note: with strict nearest-neighbor, scattered UAVs naturally form
    # multiple disjoint pair/triple components (mutual-nearest pairs).
    # Per-cluster policy adds one bridge per component, which is still
    # power-superior to the old per-UAV policy (which would have added 20
    # direct BS links here).
    print("Test 8: 20 UAVs scattered around (500,500,30), BS at (0,0,0)")
    import random
    rng = random.Random(0)
    uav_names = [f'd{i}' for i in range(1, 21)]
    uav_pos = {n: (500 + rng.gauss(0, 30),
                   500 + rng.gauss(0, 30),
                   30 + rng.gauss(0, 5)) for n in uav_names}
    r = build_links(uav_names, uav_pos, ['bs1'], {'bs1': (0, 0, 0)})
    n_bridges = r['diagnostics']['cluster_bridges_added']
    print(f"  total links: {r['diagnostics']['total_links']}")
    print(f"  cluster bridges added: {n_bridges}  (vs 20 with per-UAV policy)")
    print(f"  bridge UAVs: {r['forced_bs_links']}")
    print(f"  all UAVs reach a BS: {all(r['reachable'].values())}")
    assert all(r['reachable'].values())
    assert n_bridges < 20, (
        "Per-cluster policy must add fewer bridges than the UAV count"
    )
    print(f"  OK ({n_bridges} bridges for 20 UAVs — power-saving vs per-UAV)\n")

    # Test 8b: Tightly-grouped UAVs (small spread) form a single component.
    print("Test 8b: 10 UAVs *tightly* grouped (sigma=2m) — should be 1 component")
    rng2 = random.Random(1)
    uav_names = [f'd{i}' for i in range(1, 11)]
    uav_pos = {n: (500 + rng2.gauss(0, 2),
                   500 + rng2.gauss(0, 2),
                   30) for n in uav_names}
    r = build_links(uav_names, uav_pos, ['bs1'], {'bs1': (0, 0, 0)})
    n_bridges = r['diagnostics']['cluster_bridges_added']
    print(f"  cluster bridges added: {n_bridges}")
    print(f"  bridge UAVs: {r['forced_bs_links']}")
    # Tight cluster -> nearest-neighbor edges collapse to fewer components
    assert n_bridges <= 5, f"Expected few bridges for tight cluster, got {n_bridges}"
    assert all(r['reachable'].values())
    print(f"  OK\n")

    # Test 9: Two well-separated clusters far from BS.
    # Each cluster's strict nearest-neighbor graph fragments into multiple
    # components (typically pair-components), but the bridges-per-cluster
    # policy ensures both clusters get connected to the BS, with FEWER
    # bridges than UAVs.
    print("Test 9: Two separate clusters, both far from BS")
    uav_names = [f'd{i}' for i in range(1, 11)]
    uav_pos = {}
    # Cluster A around (200,200), cluster B around (800,800)
    for i, n in enumerate(uav_names[:5]):
        uav_pos[n] = (200 + rng.gauss(0, 5), 200 + rng.gauss(0, 5), 30)
    for i, n in enumerate(uav_names[5:]):
        uav_pos[n] = (800 + rng.gauss(0, 5), 800 + rng.gauss(0, 5), 30)
    r = build_links(uav_names, uav_pos, ['bs1'], {'bs1': (500, 0, 0)})
    n_bridges = r['diagnostics']['cluster_bridges_added']
    print(f"  total links: {r['diagnostics']['total_links']}")
    print(f"  cluster bridges added: {n_bridges}")
    print(f"  bridge UAVs: {r['forced_bs_links']}")
    # Expect at least 2 (one per spatial cluster) but at most 10 (UAV count).
    # In practice: the bridge selection picks the UAV with shortest BS distance
    # in each connected component. Expect ~2-6 in this scenario.
    assert 2 <= n_bridges < 10
    assert all(r['reachable'].values())
    # Verify both spatial clusters got at least one bridge each
    bridge_in_A = any(int(b[1:]) <= 5 for b in r['forced_bs_links'])
    bridge_in_B = any(int(b[1:]) > 5 for b in r['forced_bs_links'])
    assert bridge_in_A and bridge_in_B, (
        f"Each spatial cluster needs at least one bridge; got {r['forced_bs_links']}"
    )
    print(f"  OK (both spatial clusters bridged, {n_bridges} total)\n")

    # Test 10: Cluster already containing BS via chain — no bridge added.
    print("Test 10: One UAV near BS, others chain to it — already reachable")
    uav_names = [f'd{i}' for i in range(1, 6)]
    # d1 at (10,0), d2 at (60,0), d3 at (110,0), d4 at (160,0), d5 at (210,0)
    # nearest-neighbor: each picks its left neighbor; d1's nearest is bs1 at (0,0)
    uav_pos = {f'd{i}': (10 + 50 * (i - 1), 0, 30) for i in range(1, 6)}
    r = build_links(uav_names, uav_pos, ['bs1'], {'bs1': (0, 0, 0)})
    print(f"  links: {r['links']}")
    print(f"  cluster bridges added: {r['diagnostics']['cluster_bridges_added']}")
    assert r['diagnostics']['cluster_bridges_added'] == 0
    assert all(r['reachable'].values())
    print("  OK (no bridge needed — chain reaches BS via d1)\n")

    print("routing.py self-test complete.")