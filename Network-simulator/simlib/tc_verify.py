"""
simlib/tc_verify.py

Verify that `tc qdisc show dev <iface>` output matches the params we asked
Containernet's link.intf.config() to apply. When mismatches happen, log
them with a timestamp to a diagnostics file. This is the safety-net layer
that catches silent tc failures without blocking the main simulation loop.

Two-part design:

  1. parse_tc_qdisc(output_str) -> dict
     Pure function; unit-testable. Extracts delay/rate/loss from typical
     `tc qdisc show` output. Returns None for fields not found.

  2. TcVerifier class
     Wraps subprocess calls, compares to intended, writes to a diagnostics
     file. Tolerances configurable. One instance per simulation run.

Typical usage inside Topo.py:

    verifier = TcVerifier(log_path='pcaps/tc_diagnostics_<runid>.log')
    # ... after link.intf.config(delay='10.5ms', bw=5, loss=20):
    verifier.verify(link.intf.name, intended_delay_ms=10.5,
                    intended_rate_mbit=5.0, intended_loss_pct=20.0)
    # ... at end of run:
    verifier.summarize()
"""

import os
import re
import subprocess
import time


# ---------------- parser ----------------

_RE_DELAY = re.compile(r'\bdelay\s+([\d.]+)\s*(ms|us|s)\b', re.IGNORECASE)
_RE_RATE  = re.compile(r'\brate\s+([\d.]+)\s*([KMG]?bit)\b', re.IGNORECASE)
_RE_LOSS  = re.compile(r'\bloss\s+([\d.]+)%', re.IGNORECASE)


def _to_ms(value, unit):
    unit = unit.lower()
    if unit == 'ms':
        return float(value)
    if unit == 'us':
        return float(value) / 1000.0
    if unit == 's':
        return float(value) * 1000.0
    return None


def _to_mbit(value, unit):
    unit = unit.lower()
    v = float(value)
    if unit == 'bit':
        return v / 1e6
    if unit == 'kbit':
        return v / 1e3
    if unit == 'mbit':
        return v
    if unit == 'gbit':
        return v * 1e3
    return None


def parse_tc_qdisc(output_str, class_output_str=''):
    """
    Parse output of `tc qdisc show dev <iface>` and optionally
    `tc class show dev <iface>`. Returns:
        {'delay_ms': float or None,
         'rate_mbit': float or None,    # taken from htb class output if given
         'loss_pct': float or None,
         'has_netem': bool,
         'raw': output_str + class_output_str}

    With Mininet TCLink the qdisc tree is `htb -> netem`. `tc qdisc show`
    prints htb + netem lines, but rate lives in the htb class (`tc class
    show`) and not in the qdisc itself. So:
      - delay/loss come from the netem qdisc line
      - rate (if present) comes from the htb class line

    If class_output_str is empty, rate_mbit will be None.
    """
    has_netem = ('netem' in output_str.lower())

    # netem omits fields whose value is zero from its qdisc show output
    # (e.g. no "loss X%" token at all when loss=0%, no "delay ..." when
    # delay=0ms). If the netem qdisc IS present but the token is missing,
    # treat the value as 0 rather than None, so a configured-zero field
    # doesn't spuriously mismatch against an intended 0.
    delay_ms = None
    m = _RE_DELAY.search(output_str)
    if m:
        delay_ms = _to_ms(m.group(1), m.group(2))
    elif has_netem:
        delay_ms = 0.0

    loss_pct = None
    m = _RE_LOSS.search(output_str)
    if m:
        loss_pct = float(m.group(1))
    elif has_netem:
        loss_pct = 0.0

    # Rate comes from the htb class output (e.g.,
    # "class htb 5:1 root prio 0 rate 5Mbit ceil 5Mbit ...")
    rate_mbit = None
    if class_output_str:
        m = _RE_RATE.search(class_output_str)
        if m:
            rate_mbit = _to_mbit(m.group(1), m.group(2))

    return {
        'delay_ms': delay_ms,
        'rate_mbit': rate_mbit,
        'loss_pct': loss_pct,
        'has_netem': has_netem,
        'raw': output_str.strip() + (
            ('  ||  CLASS: ' + class_output_str.strip()) if class_output_str else ''
        ),
    }


# ---------------- verifier ----------------

class TcVerifier:
    """
    Compares actual vs intended tc settings and logs mismatches.
    Safe defaults — won't crash your simulation if tc is absent or fails.
    """

    def __init__(self, log_path,
                 delay_tol_ms=0.5,
                 rate_tol_mbit=0.05,
                 loss_tol_pct=0.5,
                 enabled=True):
        self.log_path = log_path
        self.delay_tol_ms = delay_tol_ms
        self.rate_tol_mbit = rate_tol_mbit
        self.loss_tol_pct = loss_tol_pct
        self.enabled = enabled

        self.n_checks = 0
        self.n_mismatches = 0
        self.n_missing_netem = 0
        self.n_tc_errors = 0

        # ensure log directory exists
        if enabled:
            d = os.path.dirname(os.path.abspath(log_path))
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            with open(self.log_path, 'a') as f:
                f.write(f"# tc_verify run started at {time.time()} "
                        f"({time.strftime('%Y-%m-%d %H:%M:%S')})\n")

    def _log(self, msg):
        if not self.enabled:
            return
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')

    def _run_tc(self, iface, container=None):
        """
        Return (stdout_str, error_str_or_None).

        If `container` is given, run `docker exec <container> tc qdisc show
        dev <iface>` so we read the qdisc from inside the container's network
        namespace (where Mininet places container-side interfaces like
        d1-eth0). Otherwise run on the host (for switch-side interfaces like
        s2-eth2 which live in the host namespace).
        """
        if container:
            cmd = ['docker', 'exec', container, 'tc', 'qdisc', 'show', 'dev', iface]
        else:
            cmd = ['tc', 'qdisc', 'show', 'dev', iface]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return (result.stdout or '', result.stderr.strip() or 'nonzero exit')
            return (result.stdout or '', None)
        except FileNotFoundError:
            return ('', 'tc not installed')
        except subprocess.TimeoutExpired:
            return ('', 'tc timeout')
        except Exception as e:
            return ('', f'{type(e).__name__}: {e}')

    def _run_tc_class(self, iface, container=None):
        """
        Read `tc class show dev <iface>` (HTB class info, where rate lives).
        Returns (stdout_str, error_str_or_None). Same namespace logic as _run_tc.
        """
        if container:
            cmd = ['docker', 'exec', container, 'tc', 'class', 'show', 'dev', iface]
        else:
            cmd = ['tc', 'class', 'show', 'dev', iface]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return ('', result.stderr.strip() or 'nonzero exit')
            return (result.stdout or '', None)
        except Exception as e:
            return ('', f'{type(e).__name__}: {e}')

    def verify(self, iface, intended_delay_ms=None,
               intended_rate_mbit=None, intended_loss_pct=None,
               container=None):
        """
        Read back tc settings on iface and compare.
        Any None in `intended_*` means "don't check that field".
        Pass `container='mn.d1'` for interfaces inside a container's namespace.
        Returns True if all non-None fields match within tolerance.
        """
        self.n_checks += 1
        out, err = self._run_tc(iface, container=container)
        if err is not None:
            self.n_tc_errors += 1
            self._log(f"{time.time():.3f} ERROR iface={iface}  "
                      f"container={container or '-'}  tc: {err}")
            return False

        # Rate lives in htb class output, not in qdisc. Only fetch if we
        # actually need to verify rate; otherwise skip the extra subprocess.
        class_out = ''
        if intended_rate_mbit is not None:
            class_out, class_err = self._run_tc_class(iface, container=container)
            # class_err is non-fatal: parser will simply leave rate_mbit=None,
            # and the rate-mismatch branch below will flag it.

        parsed = parse_tc_qdisc(out, class_out)
        if not parsed['has_netem']:
            self.n_missing_netem += 1
            self._log(f"{time.time():.3f} NO_NETEM iface={iface}  "
                      f"container={container or '-'}  "
                      f"qdisc not active.  raw: {parsed['raw']!r}")
            return False

        mismatches = []
        if intended_delay_ms is not None:
            a = parsed['delay_ms']
            if a is None or abs(a - intended_delay_ms) > self.delay_tol_ms:
                mismatches.append(
                    f"delay intended={intended_delay_ms:.3f}ms actual={a}"
                )
        if intended_rate_mbit is not None:
            a = parsed['rate_mbit']
            if a is None or abs(a - intended_rate_mbit) > self.rate_tol_mbit:
                mismatches.append(
                    f"rate intended={intended_rate_mbit:.3f}Mbit actual={a}"
                )
        if intended_loss_pct is not None:
            a = parsed['loss_pct']
            if a is None or abs(a - intended_loss_pct) > self.loss_tol_pct:
                mismatches.append(
                    f"loss intended={intended_loss_pct:.3f}% actual={a}"
                )

        if mismatches:
            self.n_mismatches += 1
            self._log(
                f"{time.time():.3f} MISMATCH iface={iface}  "
                + " | ".join(mismatches)
                + f"  (raw: {parsed['raw']!r})"
            )
            return False
        return True

    def summarize(self):
        msg = (f"# tc_verify summary: checks={self.n_checks}  "
               f"mismatches={self.n_mismatches}  "
               f"missing_netem={self.n_missing_netem}  "
               f"tc_errors={self.n_tc_errors}")
        self._log(msg)
        return {
            'checks': self.n_checks,
            'mismatches': self.n_mismatches,
            'missing_netem': self.n_missing_netem,
            'tc_errors': self.n_tc_errors,
        }


# ==================== self-test ====================

if __name__ == '__main__':
    import tempfile

    print("=== tc_verify.py self-test ===\n")

    # ---- Parser tests with canned `tc qdisc show` + `tc class show` ----
    print("Parser tests:")

    # Realistic Mininet TCLink output: htb at root, netem as child.
    HTB_NETEM_QDISC = (
        "qdisc htb 5: root refcnt 41 r2q 10 default 0x1\n"
        "qdisc netem 10: parent 5:1 limit 1000 delay 10.0ms loss 5%"
    )
    HTB_CLASS_5MBIT = "class htb 5:1 root prio 0 rate 5Mbit ceil 5Mbit burst 1500b"

    samples = [
        # (label, qdisc_out, class_out, expected_subset)
        (
            "htb+netem with class rate",
            HTB_NETEM_QDISC, HTB_CLASS_5MBIT,
            {'has_netem': True, 'delay_ms': 10.0, 'loss_pct': 5.0, 'rate_mbit': 5.0},
        ),
        (
            "htb+netem qdisc only (no class output)",
            HTB_NETEM_QDISC, "",
            {'has_netem': True, 'delay_ms': 10.0, 'loss_pct': 5.0, 'rate_mbit': None},
        ),
        (
            "single-line netem (legacy form)",
            "qdisc netem 8001: root limit 1000 delay 10.0ms loss 5%", "",
            {'has_netem': True, 'delay_ms': 10.0, 'loss_pct': 5.0, 'rate_mbit': None},
        ),
        (
            "netem microseconds",
            "qdisc netem 8002: root delay 250us", "",
            {'has_netem': True, 'delay_ms': 0.25, 'loss_pct': None, 'rate_mbit': None},
        ),
        (
            "class with Kbit rate",
            "qdisc netem 1: root delay 1ms",
            "class htb 1:1 root prio 0 rate 500Kbit ceil 500Kbit",
            {'has_netem': True, 'delay_ms': 1.0, 'rate_mbit': 0.5},
        ),
        (
            "class with Gbit rate",
            "qdisc netem 1: root delay 1.234ms",
            "class htb 1:1 root rate 1Gbit ceil 1Gbit",
            {'has_netem': True, 'rate_mbit': 1000.0, 'delay_ms': 1.234},
        ),
        (
            "no netem present",
            "qdisc pfifo_fast 0: root refcnt 2", "",
            {'has_netem': False, 'delay_ms': None, 'rate_mbit': None, 'loss_pct': None},
        ),
        (
            "empty input",
            "", "",
            {'has_netem': False, 'delay_ms': None, 'rate_mbit': None, 'loss_pct': None},
        ),
    ]
    for label, qdisc_out, class_out, expected in samples:
        got = parse_tc_qdisc(qdisc_out, class_out)
        ok = all(got[k] == v for k, v in expected.items())
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {label}")
        if not ok:
            print(f"    expected subset: {expected}")
            print(f"    got            : {got}")
    print()

    # ---- Verifier tests with monkey-patched _run_tc / _run_tc_class ----
    print("Verifier tests (with monkey-patched _run_tc / _run_tc_class):")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as tf:
        log_path = tf.name

    v = TcVerifier(log_path=log_path, delay_tol_ms=0.5, rate_tol_mbit=0.05,
                   loss_tol_pct=0.5)

    # Helper for setting up htb+netem responses
    def make_htb(qdisc_str, class_str=''):
        qdisc_resp = (qdisc_str, None)
        class_resp = (class_str, None) if class_str else ('', None)
        return (lambda i, container=None: qdisc_resp,
                lambda i, container=None: class_resp)

    # Case A: exact match (htb+netem with class rate)
    v._run_tc, v._run_tc_class = make_htb(
        "qdisc htb 5: root refcnt 41\nqdisc netem 10: parent 5:1 delay 10.0ms loss 5%",
        "class htb 5:1 root rate 10Mbit ceil 10Mbit",
    )
    assert v.verify('fakeA', intended_delay_ms=10.0,
                    intended_rate_mbit=10.0, intended_loss_pct=5.0)
    print("  [OK] exact match accepted")

    # Case B: within tolerance
    v._run_tc, v._run_tc_class = make_htb(
        "qdisc htb 5: root\nqdisc netem 10: parent 5:1 delay 10.3ms loss 5.2%",
        "class htb 5:1 root rate 10.03Mbit ceil 10.03Mbit",
    )
    assert v.verify('fakeB', intended_delay_ms=10.0,
                    intended_rate_mbit=10.0, intended_loss_pct=5.0)
    print("  [OK] within-tolerance accepted")

    # Case C: loss out of tolerance
    v._run_tc, v._run_tc_class = make_htb(
        "qdisc htb 5: root\nqdisc netem 10: parent 5:1 delay 10.0ms loss 10.0%",
        "class htb 5:1 root rate 10Mbit",
    )
    assert not v.verify('fakeC', intended_delay_ms=10.0,
                        intended_rate_mbit=10.0, intended_loss_pct=5.0)
    print("  [OK] out-of-tolerance rejected")

    # Case D: no netem at all
    v._run_tc = lambda i, container=None: ("qdisc pfifo_fast 0: root refcnt 2", None)
    v._run_tc_class = lambda i, container=None: ('', None)
    assert not v.verify('fakeD', intended_delay_ms=10.0)
    print("  [OK] missing netem detected")

    # Case E: tc command errored
    v._run_tc = lambda i, container=None: ('', 'tc not installed')
    v._run_tc_class = lambda i, container=None: ('', None)
    assert not v.verify('fakeE', intended_delay_ms=10.0)
    print("  [OK] tc error reported")

    # Case F: partial check (only loss specified — no class lookup needed)
    v._run_tc = lambda i, container=None: (
        "qdisc netem 1: root loss 3.0%", None
    )
    v._run_tc_class = lambda i, container=None: ('', None)
    assert v.verify('fakeF', intended_loss_pct=3.0)
    print("  [OK] partial-field check works")

    # Case G: container-namespace path is exercised (ensures arg passes through)
    captured = {}
    def fake_qdisc(i, container=None):
        captured['qdisc_iface'] = i
        captured['qdisc_container'] = container
        return ("qdisc htb 5: root\nqdisc netem 10: delay 1.0ms loss 0%", None)
    def fake_class(i, container=None):
        captured['class_iface'] = i
        captured['class_container'] = container
        return ("class htb 5:1 root rate 5Mbit", None)
    v._run_tc = fake_qdisc
    v._run_tc_class = fake_class
    v.verify('d1-eth0', intended_delay_ms=1.0, intended_rate_mbit=5.0,
             intended_loss_pct=0.0, container='mn.d1')
    assert captured == {
        'qdisc_iface': 'd1-eth0', 'qdisc_container': 'mn.d1',
        'class_iface': 'd1-eth0', 'class_container': 'mn.d1',
    }, captured
    print("  [OK] container kwarg propagated through to both tc calls")

    summary = v.summarize()
    print(f"\n  final summary: {summary}")
    print(f"  log file: {log_path}")

    # Show what was logged
    print("\n  --- log contents ---")
    with open(log_path) as f:
        for ln in f:
            print(f"    {ln.rstrip()}")

    print("\ntc_verify.py self-test complete.")