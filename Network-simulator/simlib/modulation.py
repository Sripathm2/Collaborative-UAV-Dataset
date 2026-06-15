"""
simlib/modulation.py

Seven modulation-and-coding schemes modeled on 802.11a/g OFDM, plus an
adaptive selector. Returns BER, PER, nominal rate for a given received SNR.

Schemes (key -> M-ary, code rate, nominal rate, required SNR for 10% PER@1kB):
  bpsk12      BPSK,   R=1/2,   6 Mbps,   req_snr =  4 dB
  qpsk12      QPSK,   R=1/2,  12 Mbps,   req_snr =  7 dB
  qpsk34      QPSK,   R=3/4,  18 Mbps,   req_snr = 10 dB
  qam16_12    16-QAM, R=1/2,  24 Mbps,   req_snr = 13 dB
  qam16_34    16-QAM, R=3/4,  36 Mbps,   req_snr = 17 dB
  qam64_23    64-QAM, R=2/3,  48 Mbps,   req_snr = 21 dB
  qam64_34    64-QAM, R=3/4,  54 Mbps,   req_snr = 25 dB

Assumptions / approximations:
  - BER formulas are Gray-coded closed-form approximations on Es/N0 (received
    SNR, dB) in AWGN. These match the forms used in the original Topo.py.
  - Convolutional coding is modeled as an effective SNR shift equal to the
    scheme's coding gain (approximate values for K=7 conv codes in 802.11):
      R=1/2 -> +5.0 dB;  R=2/3 -> +4.0 dB;  R=3/4 -> +3.0 dB
    This is a simulation-quality approximation, not a hard-decision decoder
    simulation. It reproduces the ordering of schemes and approximately the
    required-SNR thresholds in the 802.11 standard.
  - PER is computed as 1 - (1-BER)^L assuming independent bit errors, where
    L is packet length in bits. Bursty fading is NOT modeled here; that's
    the job of shadow_fading.ShadowFadingField upstream of this module.

SNR contract: the snr_db argument is Es/N0 in dB, i.e.,
    snr_db = tx_power_dBm - path_loss_dB - noise_floor_dBm
Do the path loss and shadow fading elsewhere and hand the net SNR to this
module.
"""

import math


# ---------------- scheme table ----------------

_SCHEMES = {
    'bpsk12':   {'M': 2,  'R_code': 0.5,    'cg_db': 5.0, 'rate_mbps':  6, 'req_snr_db':  4},
    'qpsk12':   {'M': 4,  'R_code': 0.5,    'cg_db': 5.0, 'rate_mbps': 12, 'req_snr_db':  7},
    'qpsk34':   {'M': 4,  'R_code': 0.75,   'cg_db': 3.0, 'rate_mbps': 18, 'req_snr_db': 10},
    'qam16_12': {'M': 16, 'R_code': 0.5,    'cg_db': 5.0, 'rate_mbps': 24, 'req_snr_db': 13},
    'qam16_34': {'M': 16, 'R_code': 0.75,   'cg_db': 3.0, 'rate_mbps': 36, 'req_snr_db': 17},
    'qam64_23': {'M': 64, 'R_code': 2/3,    'cg_db': 4.0, 'rate_mbps': 48, 'req_snr_db': 21},
    'qam64_34': {'M': 64, 'R_code': 0.75,   'cg_db': 3.0, 'rate_mbps': 54, 'req_snr_db': 25},
}

# Ordered from most robust (lowest req SNR) to highest-rate. Used by
# the adaptive selector.
_SCHEMES_BY_ROBUSTNESS = [
    'bpsk12', 'qpsk12', 'qpsk34',
    'qam16_12', 'qam16_34',
    'qam64_23', 'qam64_34',
]


# ---------------- BER core ----------------

def _uncoded_ber(M, snr_db):
    """Uncoded BER for M-ary modulation given Es/N0 in dB (AWGN, Gray)."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    snr_lin = max(snr_lin, 1e-30)
    if M == 2:    # BPSK
        return 0.5 * math.erfc(math.sqrt(snr_lin))
    if M == 4:    # QPSK
        return 0.5 * math.erfc(math.sqrt(snr_lin / 2.0))
    if M == 16:   # 16-QAM (Gray, approximation)
        return (3.0 / 8.0) * math.erfc(math.sqrt(snr_lin * 4.0 / 5.0))
    if M == 64:   # 64-QAM (Gray, approximation)
        return (7.0 / 24.0) * math.erfc(math.sqrt(snr_lin / 7.0))
    raise ValueError(f"Unsupported modulation order M={M}")


def _coded_ber(scheme_key, snr_db):
    s = _SCHEMES[scheme_key]
    eff_snr = snr_db + s['cg_db']
    ber = _uncoded_ber(s['M'], eff_snr)
    # numerical clamp: never exactly 0 (log traps) nor >0.5 (silly)
    return min(max(ber, 1e-12), 0.5)


# ---------------- public API ----------------

def list_schemes():
    """All scheme keys, ordered most robust -> highest rate."""
    return list(_SCHEMES_BY_ROBUSTNESS)


def get_rate_mbps(scheme):
    if scheme not in _SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}")
    return _SCHEMES[scheme]['rate_mbps']


def get_required_snr_db(scheme):
    if scheme not in _SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}")
    return _SCHEMES[scheme]['req_snr_db']


def get_ber(scheme, snr_db):
    if scheme not in _SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}")
    return _coded_ber(scheme, snr_db)


def get_per(scheme, snr_db, packet_bits=8000):
    """Packet error rate assuming independent bit errors."""
    ber = get_ber(scheme, snr_db)
    # Use (1 - (1-BER)^L) = 1 - exp(L * log(1-BER)) for numerical stability
    # but for ber <= 0.5, direct form is fine:
    per = 1.0 - (1.0 - ber) ** packet_bits
    return min(max(per, 0.0), 1.0)


def select_adaptive(snr_db, margin_db=3.0):
    """
    Return the highest-rate scheme whose required SNR + margin <= snr_db.
    If no scheme qualifies, returns the most robust scheme (bpsk12) so the
    link is still attempted (packet loss will be high).
    """
    chosen = _SCHEMES_BY_ROBUSTNESS[0]
    for key in _SCHEMES_BY_ROBUSTNESS:
        if snr_db >= _SCHEMES[key]['req_snr_db'] + margin_db:
            chosen = key
    return chosen


def resolve_scheme(scheme_or_adaptive, snr_db, margin_db=3.0):
    """If 'adaptive', resolve to best fit; otherwise return unchanged."""
    if scheme_or_adaptive == 'adaptive':
        return select_adaptive(snr_db, margin_db)
    if scheme_or_adaptive not in _SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme_or_adaptive!r}")
    return scheme_or_adaptive


def compute_link(scheme_or_adaptive, snr_db, packet_bits=8000,
                 adaptive_margin_db=3.0):
    """
    End-to-end link computation: return dict with scheme used, SNR, BER, PER,
    nominal rate. This is the function Topo.py will call per link per window.
    """
    scheme = resolve_scheme(scheme_or_adaptive, snr_db, adaptive_margin_db)
    ber = get_ber(scheme, snr_db)
    per = 1.0 - (1.0 - ber) ** packet_bits
    per = min(max(per, 0.0), 1.0)
    return {
        'scheme': scheme,
        'snr_db': snr_db,
        'ber': ber,
        'per': per,
        'rate_mbps': _SCHEMES[scheme]['rate_mbps'],
        'adaptive': scheme_or_adaptive == 'adaptive',
    }


# ---------------- self-test ----------------

if __name__ == '__main__':
    print("=== modulation.py self-test ===\n")

    print("Scheme table (sanity):")
    print(f"  {'scheme':<10} {'M':>3} {'R':>5} {'CG(dB)':>7} {'rate':>6} {'req_SNR':>8}")
    for k in _SCHEMES_BY_ROBUSTNESS:
        s = _SCHEMES[k]
        print(f"  {k:<10} {s['M']:>3} {s['R_code']:>5.2f} {s['cg_db']:>7.1f} "
              f"{s['rate_mbps']:>4} Mb {s['req_snr_db']:>6} dB")
    print()

    # Test 1: BER monotonic in SNR for each scheme
    print("Test 1: BER is monotonically decreasing with SNR")
    for k in _SCHEMES_BY_ROBUSTNESS:
        prev = 1.0
        monotone = True
        for snr in range(-5, 40):
            b = get_ber(k, snr)
            if b > prev + 1e-15:
                monotone = False
                break
            prev = b
        print(f"  {k:<10}: {'OK' if monotone else 'FAIL'}")
    print()

    # Test 2: BER ordering at 15 dB — more robust schemes have lower BER
    print("Test 2: At fixed SNR, more robust schemes have <= BER than higher-rate")
    snr_ref = 15.0
    bers = [(k, get_ber(k, snr_ref)) for k in _SCHEMES_BY_ROBUSTNESS]
    print(f"  At SNR = {snr_ref} dB:")
    for k, b in bers:
        print(f"    {k:<10}: BER = {b:.3e}")
    ordered = all(bers[i][1] <= bers[i + 1][1] + 1e-15 for i in range(len(bers) - 1))
    print(f"  Monotone BPSK12 <= ... <= QAM64_34: {'OK' if ordered else 'FAIL'}")
    print()

    # Test 3: Adaptive selector picks correct scheme across SNR sweep
    print("Test 3: Adaptive selection vs SNR (margin=3 dB)")
    print(f"  {'SNR(dB)':>8} | {'chosen':<10} | {'rate':>6}")
    for snr in (0, 5, 10, 15, 20, 25, 30, 35):
        k = select_adaptive(snr, margin_db=3.0)
        print(f"  {snr:>8} | {k:<10} | {_SCHEMES[k]['rate_mbps']:>4} Mb")
    print()

    # Test 4: PER: full link computation
    print("Test 4: compute_link() end-to-end (adaptive, 1000-byte packets)")
    pkt_bits = 8000
    print(f"  {'SNR':>5} | {'scheme':<10} | {'BER':>10} | {'PER':>8} | {'rate':>6}")
    for snr in (0, 5, 10, 15, 20, 25, 30):
        r = compute_link('adaptive', snr, packet_bits=pkt_bits)
        print(f"  {snr:>3} dB | {r['scheme']:<10} | {r['ber']:>10.2e} | "
              f"{r['per']:>7.1%} | {r['rate_mbps']:>4} Mb")
    print()

    # Test 5: PER boundary behaviour
    print("Test 5: PER saturates at low/high SNR")
    per_low  = get_per('qam64_34', -20, 8000)
    per_high = get_per('bpsk12', 40, 8000)
    print(f"  qam64_34 @ SNR=-20: PER = {per_low:.4f} (expect ~1)")
    print(f"  bpsk12   @ SNR= 40: PER = {per_high:.2e} (expect ~0)")
    print()

    # Test 6: Unknown scheme raises
    print("Test 6: Unknown scheme raises ValueError")
    try:
        get_ber('qpsk7', 10)
        print("  FAIL (no exception)")
    except ValueError as e:
        print(f"  OK: {e}")
    print()

    # Test 7: explicit scheme pass-through
    print("Test 7: resolve_scheme with explicit name passes through")
    s = resolve_scheme('qam16_34', 5.0)
    print(f"  resolve_scheme('qam16_34', 5.0) -> {s} ({'OK' if s == 'qam16_34' else 'FAIL'})")
    print()

    print("modulation.py self-test complete.")
