"""
simlib/config_parser.py

Parses a single line from configurations.txt into a structured scenario dict.

Supports two line formats:

OLD (4 fields, backward compatible):
    <attacks>-<drones>-<basestations>-<payload>
    e.g.  benign-5-1-image
          dos=1000+ddos=100-15-2-video
          blackhole-20-1-image
    Missing fields auto-filled with defaults:
        pathloss   = 'logdist'
        modulation = 'adaptive'
        missions   = 'mixed'
        tx_power   = 20 dBm
        noise      = 95  (interpreted as -95 dBm)

NEW (9 fields, explicit):
    <attacks>-<drones>-<basestations>-<payload>-<pathloss>-<modulation>-<missions>-<txpower>-<noise>
    e.g. benign-5-1-image-logdist-adaptive-mixed-20-95
         dos=1000-10-2-video-3gpp-qam16_34-spiral-20-95
         replay=50,200,10,5,inc-10-1-image-logdist-adaptive-grid,random-25-100

Field semantics:
  <attacks>     :  'benign', or '+' separated list e.g. 'dos=1000+wormhole'
                   Each attack is either a bare name (benign, blackhole,
                   wormhole) or <name>=<params>.
                   Params for dos/ddos:   inter-packet interval in microseconds
                                          (1 means hping3 --flood)
                   Params for replay:     buf,delay_ms,rate_pps,ttl_dec,seq_mode
                                          seq_mode in {inc, rand, zero}
  <drones>      :  positive int
  <basestations>:  positive int
  <payload>     :  'image' or 'video'  (substring match in FTP_client.py)
  <pathloss>    :  'logdist' | '3gpp'
  <modulation>  :  bpsk12 | qpsk12 | qpsk34 | qam16_12 | qam16_34 |
                   qam64_23 | qam64_34 | adaptive
  <missions>    :  one of:
                     - single keyword: spiral | grid | hover_transit | random
                     - 'mixed'  (fresh-random per invocation)
                     - comma list: 'spiral,grid,random'  (cycles if shorter
                       than UAV count)
  <txpower>     :  positive int, dBm
  <noise>       :  positive int (HYPHEN-FREE by convention), interpreted as
                   NEGATIVE dBm. Example: 95 -> noise_floor = -95 dBm.

Parser returns a dict with everything resolved and validated.
"""

import random


# -------- constants / defaults --------

DEFAULT_PATHLOSS = 'logdist'
DEFAULT_MODULATION = 'adaptive'
DEFAULT_MISSIONS = 'mixed'
DEFAULT_TX_POWER_DBM = 20
DEFAULT_NOISE_POS = 95   # interpreted as -95 dBm

VALID_PATHLOSS = {'logdist', '3gpp'}
VALID_MODULATION = {
    'bpsk12', 'qpsk12', 'qpsk34',
    'qam16_12', 'qam16_34',
    'qam64_23', 'qam64_34',
    'adaptive',
}
VALID_MISSION_KEYWORDS = {'spiral', 'grid', 'hover_transit', 'random'}
VALID_SINGLE_MISSION_TOKENS = VALID_MISSION_KEYWORDS | {'mixed'}
VALID_PAYLOADS = {'image', 'video'}
VALID_SEQ_MODES = {'inc', 'rand', 'zero'}

# Known attack names (bare or with '=')
KNOWN_ATTACK_NAMES = {'benign', 'dos', 'ddos', 'blackhole', 'wormhole', 'replay'}


# -------- helpers --------

def _parse_attacks(attacks_field):
    """
    Returns list of dicts; 'benign' returns [{'type': 'benign'}].
    Combined attacks joined by '+' become multiple list entries.
    """
    out = []
    for tok in attacks_field.split('+'):
        tok = tok.strip()
        if not tok:
            raise ValueError(f"Empty attack token in {attacks_field!r}")
        if '=' in tok:
            name, rhs = tok.split('=', 1)
        else:
            name, rhs = tok, None
        if name not in KNOWN_ATTACK_NAMES:
            raise ValueError(
                f"Unknown attack name {name!r}. Known: {sorted(KNOWN_ATTACK_NAMES)}"
            )

        d = {'type': name}
        if name in ('dos', 'ddos'):
            if rhs is None:
                raise ValueError(f"{name} requires an '=<rate>' value")
            try:
                d['interval_us'] = int(rhs)
            except ValueError:
                raise ValueError(f"{name} rate must be int, got {rhs!r}")
            if d['interval_us'] < 1:
                raise ValueError(f"{name} rate must be >= 1 (got {d['interval_us']})")
        elif name == 'replay':
            if rhs is None:
                raise ValueError("replay requires '=buf,delay,rate,ttl,seq'")
            parts = rhs.split(',')
            if len(parts) != 5:
                raise ValueError(
                    f"replay expects 5 comma-separated params "
                    f"(buf,delay_ms,rate_pps,ttl_dec,seq_mode), got {rhs!r}"
                )
            try:
                d['buffer_packets'] = int(parts[0])
                d['delay_ms'] = int(parts[1])
                d['rate_pps'] = int(parts[2])
                d['ttl_decrement'] = int(parts[3])
            except ValueError:
                raise ValueError(f"replay numeric fields invalid: {rhs!r}")
            seq_mode = parts[4].strip()
            if seq_mode not in VALID_SEQ_MODES:
                raise ValueError(
                    f"replay seq_mode must be in {VALID_SEQ_MODES}, got {seq_mode!r}"
                )
            d['seq_mode'] = seq_mode
        elif name in ('benign', 'blackhole', 'wormhole'):
            if rhs is not None:
                raise ValueError(f"{name} takes no parameters (got {rhs!r})")
        out.append(d)
    return out


def _parse_missions(missions_field, num_drones, rng):
    """
    Returns a list of mission strings, one per drone.

    - single keyword: applied to all
    - 'mixed': random selection seeded by rng for reproducibility within this
      invocation (rng is fresh-per-invocation per Shree's spec)
    - comma list: cycles to fill num_drones
    """
    f = missions_field.strip()
    if f in VALID_MISSION_KEYWORDS:
        return [f] * num_drones
    if f == 'mixed':
        pool = sorted(VALID_MISSION_KEYWORDS)
        return [rng.choice(pool) for _ in range(num_drones)]
    if ',' in f:
        items = [x.strip() for x in f.split(',') if x.strip()]
        for m in items:
            if m not in VALID_MISSION_KEYWORDS:
                raise ValueError(
                    f"Unknown mission {m!r}. Valid: {sorted(VALID_MISSION_KEYWORDS)}"
                )
        if not items:
            raise ValueError("Empty mission list")
        # cycle
        return [items[i % len(items)] for i in range(num_drones)]
    raise ValueError(
        f"Bad missions field {missions_field!r}. Expected a keyword "
        f"({sorted(VALID_SINGLE_MISSION_TOKENS)}) or comma-separated list."
    )


# -------- main entry point --------

def parse_line(line, rng=None):
    """
    Parse a single line of configurations.txt. Returns a dict:
        {
          'raw': original line,
          'attacks': [ {'type': ..., ...}, ... ],
          'num_drones': int,
          'num_basestations': int,
          'payload': 'image'|'video',
          'pathloss_model': 'logdist'|'3gpp',
          'modulation_scheme': '<mod>',
          'missions': [per-drone-mission strings, len == num_drones],
          'tx_power_dBm': int,
          'noise_floor_dBm': int (negative),
          'format': 'old'|'new',
        }

    rng: optional random.Random used for 'mixed' mission assignment.
         If None, a fresh Random() is created (fresh-per-invocation).
    """
    if rng is None:
        rng = random.Random()

    line = line.strip()
    if not line or line.startswith('#'):
        raise ValueError("Empty or comment line")

    fields = line.split('-')
    # attacks can contain '=' which may in turn contain ',' but not '-'.
    # So splitting on '-' is safe IF no '-' appears inside replay params.
    # replay params are integers + seq_mode token; no '-'.
    # mission list is comma-separated, no '-'.
    # Therefore a naive split on '-' is OK.

    if len(fields) == 4:
        fmt = 'old'
        attacks_s, drones_s, bs_s, payload_s = fields
        pathloss_s = DEFAULT_PATHLOSS
        modulation_s = DEFAULT_MODULATION
        missions_s = DEFAULT_MISSIONS
        tx_s = str(DEFAULT_TX_POWER_DBM)
        noise_s = str(DEFAULT_NOISE_POS)
    elif len(fields) == 9:
        fmt = 'new'
        (attacks_s, drones_s, bs_s, payload_s,
         pathloss_s, modulation_s, missions_s, tx_s, noise_s) = fields
    else:
        raise ValueError(
            f"Line has {len(fields)} dash-separated fields; expected 4 "
            f"(old) or 9 (new).  Line: {line!r}"
        )

    # attacks
    attacks = _parse_attacks(attacks_s)

    # ints
    try:
        num_drones = int(drones_s)
        num_bs = int(bs_s)
        tx_dbm = int(tx_s)
        noise_pos = int(noise_s)
    except ValueError as e:
        raise ValueError(f"Non-int field in {line!r}: {e}")
    if num_drones < 1:
        raise ValueError(f"num_drones must be >= 1, got {num_drones}")
    if num_bs < 1:
        raise ValueError(f"num_basestations must be >= 1, got {num_bs}")
    if noise_pos < 1:
        raise ValueError(f"noise must be positive (interpreted as -dBm), got {noise_pos}")

    # payload
    if payload_s not in VALID_PAYLOADS:
        raise ValueError(f"payload must be in {VALID_PAYLOADS}, got {payload_s!r}")

    # pathloss / modulation
    if pathloss_s not in VALID_PATHLOSS:
        raise ValueError(f"pathloss must be in {VALID_PATHLOSS}, got {pathloss_s!r}")
    if modulation_s not in VALID_MODULATION:
        raise ValueError(f"modulation must be in {VALID_MODULATION}, got {modulation_s!r}")

    # missions
    missions = _parse_missions(missions_s, num_drones, rng)

    return {
        'raw': line,
        'format': fmt,
        'attacks': attacks,
        'num_drones': num_drones,
        'num_basestations': num_bs,
        'payload': payload_s,
        'pathloss_model': pathloss_s,
        'modulation_scheme': modulation_s,
        'missions': missions,
        'tx_power_dBm': tx_dbm,
        'noise_floor_dBm': -noise_pos,
    }


def parse_file_line(path, line_index, rng=None):
    """
    Read line at 1-indexed position from configurations.txt and parse it.
    (Matches the bash script convention: python3 Topo.py <N>.)
    Blank lines are skipped in the index count.
    """
    with open(path, 'r') as f:
        all_lines = f.readlines()
    # Match the behaviour you currently have in Topo.py:
    #   config_line = f.readlines()[config_number].strip()
    # i.e. zero-indexed into the raw list. We replicate that; the bash script
    # passes integers; caller is responsible for the index convention.
    if line_index < 0 or line_index >= len(all_lines):
        raise IndexError(f"line_index {line_index} out of range (0..{len(all_lines)-1})")
    return parse_line(all_lines[line_index], rng=rng)


# -------- self-test --------

if __name__ == '__main__':
    import pprint
    tests = [
        # OLD FORMAT - should auto-fill defaults
        ('benign-5-1-image', True),
        ('dos=1000-10-2-video', True),
        ('dos=1000+ddos=100-15-1-image', True),
        ('blackhole+wormhole-20-2-video', True),
        ('ddos=1000+ddos=100-5-2-image', True),

        # NEW FORMAT
        ('benign-5-1-image-logdist-adaptive-mixed-20-95', True),
        ('dos=1000-10-2-video-3gpp-qam16_34-spiral-20-90', True),
        ('replay=50,200,10,5,inc-10-1-image-logdist-adaptive-grid-25-95', True),
        ('blackhole-20-2-video-logdist-qpsk12-spiral,grid,random-20-95', True),
        ('benign-3-1-image-logdist-adaptive-hover_transit,spiral,grid-20-95', True),

        # BAD INPUTS - should raise
        ('', False),
        ('#comment', False),
        ('too-few-fields', False),
        ('benign-0-1-image', False),                            # 0 drones
        ('benign-5-1-audio', False),                            # bad payload
        ('benign-5-1-image-foo-adaptive-mixed-20-95', False),   # bad pathloss
        ('benign-5-1-image-logdist-xxx-mixed-20-95', False),    # bad modulation
        ('benign-5-1-image-logdist-adaptive-alien-20-95', False),  # bad mission
        ('dos-5-1-image', False),                               # dos w/o rate
        ('replay=50,200,10,5-5-1-image', False),                # replay missing seq_mode
        ('replay=50,200,10,5,bogus-5-1-image', False),          # bad seq_mode
        ('benign-5-1-image-logdist-adaptive-mixed-20-0', False),  # noise=0
        ('unknown_attack-5-1-image', False),                    # unknown attack
    ]
    print("=== config_parser.py self-test ===\n")
    passed = failed = 0
    for i, (line, should_parse) in enumerate(tests, 1):
        try:
            rng = random.Random(42)
            out = parse_line(line, rng=rng)
            if should_parse:
                print(f"[{i:>2}] PARSE OK  : {line!r}")
                # one-line summary
                atk_str = ','.join(a['type'] + (f"=..." if len(a) > 1 else "")
                                   for a in out['attacks'])
                print(f"       -> fmt={out['format']}  attacks=[{atk_str}]  "
                      f"drones={out['num_drones']}  bs={out['num_basestations']}  "
                      f"payload={out['payload']}  pl={out['pathloss_model']}  "
                      f"mod={out['modulation_scheme']}")
                print(f"       -> missions={out['missions']}  "
                      f"tx={out['tx_power_dBm']}dBm  noise={out['noise_floor_dBm']}dBm")
                passed += 1
            else:
                print(f"[{i:>2}] FAIL (expected reject): {line!r} -> {out}")
                failed += 1
        except (ValueError, IndexError) as e:
            if not should_parse:
                print(f"[{i:>2}] REJECT OK : {line!r}  ({e})")
                passed += 1
            else:
                print(f"[{i:>2}] FAIL (expected parse): {line!r}  ({e})")
                failed += 1
        print()

    # Deep dump of one specific case
    print("=== detail of replay parse ===")
    out = parse_line('replay=50,200,10,5,inc-10-1-image-logdist-adaptive-grid-25-95')
    pprint.pprint(out)
    print()

    print(f"\nSummary: {passed} passed, {failed} failed")
