#!/usr/bin/env python3
"""
diversity_table.py

Compute Hellinger and "JSD" (symmetric KL, per project convention) between
each attack class and Benign on per-flow inter-arrival-time (IAT) histograms,
then emit Table 8 (tab:diversity) as long-format LaTeX rows matching the
paper's existing layout.

Metric (replicates earlier project numbers — values can exceed 1):
  - per flow:   h = np.histogram(iat, bins=50, density=True)[0]
  - Hellinger:  sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))   on density vectors
  - "JSD":      KL(p||q) + KL(q||p)                       on re-normalised hist

For each (dataset, attack class) we average pairwise distance across every
(attack flow, benign flow) pair after reservoir-sampling N flows per label.

Datasets/files hardcoded below. Run:  python3 diversity_table.py
"""

import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

# ============================== HARDCODE HERE ==============================

DATASETS = {
    'UAV-CAS':    '../Datasets/UAV-CAS_ts.csv',
    'UNSW-NB15':  '../Datasets/UNSW-NB15_ts.csv',
    'CICIOT2023': '../Datasets/CICIOT2023_ts.csv',
    'UAV-NIDD':   '../Datasets/UAVNIDD_ts.csv',
}

# How the dataset name should appear in the first column of the table.
DATASET_DISPLAY = {
    'UAV-CAS': r'\datasetname',
}

# Case-insensitive prefix match against label strings. First match wins.
# Add any other benign aliases you encounter here.
BENIGN_PATTERNS = ['benign', 'normal', 'background']

OUT_TEX = '../results/Step_4/table8_diversity.tex'
OUT_CSV = '../results/Step_4/diversity_results.csv'

N_SAMPLES_PER_LABEL = 200       # reservoir size per (dataset, label)
MIN_PACKETS         = 100
MAX_PACKETS         = 5000
N_BINS              = 50
CHUNKSIZE           = 5000
SEED                = 0

# ===========================================================================


def parse_list(s):
    if isinstance(s, (list, np.ndarray)):
        return s
    return ast.literal_eval(s)


def pkts_to_iat_hist(pkts):
    """Return density histogram (length N_BINS) of IATs, or None if too short."""
    if len(pkts) > MAX_PACKETS:
        pkts = pkts[:MAX_PACKETS]
    pt = np.asarray(pkts, dtype=np.float64)
    if len(pt) < 2:
        return None
    iat = np.diff(pt)
    iat = iat[iat > 0]
    if iat.size < 2:
        return None
    h, _ = np.histogram(iat, bins=N_BINS, density=True)
    return h.astype(np.float64)


def stream_dataset(name, path, rng):
    """Reservoir-sample density histograms per label while streaming the CSV."""
    print(f"\n[{name}] streaming {path}")
    res  = defaultdict(list)
    seen = defaultdict(int)

    reader = pd.read_csv(path, chunksize=CHUNKSIZE,
                         usecols=lambda c: c.lower() in {'packet_time', 'label'})
    for chunk in reader:
        ren = {c: 'packet_time' for c in chunk.columns if c.lower() == 'packet_time'}
        ren.update({c: 'Label' for c in chunk.columns if c.lower() == 'label'})
        chunk = chunk.rename(columns=ren)
        for pt_str, lab in zip(chunk['packet_time'].values, chunk['Label'].values):
            try:
                pkts = parse_list(pt_str)
            except Exception:
                continue
            if len(pkts) < MIN_PACKETS:
                continue
            h = pkts_to_iat_hist(pkts)
            if h is None:
                continue
            seen[lab] += 1
            n = seen[lab]
            if n <= N_SAMPLES_PER_LABEL:
                res[lab].append(h)
            else:
                j = int(rng.integers(0, n))
                if j < N_SAMPLES_PER_LABEL:
                    res[lab][j] = h

    for lab in sorted(res, key=lambda k: -len(res[k])):
        print(f"    {lab:<30s} seen={seen[lab]:>7d}  kept={len(res[lab])}")
    return res


def hellinger_density(p, q):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2)))


def sym_kl(p, q, eps=1e-10):
    a = p + eps; a = a / a.sum()
    b = q + eps; b = b / b.sum()
    return float(entropy(a, b) + entropy(b, a))


def vs_benign(class_hists, benign_hists):
    H, J = [], []
    for p in class_hists:
        for q in benign_hists:
            H.append(hellinger_density(p, q))
            J.append(sym_kl(p, q))
    return float(np.mean(H)), float(np.mean(J))


def find_benign_label(per_label):
    """Pick the first label whose lowercased string starts with any BENIGN_PATTERNS prefix."""
    for lab in per_label:
        norm = str(lab).strip().lower()
        for pat in BENIGN_PATTERNS:
            if norm.startswith(pat):
                return lab
    return None


def main():
    rng = np.random.default_rng(SEED)

    per_dataset = {}
    for name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"\n[{name}] SKIP - file not found: {path}")
            continue
        per_dataset[name] = stream_dataset(name, path, rng)

    rows = []
    for name, per_label in per_dataset.items():
        ben_lab = find_benign_label(per_label)
        if ben_lab is None or len(per_label[ben_lab]) < 2:
            print(f"  [{name}] no usable benign class found "
                  f"(labels seen: {list(per_label.keys())}); skipping")
            continue
        ben = per_label[ben_lab]
        print(f"  [{name}] using '{ben_lab}' as benign reference ({len(ben)} flows)")

        # Compute every non-benign label in the dataset, no name matching.
        for lab, arrs in per_label.items():
            if lab == ben_lab or len(arrs) < 2:
                continue
            print(f"  {name} :: {lab} vs {ben_lab}  ({len(arrs)} x {len(ben)} pairs)")
            H, J = vs_benign(arrs, ben)
            print(f"    Hellinger={H:.3f}  JSD(symKL)={J:.3f}")
            rows.append({
                'dataset':       name,
                'attack':        lab,
                'benign_label':  ben_lab,
                'n_attack':      len(arrs),
                'n_benign':      len(ben),
                'Hellinger':     H,
                'JSD':           J,
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    write_table(res, OUT_TEX)


def latex_escape(s):
    """Escape characters that LaTeX would otherwise interpret in label text."""
    return (str(s)
            .replace('\\', r'\textbackslash{}')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('$', r'\$')
            .replace('#', r'\#')
            .replace('_', r'\_')
            .replace('{', r'\{')
            .replace('}', r'\}')
            .replace('~', r'\textasciitilde{}')
            .replace('^', r'\textasciicircum{}'))


def write_table(df, out_path):
    lines = [
        r"% Table tab:diversity",
        r"% Hellinger and 'JSD' (= symmetric KL, per project convention) between",
        r"% attack-class flows and Benign flows on per-flow IAT density histograms.",
        r"% Higher = more separable from benign.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical diversity comparison (Hellinger Distance / JSD) on inter-arrival time distributions. Higher values indicate greater separation between benign and attack traffic under varying network conditions.}",
        r"\label{tab:diversity}",
        r"\footnotesize",
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"\textbf{Dataset} & \textbf{Attack} & \textbf{Hellinger} & \textbf{JSD} \\",
        r"\hline",
    ]

    for ds_name in DATASETS:
        if ds_name not in df['dataset'].values:
            continue
        sub = df[df['dataset'] == ds_name].sort_values('Hellinger', ascending=False)
        if sub.empty:
            continue
        ds_disp = DATASET_DISPLAY.get(ds_name, ds_name)
        for i, r in enumerate(sub.itertuples(index=False)):
            cell0 = ds_disp if i == 0 else ''
            atk = latex_escape(r.attack)
            lines.append(f"{cell0:<14s} & {atk:<14s} & {r.Hellinger:.2f} & {r.JSD:.2f} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(out_path).write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
