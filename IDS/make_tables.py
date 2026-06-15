#!/usr/bin/env python3
"""make_tables.py
Reads CSVs from results/ids/{t11,t12,t13}/ and emits LaTeX tabular bodies.
Each .tex file contains ONLY the body (no \\begin{table}/caption).
"""
from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================================
# CONFIG
# =========================================================================
DATASETS_DIR = Path("../Datasets")
RESULTS_DIR  = Path("../results/ids")
TABLES_DIR   = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILES = {
    "UAV-CAS":    {"ts":   DATASETS_DIR / "UAV-CAS_ts.csv",
                    "stat": DATASETS_DIR / "UAV-CAS_stat.csv"},
    "UNSW-NB15":  {"ts":   DATASETS_DIR / "UNSW-NB15_ts.csv",
                    "stat": DATASETS_DIR / "UNSW-NB15_stat.csv"},
    "CICIOT2023": {"ts":   DATASETS_DIR / "CICIOT2023_ts.csv",
                    "stat": DATASETS_DIR / "CICIOT2023_stat.csv"},
    "UAV-NIDD":   {"ts":   DATASETS_DIR / "UAVNIDD_ts.csv"},
    "CICIDS2017": {"stat": DATASETS_DIR / "CICIDS2017_stat.csv"},
}

MODELS  = ["1D-CNN", "LSTM", "RF", "SGD", "LR",
           "MLP", "LightGBM", "ConvNet", "TinyML", "CNN-BiLSTM"]
SCALERS = ["standard", "minmax", "robust"]
INPUTS  = ["ts", "stat"]
METRICS = ["f1", "recall", "auroc"]

UAVCAS_SINGLE = ["Benign", "DoS", "DDoS", "Blackhole", "Wormhole", "Replay"]
UAVCAS_COLLAB = [
    "Blackhole+DoS", "Blackhole+DDoS", "DoS+Wormhole", "DDoS+Wormhole",
    "Blackhole+Wormhole", "Blackhole+Replay", "DDoS+Replay",
    "DoS+Replay", "Replay+Wormhole",
]
UAVCAS_T12_CLASSES = UAVCAS_SINGLE + UAVCAS_COLLAB


def _has_input(name, kind):
    return name in DATASET_FILES and kind in DATASET_FILES[name]


def fmt(v):
    if pd.isna(v): return "[PH]"
    return f"{v*100:.2f}"


def emit_t11():
    datasets = list(DATASET_FILES.keys())
    for sc in SCALERS:
        path = RESULTS_DIR / "t11" / f"t11_{sc}.csv"
        if not path.exists():
            print(f"missing {path}"); continue
        df = pd.read_csv(path)
        rows = []
        for m in MODELS:
            sub = df[df["model"] == m]
            cells = [m]
            for ds in datasets:
                for inp in INPUTS:
                    if not _has_input(ds, inp):
                        cells.append("--"); continue
                    r = sub[(sub["dataset"]==ds) & (sub["input_type"]==inp)]
                    v = r["f1_weighted"].values[0] if len(r) else np.nan
                    cells.append(fmt(v))
            rows.append(" & ".join(cells) + r" \\")
        header_cells = ["Model"]
        for ds in datasets:
            for inp in INPUTS:
                header_cells.append(f"{ds}/{inp}")
        header = " & ".join(header_cells) + r" \\"
        out = "% header (reference only):\n% " + header + "\n" + "\n".join(rows) + "\n"
        outp = TABLES_DIR / f"t11_{sc}.tex"
        outp.write_text(out)
        print(f"wrote {outp}")


def emit_t12():
    for inp in INPUTS:
        for sc in SCALERS:
            for metric in METRICS:
                path = RESULTS_DIR / "t12" / f"t12_{inp}_{sc}_{metric}.csv"
                if not path.exists():
                    print(f"missing {path}"); continue
                df = pd.read_csv(path)
                cls_present = [c for c in UAVCAS_T12_CLASSES if c in df.columns]
                rows = []
                for m in MODELS:
                    r = df[df["model"]==m]
                    if not len(r): continue
                    cells = [m] + [fmt(r[c].values[0]) for c in cls_present]
                    rows.append(" & ".join(cells) + r" \\")
                out = "\n".join(rows) + "\n"
                outp = TABLES_DIR / f"t12_{inp}_{sc}_{metric}.tex"
                outp.write_text(out)
                print(f"wrote {outp}")


def emit_t13():
    for inp in INPUTS:
        for sc in SCALERS:
            path = RESULTS_DIR / "t13" / f"t13_{inp}_{sc}.csv"
            if not path.exists():
                print(f"missing {path}"); continue
            df = pd.read_csv(path).set_index("model")
            rows = []
            for comp in UAVCAS_COLLAB:
                cells = [comp]
                for m in MODELS:
                    if m in df.index and comp in df.columns:
                        cells.append(fmt(df.loc[m, comp]))
                    else:
                        cells.append("[PH]")
                rows.append(" & ".join(cells) + r" \\")
            out = "\n".join(rows) + "\n"
            outp = TABLES_DIR / f"t13_{inp}_{sc}.tex"
            outp.write_text(out)
            print(f"wrote {outp}")


if __name__ == "__main__":
    emit_t11()
    emit_t12()
    emit_t13()
