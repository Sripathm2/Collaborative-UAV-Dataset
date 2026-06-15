#!/usr/bin/env python3
"""run_confusion.py
Fits each baseline once on UAV-CAS using the canonical (input, scaler)
combo and saves a confusion matrix per model for the 10-panel figure.
Single 80/20 stratified split (no CV). RUS('not minority') on train.
Self-contained.
"""
from __future__ import annotations
import ast
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import (LabelEncoder, StandardScaler,
                                      MinMaxScaler, RobustScaler)
from sklearn.metrics         import confusion_matrix

from models import Baseline, free_keras

# =========================================================================
# CONFIG
# =========================================================================
DATASETS_DIR = Path("../Datasets")
RESULTS_DIR  = Path("../results/ids")
(RESULTS_DIR / "confusion").mkdir(parents=True, exist_ok=True)

DATASET_FILES = {
    "UAV-CAS": {"ts":   DATASETS_DIR / "UAV-CAS_ts.csv",
                 "stat": DATASETS_DIR / "UAV-CAS_stat.csv"},
}

RANDOM_STATE     = 41
SEQ_LEN          = 512
ROW_CAP          = 1000
CANONICAL_INPUT  = "ts"
CANONICAL_SCALER = "standard"
MODELS = ["1D-CNN", "LSTM", "RF", "SGD", "LR",
          "MLP", "LightGBM", "ConvNet", "TinyML", "CNN-BiLSTM"]

UAVCAS_SINGLE = ["Benign", "DoS", "DDoS", "Blackhole", "Wormhole", "Replay"]
UAVCAS_COLLAB = [
    "Blackhole+DoS", "Blackhole+DDoS", "DoS+Wormhole", "DDoS+Wormhole",
    "Blackhole+Wormhole", "Blackhole+Replay", "DDoS+Replay",
    "DoS+Replay", "Replay+Wormhole",
]
UAVCAS_T12_CLASSES = UAVCAS_SINGLE + UAVCAS_COLLAB


# =========================================================================
# LABEL NORMALIZATION + LOADERS (mirrors run_t12)
# =========================================================================
def _norm_uavcas(label):
    if not isinstance(label, str): return "Benign"
    s = label.strip()
    if "+" not in s: return s
    parts = sorted(p.strip() for p in s.split("+") if p.strip())
    return "+".join(parts)


def _parse_list_cell(cell):
    if isinstance(cell, list): return cell
    if not isinstance(cell, str): return []
    s = cell.strip()
    if not s or s in ("[]", "nan", "NaN"): return []
    try: return ast.literal_eval(s)
    except Exception: return []


def _ts_to_features(times, sizes, seq_len):
    if len(times) >= 2:
        iat = np.diff(np.asarray(times, dtype=np.float64))
    else:
        iat = np.zeros(0, dtype=np.float64)
    iat_v  = np.zeros(seq_len, dtype=np.float32)
    size_v = np.zeros(seq_len, dtype=np.float32)
    n_iat = min(len(iat),  seq_len)
    n_sz  = min(len(sizes), seq_len)
    iat_v[:n_iat]  = iat[:n_iat]
    size_v[:n_sz]  = np.asarray(sizes[:n_sz], dtype=np.float32)
    return np.concatenate([iat_v, size_v]).astype(np.float32)


def _load_ts(path, normer, row_cap):
    df = pd.read_csv(path, low_memory=False)
    if row_cap and len(df) > row_cap:
        df = df.sample(n=row_cap, random_state=RANDOM_STATE).reset_index(drop=True)
    feats, labels = [], []
    for _, row in df.iterrows():
        norm = normer(row["Label"])
        if norm is None: continue
        ts_l = _parse_list_cell(row["packet_time"])
        sz_l = _parse_list_cell(row["packet_size"])
        if len(ts_l) < 2 and len(sz_l) < 2: continue
        feats.append(_ts_to_features(ts_l, sz_l, SEQ_LEN))
        labels.append(norm)
    if not feats:
        raise RuntimeError(f"no usable rows in {path}")
    return np.vstack(feats), pd.Series(labels, name="Label")


def _load_stat(path, normer, row_cap):
    df = pd.read_csv(path, low_memory=False)
    lbl_col = "Label" if "Label" in df.columns else "label"
    df["__lbl__"] = df[lbl_col].apply(normer)
    df = df[df["__lbl__"].notna()].reset_index(drop=True)
    if row_cap and len(df) > row_cap:
        df = df.sample(n=row_cap, random_state=RANDOM_STATE).reset_index(drop=True)
    y = df["__lbl__"].copy(); y.name = "Label"
    X_df = df.drop(columns=[c for c in (lbl_col, "__lbl__") if c in df.columns])
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_df.values.astype(np.float32), y


def _load_dataset(name, kind, normer):
    path = DATASET_FILES[name][kind]
    if kind == "ts":   return _load_ts(path, normer, ROW_CAP)
    if kind == "stat": return _load_stat(path, normer, ROW_CAP)
    raise ValueError(kind)


def _get_scaler(kind):
    if kind == "standard": return StandardScaler()
    if kind == "minmax":   return MinMaxScaler()
    if kind == "robust":   return RobustScaler()
    raise ValueError(kind)


def _rus(X, y):
    rus = RandomUnderSampler(sampling_strategy="not minority",
                              random_state=RANDOM_STATE)
    yv = y.values if isinstance(y, pd.Series) else np.asarray(y)
    return rus.fit_resample(X, yv)


# =========================================================================
# MAIN
# =========================================================================
def main():
    print(f"loading UAV-CAS / {CANONICAL_INPUT} ...")
    X, y = _load_dataset("UAV-CAS", CANONICAL_INPUT, _norm_uavcas)
    keep = set(UAVCAS_T12_CLASSES)
    mask = y.isin(keep).values
    X = X[mask]; y = y[mask].reset_index(drop=True)
    present = [c for c in UAVCAS_T12_CLASSES if c in set(y.values)]
    le = LabelEncoder().fit(present)
    y_int = le.transform(y.values)
    classes = list(le.classes_)
    print(f"  {len(y_int):,} rows  {len(classes)} classes")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_int, test_size=0.2, random_state=RANDOM_STATE, stratify=y_int)
    X_tr, y_tr = _rus(X_tr, y_tr)
    y_tr = np.asarray(y_tr, dtype=np.int64)

    sc = _get_scaler(CANONICAL_SCALER)
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)

    out_dir = RESULTS_DIR / "confusion"
    for model_name in MODELS:
        t0 = time.time()
        model = None
        try:
            model = Baseline(model_name, n_classes=len(classes))
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            cm = confusion_matrix(y_te, y_pred, labels=range(len(classes)))
            np.save(out_dir / f"cm_{model_name}.npy", cm)
            print(f"  {model_name:11s} done ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAIL {model_name}: {e}")
        finally:
            del model; free_keras(); gc.collect()

    pd.Series(classes).to_csv(out_dir / "classes.csv",
                               index=False, header=False)
    print(f"wrote confusion matrices to {out_dir}")


if __name__ == "__main__":
    main()
