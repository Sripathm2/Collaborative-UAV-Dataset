#!/usr/bin/env python3
"""run_t13.py
T13: collaborative-attack AUROC, multi-label sigmoid heads.
Train on single-attack only (Benign + 5 primitives), test on 9 collab
compositions. Self-contained. Emits 6 CSVs (2 inputs x 3 scalers).
"""
from __future__ import annotations
import ast
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import (StandardScaler, MinMaxScaler,
                                      RobustScaler)
from sklearn.metrics         import roc_auc_score

from models import Baseline, free_keras

# =========================================================================
# CONFIG
# =========================================================================
DATASETS_DIR = Path("../Datasets")
RESULTS_DIR  = Path("../results/ids")
(RESULTS_DIR / "t13").mkdir(parents=True, exist_ok=True)

DATASET_FILES = {
    "UAV-CAS": {"ts":   DATASETS_DIR / "UAV-CAS_ts.csv",
                 "stat": DATASETS_DIR / "UAV-CAS_stat.csv"},
}

RANDOM_STATE = 41
N_FOLDS      = 5
SEQ_LEN      = 512
ROW_CAP      = None
MODELS  = ["1D-CNN", "LSTM", "RF", "SGD", "LR",
           "MLP", "LightGBM", "ConvNet", "TinyML", "CNN-BiLSTM"]
SCALERS = ["standard", "minmax", "robust"]
INPUTS  = ["ts", "stat"]

UAVCAS_SINGLE = ["Benign", "DoS", "DDoS", "Blackhole", "Wormhole", "Replay"]
UAVCAS_COLLAB = [
    "Blackhole+DoS", "Blackhole+DDoS", "DoS+Wormhole", "DDoS+Wormhole",
    "Blackhole+Wormhole", "Blackhole+Replay", "DDoS+Replay",
    "DoS+Replay", "Replay+Wormhole",
]
PRIMITIVES = ["DoS", "DDoS", "Blackhole", "Wormhole", "Replay"]


# =========================================================================
# LABEL NORMALIZATION
# =========================================================================
def _norm_uavcas(label):
    if not isinstance(label, str): return "Benign"
    s = label.strip()
    if "+" not in s: return s
    parts = sorted(p.strip() for p in s.split("+") if p.strip())
    return "+".join(parts)


# =========================================================================
# LOADERS
# =========================================================================
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


# =========================================================================
# PREPROCESS
# =========================================================================
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
# CORE TASK
# =========================================================================
def task_t13(model_name, input_type, scaler_kind):
    X, y = _load_dataset("UAV-CAS", input_type, _norm_uavcas)
    y_str = y.values
    is_single = np.array([("+" not in s) for s in y_str])
    X_s, y_s = X[is_single],  y_str[is_single]
    X_c, y_c = X[~is_single], y_str[~is_single]
    ms = np.isin(y_s, UAVCAS_SINGLE);  X_s, y_s = X_s[ms], y_s[ms]
    mc = np.isin(y_c, UAVCAS_COLLAB);  X_c, y_c = X_c[mc], y_c[mc]

    PRIM_INDEX = {p: i for i, p in enumerate(PRIMITIVES)}
    N_HEADS = len(PRIMITIVES)

    def to_multihot(s):
        Y = np.zeros(N_HEADS, dtype=np.float32)
        if s == "Benign": return Y
        for p in s.split("+"):
            if p in PRIM_INDEX: Y[PRIM_INDEX[p]] = 1.0
        return Y

    Y_single = np.vstack([to_multihot(s) for s in y_s])
    accum = {c: [] for c in UAVCAS_COLLAB}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)
    for tr, te in skf.split(X_s, y_s):
        X_tr, X_te = X_s[tr], X_s[te]
        y_te_str   = y_s[te]
        X_tr_bal, y_tr_str_bal = _rus(X_tr, y_s[tr])
        Y_tr_bal = np.vstack([to_multihot(s) for s in
                                (y_tr_str_bal if isinstance(y_tr_str_bal, np.ndarray)
                                 else y_tr_str_bal.values)])
        sc = _get_scaler(scaler_kind)
        X_tr_bal = sc.fit_transform(X_tr_bal).astype(np.float32)
        X_te     = sc.transform(X_te).astype(np.float32)
        X_co     = sc.transform(X_c).astype(np.float32)
        model = Baseline(model_name, n_classes=N_HEADS, multilabel=True)
        model.fit(X_tr_bal, Y_tr_bal)
        benign_idx = (y_te_str == "Benign")
        X_benign = X_te[benign_idx]
        proba_neg_all = (model.predict_proba(X_benign)
                         if len(X_benign) else np.zeros((0, N_HEADS)))
        proba_pos_all = (model.predict_proba(X_co)
                         if len(X_co) else np.zeros((0, N_HEADS)))
        for comp in UAVCAS_COLLAB:
            parts = comp.split("+")
            if not all(p in PRIM_INDEX for p in parts): continue
            head_idx = [PRIM_INDEX[p] for p in parts]
            pos_mask = (y_c == comp)
            if pos_mask.sum() == 0 or proba_neg_all.shape[0] == 0: continue
            score_pos = proba_pos_all[pos_mask][:, head_idx].mean(axis=1)
            score_neg = proba_neg_all[:, head_idx].mean(axis=1)
            y_true  = np.concatenate([np.ones(len(score_pos)),
                                       np.zeros(len(score_neg))])
            y_score = np.concatenate([score_pos, score_neg])
            try:
                accum[comp].append(float(roc_auc_score(y_true, y_score)))
            except Exception:
                pass
        del model; free_keras()

    return {c: (float(np.mean(vs)) if vs else float("nan"))
            for c, vs in accum.items()}


# =========================================================================
# MAIN
# =========================================================================
def main():
    for inp in INPUTS:
        for sc in SCALERS:
            rows = []
            for model_name in MODELS:
                t0 = time.time()
                try:
                    res = task_t13(model_name, inp, sc)
                except Exception as e:
                    print(f"  FAIL {model_name}/{inp}/{sc}: {e}")
                    res = {c: float("nan") for c in UAVCAS_COLLAB}
                rows.append({"model": model_name, **res})
                print(f"{inp} {sc:9s} {model_name:11s} done "
                      f"({time.time()-t0:.1f}s)")
                gc.collect()
            df = pd.DataFrame(rows)
            path = RESULTS_DIR / "t13" / f"t13_{inp}_{sc}.csv"
            df.to_csv(path, index=False)
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
