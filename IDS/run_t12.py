#!/usr/bin/env python3
"""run_t12.py
T12: UAV-CAS native multi-class. Per-class F1 / Recall / AUROC.
Self-contained. Emits 18 CSVs (2 inputs x 3 scalers x 3 metrics).
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
from sklearn.preprocessing   import (LabelEncoder, label_binarize,
                                      StandardScaler, MinMaxScaler,
                                      RobustScaler)
from sklearn.metrics         import f1_score, recall_score, roc_auc_score

from models import Baseline, free_keras

# =========================================================================
# CONFIG
# =========================================================================
DATASETS_DIR = Path("../Datasets")
RESULTS_DIR  = Path("../results/ids")
(RESULTS_DIR / "t12").mkdir(parents=True, exist_ok=True)

DATASET_FILES = {
    "UAV-CAS":    {"ts":   DATASETS_DIR / "UAV-CAS_ts.csv",
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
METRICS = ["f1", "recall", "auroc"]

UAVCAS_SINGLE = ["Benign", "DoS", "DDoS", "Blackhole", "Wormhole", "Replay"]
UAVCAS_COLLAB = [
    "Blackhole+DoS", "Blackhole+DDoS", "DoS+Wormhole", "DDoS+Wormhole",
    "Blackhole+Wormhole", "Blackhole+Replay", "DDoS+Replay",
    "DoS+Replay", "Replay+Wormhole",
]
UAVCAS_T12_CLASSES = UAVCAS_SINGLE + UAVCAS_COLLAB


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
def task_t12(model_name, input_type, scaler_kind):
    X, y = _load_dataset("UAV-CAS", input_type, _norm_uavcas)
    keep = set(UAVCAS_T12_CLASSES)
    mask = y.isin(keep).values
    X = X[mask]; y = y[mask].reset_index(drop=True)
    present = [c for c in UAVCAS_T12_CLASSES if c in set(y.values)]
    le = LabelEncoder().fit(present)
    y_int = le.transform(y.values)
    classes = list(le.classes_)
    n_classes = len(classes)

    accum = {c: {"f1": [], "recall": [], "auroc": []} for c in classes}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)
    for tr, te in skf.split(X, y_int):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y_int[tr], y_int[te]
        X_tr, y_tr = _rus(X_tr, y_tr)
        y_tr = np.asarray(y_tr, dtype=np.int64)
        sc = _get_scaler(scaler_kind)
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_te = sc.transform(X_te).astype(np.float32)
        model = Baseline(model_name, n_classes=n_classes)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        y_pred = np.argmax(proba, axis=1)
        f1s  = f1_score(y_te, y_pred, labels=range(n_classes),
                         average=None, zero_division=0)
        recs = recall_score(y_te, y_pred, labels=range(n_classes),
                             average=None, zero_division=0)
        y_bin = label_binarize(y_te, classes=range(n_classes))
        for k in range(n_classes):
            accum[classes[k]]["f1"].append(float(f1s[k]))
            accum[classes[k]]["recall"].append(float(recs[k]))
            if y_bin[:, k].sum() == 0 or y_bin[:, k].sum() == len(y_bin):
                continue
            try:
                accum[classes[k]]["auroc"].append(
                    float(roc_auc_score(y_bin[:, k], proba[:, k])))
            except Exception:
                pass
        del model; free_keras()

    out = {c: {m: (float(np.mean(vs)) if vs else float("nan"))
               for m, vs in md.items()} for c, md in accum.items()}
    return {"classes": classes, "result": out}


# =========================================================================
# MAIN
# =========================================================================
def main():
    for inp in INPUTS:
        for sc in SCALERS:
            rows_by_metric = {m: [] for m in METRICS}
            classes_seen = None
            for model_name in MODELS:
                t0 = time.time()
                try:
                    res = task_t12(model_name, inp, sc)
                except Exception as e:
                    print(f"  FAIL {model_name}/{inp}/{sc}: {e}")
                    res = None
                if res is None:
                    classes = classes_seen or UAVCAS_T12_CLASSES
                    nan_row = {c: float("nan") for c in classes}
                    for m in METRICS:
                        rows_by_metric[m].append({"model": model_name, **nan_row})
                else:
                    classes = res["classes"]
                    classes_seen = classes
                    for m in METRICS:
                        row = {"model": model_name}
                        for c in classes:
                            row[c] = res["result"][c][m]
                        rows_by_metric[m].append(row)
                print(f"{inp} {sc:9s} {model_name:11s} done "
                      f"({time.time()-t0:.1f}s)")
                gc.collect()
            for m in METRICS:
                df = pd.DataFrame(rows_by_metric[m])
                path = RESULTS_DIR / "t12" / f"t12_{inp}_{sc}_{m}.csv"
                df.to_csv(path, index=False)
                print(f"  wrote {path}")


if __name__ == "__main__":
    main()
