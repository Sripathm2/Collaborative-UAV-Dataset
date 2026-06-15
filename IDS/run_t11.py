#!/usr/bin/env python3
"""run_t11.py
T11: cross-dataset Benign-vs-DoS binary classification.
Memory-conscious + checkpoint-resumable.

CHECKPOINT BEHAVIOR
- Every completed (model, dataset, input, scaler) result is appended
  immediately to results/ids/t11/t11_checkpoint.csv and fsync'd.
- On startup, that file is read; combos already present are SKIPPED.
- After all work done (or on every dataset boundary), per-scaler CSVs
  are (re)derived from the checkpoint.
- Safe to Ctrl-C / OOM-kill at any point; rerun resumes.

Self-contained.
"""
from __future__ import annotations
import ast
import csv
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing   import (LabelEncoder, StandardScaler,
                                      MinMaxScaler, RobustScaler)
from sklearn.metrics         import f1_score

from models import Baseline, free_keras

# =========================================================================
# CONFIG
# =========================================================================
DATASETS_DIR = Path("../Datasets")
RESULTS_DIR  = Path("../results/ids")
T11_DIR      = RESULTS_DIR / "t11"
T11_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT   = T11_DIR / "t11_checkpoint.csv"

DATASET_FILES = {
    "UAV-CAS":    {"ts":   DATASETS_DIR / "UAV-CAS_ts.csv",
                    "stat": DATASETS_DIR / "UAV-CAS_stat.csv"},
    "UNSW-NB15":  {#"ts":   DATASETS_DIR / "UNSW-NB15_ts.csv",
                    "stat": DATASETS_DIR / "UNSW-NB15_stat.csv"},
    "CICIOT2023": {"ts":   DATASETS_DIR / "CICIOT2023_ts.csv",
                    "stat": DATASETS_DIR / "CICIOT2023_stat.csv"},
    "UAV-NIDD":   {"ts":   DATASETS_DIR / "UAVNIDD_ts.csv"},
    "CICIDS2017": {"stat": DATASETS_DIR / "CICIDS2017_stat.csv"},
}

RANDOM_STATE = 41
N_FOLDS      = 5
SEQ_LEN      = 512
ROW_CAP      = None
MODELS  = ["1D-CNN", "LSTM", "RF", "SGD", "LR",
           "MLP", "LightGBM", "ConvNet", "TinyML", "CNN-BiLSTM"]
SCALERS = ["standard", "minmax", "robust"]
INPUTS  = ["ts", "stat"]

CKPT_COLS = ["model", "dataset", "input_type", "scaler", "f1_weighted"]


# =========================================================================
# CHECKPOINT I/O
# =========================================================================
def _load_checkpoint() -> tuple[set, list]:
    """Return (set_of_done_keys, list_of_existing_rows)."""
    if not CHECKPOINT.exists():
        return set(), []
    try:
        df = pd.read_csv(CHECKPOINT)
    except Exception as e:
        print(f"  checkpoint unreadable ({e}); starting fresh")
        return set(), []
    done = set()
    rows = []
    for _, r in df.iterrows():
        key = (str(r["model"]), str(r["dataset"]),
               str(r["input_type"]), str(r["scaler"]))
        done.add(key)
        rows.append({c: r[c] for c in CKPT_COLS})
    print(f"  checkpoint: {len(done)} combos already done")
    return done, rows


def _append_checkpoint(row: dict):
    """Append one row to checkpoint, flush + fsync so a kill is safe."""
    new = not CHECKPOINT.exists()
    with open(CHECKPOINT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CKPT_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CKPT_COLS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _derive_per_scaler_csvs(all_rows: list):
    """Re-derive t11_{scaler}.csv from the full row list."""
    if not all_rows:
        return
    df = pd.DataFrame(all_rows)
    for sc in SCALERS:
        sub = df[df["scaler"] == sc].copy()
        path = T11_DIR / f"t11_{sc}.csv"
        sub.to_csv(path, index=False)
        print(f"  derived {path}  ({len(sub)} rows)")


# =========================================================================
# LABEL NORMALIZATION
# =========================================================================
def _norm_t11(label):
    if not isinstance(label, str):
        return None
    s = label.strip().lower()
    if s.startswith("benign") or s == "normal":
        return "Benign"
    if s.startswith("dos") and not s.startswith("ddos"):
        return "DoS"
    return None


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
    del df
    if not feats:
        raise RuntimeError(f"no usable rows in {path}")
    X = np.vstack(feats).astype(np.float32)
    y = pd.Series(labels, name="Label")
    del feats, labels
    return X, y


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
    X = X_df.values.astype(np.float32)
    del df, X_df
    return X, y


def _has_input(name, kind):
    return name in DATASET_FILES and kind in DATASET_FILES[name]


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
    Xb, yb = rus.fit_resample(X, yv)
    del rus
    return Xb, yb


# =========================================================================
# CV ON CACHED ARRAYS
# =========================================================================
def _cv_one(X, y_int, n_classes, folds, model_name, scaler_kind):
    f1s = []
    for tr, te in folds:
        X_tr_raw, X_te_raw = X[tr], X[te]
        y_tr_raw, y_te     = y_int[tr], y_int[te]
        X_tr, y_tr = _rus(X_tr_raw, y_tr_raw)
        y_tr = np.asarray(y_tr, dtype=np.int64)
        sc = _get_scaler(scaler_kind)
        X_tr = sc.fit_transform(X_tr).astype(np.float32)
        X_te = sc.transform(X_te_raw).astype(np.float32)
        model = Baseline(model_name, n_classes=n_classes)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        f1s.append(f1_score(y_te, y_pred, average="weighted",
                             zero_division=0))
        del model, sc, X_tr, X_te, y_tr, X_tr_raw, X_te_raw, y_tr_raw
        free_keras(); gc.collect()
    return float(np.mean(f1s)) if f1s else float("nan")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=== T11 with checkpoint/resume ===")
    done_keys, existing_rows = _load_checkpoint()
    all_rows = list(existing_rows)
    datasets = list(DATASET_FILES.keys())

    def _record(row: dict):
        key = (row["model"], row["dataset"],
               row["input_type"], row["scaler"])
        if key in done_keys:
            return
        _append_checkpoint(row)
        done_keys.add(key)
        all_rows.append(row)

    for ds in datasets:
        for inp in INPUTS:
            # missing combo -> write NaN rows for every (scaler, model)
            if not _has_input(ds, inp):
                for sc in SCALERS:
                    for m in MODELS:
                        _record({"model": m, "dataset": ds,
                                  "input_type": inp, "scaler": sc,
                                  "f1_weighted": float("nan")})
                continue

            # determine which (scaler, model) combos are missing for this
            # (ds, inp).  If all 30 are present, skip the load entirely.
            todo = [(sc, m) for sc in SCALERS for m in MODELS
                    if (m, ds, inp, sc) not in done_keys]
            if not todo:
                print(f"\n=== SKIP {ds}/{inp} (all 30 combos in checkpoint) ===")
                continue

            print(f"\n=== loading {ds} / {inp} "
                  f"({len(todo)}/{len(SCALERS)*len(MODELS)} combos pending) ===")
            t0 = time.time()
            try:
                X, y = _load_dataset(ds, inp, _norm_t11)
            except Exception as e:
                print(f"  LOAD FAIL {ds}/{inp}: {e}")
                for sc, m in todo:
                    _record({"model": m, "dataset": ds,
                              "input_type": inp, "scaler": sc,
                              "f1_weighted": float("nan")})
                continue

            le = LabelEncoder()
            y_int = le.fit_transform(y.values)
            n_classes = len(le.classes_)
            print(f"  loaded shape={X.shape} classes={list(le.classes_)} "
                  f"({time.time()-t0:.1f}s)")

            if n_classes < 2:
                print(f"  SKIP {ds}/{inp}: only {n_classes} class(es)")
                for sc, m in todo:
                    _record({"model": m, "dataset": ds,
                              "input_type": inp, "scaler": sc,
                              "f1_weighted": float("nan")})
                del X, y, y_int, le; gc.collect()
                continue

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=RANDOM_STATE)
            folds = list(skf.split(X, y_int))

            for sc, m in todo:
                t1 = time.time()
                try:
                    f1 = _cv_one(X, y_int, n_classes, folds, m, sc)
                except Exception as e:
                    print(f"  FAIL {m}/{ds}/{inp}/{sc}: {e}")
                    f1 = float("nan")
                print(f"  {sc:9s} {m:11s} f1={f1:.4f} "
                      f"({time.time()-t1:.1f}s)")
                _record({"model": m, "dataset": ds,
                          "input_type": inp, "scaler": sc,
                          "f1_weighted": f1})
                gc.collect()

            del X, y, y_int, le, folds, skf
            free_keras(); gc.collect()
            print(f"  freed {ds}/{inp}")

            # snapshot per-scaler CSVs at each dataset boundary so partial
            # progress is queryable.
            _derive_per_scaler_csvs(all_rows)

    # final write
    _derive_per_scaler_csvs(all_rows)
    print("\nDONE.")


if __name__ == "__main__":
    main()