#!/usr/bin/env python3
"""models.py
Uniform fit/predict/predict_proba wrapper for the 10 UAV-CAS IDS baselines.
Architectures (Bi-LSTM, ConvNet, CNN, LSTM, MLP, TinyML) are inlined here,
not imported from external files.
"""
from __future__ import annotations
import os
import gc
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models   import Sequential, Model
from tensorflow.keras.layers   import (Conv1D, Dense, Dropout, LSTM,
                                        MaxPooling1D, Reshape, Bidirectional,
                                        BatchNormalization, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1
from tensorflow.keras.utils    import to_categorical

from sklearn.ensemble    import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput  import MultiOutputClassifier
import lightgbm as lgb

# ---- knobs (mirrored from run_*.py call sites) --------------------------
RANDOM_STATE   = 41
DL_EPOCHS      = 5
DL_BATCH       = 256
DL_VERBOSE     = 1
SKLEARN_N_JOBS = -1
MODEL_BATCH_OVERRIDES = {"LSTM": 32, "CNN-BiLSTM": 32}

KERAS_SEQ_MODELS  = {"1D-CNN", "LSTM", "ConvNet", "CNN-BiLSTM"}
KERAS_FLAT_MODELS = {"MLP"}
SKLEARN_MODELS    = {"RF", "SGD", "LR", "LightGBM"}


# =========================================================================
# architecture builders (inlined from Bi_LSTM.py / ConvNet.py /
# Cyber_physical.py / TinyML.py)
# =========================================================================
def _arch_cnn(input_shape, n):
    m = Sequential([
        keras.Input(shape=input_shape),
        Conv1D(64,  kernel_size=3, activation='relu'),
        Conv1D(128, kernel_size=3, activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(n, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
    return m


def _arch_lstm(input_shape, n):
    reg = L1(1e-4)
    m = Sequential([
        keras.Input(shape=input_shape),
        LSTM(512, kernel_regularizer=reg, return_sequences=True,  name='HL1'),
        LSTM(512, kernel_initializer='he_uniform', activation='relu',
             return_sequences=False, name='HL2'),
        Dropout(0.5),
        Dense(n, activation='softmax', name='Output'),
    ])
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
    return m


def _arch_convnet(input_shape, n):
    m = Sequential([
        keras.Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(n, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
    return m


def _arch_bilstm(input_shape, n):
    m = Sequential([
        keras.Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=5),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False)),
        Reshape((128, 1)),
        MaxPooling1D(pool_size=5),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(n, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
    return m


def _arch_mlp(input_shape, n):
    m = Sequential([
        keras.Input(shape=input_shape),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(n, activation='sigmoid', name='Output'),
    ])
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
    return m


def _arch_tinyml_stage1(input_shape, n):
    m = Sequential([
        keras.Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Flatten(),
        Dense(n, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])
    return m


def _tinyml_rf():
    return RandomForestClassifier(n_estimators=100, n_jobs=SKLEARN_N_JOBS,
                                   random_state=RANDOM_STATE)


def _build_keras(name, input_shape, n_classes):
    if name == "1D-CNN":     return _arch_cnn(input_shape, n_classes)
    if name == "LSTM":       return _arch_lstm(input_shape, n_classes)
    if name == "ConvNet":    return _arch_convnet(input_shape, n_classes)
    if name == "CNN-BiLSTM": return _arch_bilstm(input_shape, n_classes)
    if name == "MLP":        return _arch_mlp(input_shape, n_classes)
    raise ValueError(name)


def _build_keras_multilabel(name, input_shape, n_heads):
    base = _build_keras(name, input_shape, n_heads)
    inputs = base.inputs
    x = base.layers[-2].output
    out = Dense(n_heads, activation="sigmoid", name="ml_out")(x)
    new_model = Model(inputs=inputs, outputs=out)
    new_model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss="binary_crossentropy", metrics=["accuracy"])
    return new_model


def _build_sklearn(name):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, n_jobs=SKLEARN_N_JOBS,
                                       random_state=RANDOM_STATE)
    if name == "SGD":
        return SGDClassifier(loss="log_loss", random_state=RANDOM_STATE,
                              max_iter=50, tol=1e-3)
    if name == "LR":
        return LogisticRegression(max_iter=300, n_jobs=SKLEARN_N_JOBS,
                                   random_state=RANDOM_STATE)
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=200, n_jobs=SKLEARN_N_JOBS,
                                   random_state=RANDOM_STATE, verbose=-1)
    raise ValueError(name)


def free_keras():
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


# =========================================================================
# Baseline wrapper
# =========================================================================
class Baseline:
    """Uniform classifier wrapper.

    Single-label : fit(X, y) where y is 1-D class indices.
    Multi-label  : fit(X, Y) where Y is (N, n_heads) multi-hot float32.
    predict_proba returns (N, n_classes) in single, (N, n_heads) in multi.
    """
    def __init__(self, name: str, n_classes: int,
                 multilabel: bool = False,
                 epochs: int = DL_EPOCHS, batch: int = DL_BATCH):
        self.name       = name
        self.n_classes  = n_classes
        self.multilabel = multilabel
        self.epochs     = epochs
        self.batch      = MODEL_BATCH_OVERRIDES.get(name, batch)
        self._model              = None
        self._tinyml_extractor   = None
        self._tinyml_rf          = None
        self._tinyml_rfs_per_head: list = []

    # ---- input shaping ----
    def _shape(self, X):
        if self.name in KERAS_SEQ_MODELS:
            return X.reshape(X.shape[0], X.shape[1], 1)
        return X

    # ---- fit ----
    def fit(self, X, y):
        Xs = self._shape(X)
        if not self.multilabel:
            y = np.asarray(y, dtype=np.int64)
            return self._fit_single(Xs, y)
        Y = np.asarray(y, dtype=np.float32)
        return self._fit_multi(Xs, Y)

    def _fit_single(self, Xs, y):
        if self.name in KERAS_SEQ_MODELS or self.name in KERAS_FLAT_MODELS:
            self._model = _build_keras(self.name, Xs.shape[1:], self.n_classes)
            yc = to_categorical(y, num_classes=self.n_classes)
            self._model.fit(Xs, yc, epochs=self.epochs, batch_size=self.batch,
                            verbose=DL_VERBOSE)
            return self
        if self.name == "TinyML":
            stage1 = _arch_tinyml_stage1(Xs.shape[1:], self.n_classes)
            yc = to_categorical(y, num_classes=self.n_classes)
            stage1.fit(Xs, yc, epochs=self.epochs, batch_size=self.batch,
                       verbose=DL_VERBOSE)
            self._tinyml_extractor = tf.keras.Model(
                inputs=stage1.inputs, outputs=stage1.layers[-2].output)
            feats = self._tinyml_extractor.predict(Xs, verbose=0)
            self._tinyml_rf = _tinyml_rf()
            self._tinyml_rf.fit(feats, y)
            return self
        if self.name in SKLEARN_MODELS:
            self._model = _build_sklearn(self.name)
            self._model.fit(Xs, y)
            return self
        raise ValueError(self.name)

    def _fit_multi(self, Xs, Y):
        n_heads = Y.shape[1]
        if self.name in KERAS_SEQ_MODELS or self.name in KERAS_FLAT_MODELS:
            self._model = _build_keras_multilabel(self.name, Xs.shape[1:], n_heads)
            self._model.fit(Xs, Y, epochs=self.epochs, batch_size=self.batch,
                            verbose=DL_VERBOSE)
            return self
        if self.name == "TinyML":
            stage1_softmax = _arch_tinyml_stage1(Xs.shape[1:], n_heads)
            x = stage1_softmax.layers[-2].output
            out = Dense(n_heads, activation="sigmoid")(x)
            stage1 = Model(inputs=stage1_softmax.inputs, outputs=out)
            stage1.compile(optimizer="adam", loss="binary_crossentropy",
                           metrics=["accuracy"])
            stage1.fit(Xs, Y, epochs=self.epochs, batch_size=self.batch,
                       verbose=DL_VERBOSE)
            self._tinyml_extractor = tf.keras.Model(
                inputs=stage1.inputs, outputs=stage1.layers[-3].output)
            feats = self._tinyml_extractor.predict(Xs, verbose=0)
            self._tinyml_rfs_per_head = []
            for h in range(n_heads):
                rf = RandomForestClassifier(n_estimators=100,
                                             n_jobs=SKLEARN_N_JOBS,
                                             random_state=RANDOM_STATE)
                rf.fit(feats, Y[:, h].astype(np.int64))
                self._tinyml_rfs_per_head.append(rf)
            return self
        if self.name in SKLEARN_MODELS:
            base = _build_sklearn(self.name)
            self._model = MultiOutputClassifier(base, n_jobs=1)
            self._model.fit(Xs, Y.astype(np.int64))
            return self
        raise ValueError(self.name)

    # ---- predict_proba ----
    def predict_proba(self, X):
        Xs = self._shape(X)
        if not self.multilabel:
            return self._predict_single(Xs)
        return self._predict_multi(Xs)

    def _predict_single(self, Xs):
        if self.name in KERAS_SEQ_MODELS or self.name in KERAS_FLAT_MODELS:
            return self._model.predict(Xs, verbose=0)
        if self.name == "TinyML":
            feats = self._tinyml_extractor.predict(Xs, verbose=0)
            return self._tinyml_rf.predict_proba(feats)
        if self.name in SKLEARN_MODELS:
            if hasattr(self._model, "predict_proba"):
                return self._model.predict_proba(Xs)
            scores = self._model.decision_function(Xs)
            if scores.ndim == 1:
                p = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - p, p])
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        raise ValueError(self.name)

    def _predict_multi(self, Xs):
        if self.name in KERAS_SEQ_MODELS or self.name in KERAS_FLAT_MODELS:
            return self._model.predict(Xs, verbose=0)
        if self.name == "TinyML":
            feats = self._tinyml_extractor.predict(Xs, verbose=0)
            cols = [rf.predict_proba(feats)[:, 1] if rf.classes_.size == 2
                    else np.zeros(len(feats))
                    for rf in self._tinyml_rfs_per_head]
            return np.column_stack(cols)
        if self.name in SKLEARN_MODELS:
            ps = self._model.predict_proba(Xs)
            cols = []
            for p in ps:
                if p.shape[1] == 2:
                    cols.append(p[:, 1])
                else:
                    cols.append(p[:, 0])
            return np.column_stack(cols)
        raise ValueError(self.name)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.multilabel:
            return (proba >= 0.5).astype(np.int64)
        return np.argmax(proba, axis=1)
