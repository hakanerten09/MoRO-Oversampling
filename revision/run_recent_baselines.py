# -*- coding: utf-8 -*-
"""
MoRO revision — three-baseline add-on
=====================================

Methods
-------
1) SMOTE-IPF
2) KWSMOTE-2025
3) RE-SMOTE-2024

Goal
----
Run all three baselines in ONE script under the SAME protocol and write
separate add-on result files. Historical MoRO results/figures are NOT modified.

IMPORTANT RE-SMOTE NOTE
-----------------------
The 2024 RE-SMOTE paper explicitly provides an authors' GitHub repository:
    https://github.com/blue9792/RE-SMOTE

However, that repository is research-code style and is not exposed as a clean
pip-installable sklearn/imblearn sampler. Therefore this script uses a
self-contained compatibility implementation of the published RE-SMOTE
generation logic (safe/boundary split + radius-based generation + local
quality filtering) so that it can run on current Python/scikit-learn.

For manuscript wording, describe this as:
    "an independent compatibility implementation based on the published
     algorithm and authors' public repository"
unless you later validate byte-for-byte equivalence with the authors' code.

Paper
-----
D. E, J. Liu, M. Zhang, et al.
"RE-SMOTE: A Novel Imbalanced Sampling Method Based on SMOTE with Radius Estimation"
Computers, Materials & Continua, 81(3), 3853–3880, 2024.
DOI: 10.32604/cmc.2024.057538

Dependencies
------------
pip install numpy pandas scikit-learn openml smote-variants

Run
---
python moro_smoteipf_kwsmote_resmote.py

Outputs
-------
recent3_resmote_raw_folds.csv
recent3_resmote_summary_mean.csv
recent3_resmote_overall.csv
recent3_resmote_applicability.csv
recent3_resmote_run_info.txt
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import openml

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

NUMERIC_DATASETS = [
    ("banknote-authentication", 1462),
    ("diabetes", 37),
    ("ecoli", 40671),
    ("glass", 41),
    ("heart-statlog", 53),
    ("page-blocks", 30),
    ("parkinsons", 1488),
    ("vehicle", 54),
    ("vertebra-column", 1524),
    ("wdbc", 1510),
]

# Same KWSMOTE settings used previously.
KWSMOTE_K = 5
KWSMOTE_L = 3
KWSMOTE_TAU = 0.01

# RE-SMOTE compatibility settings.
RESMOTE_K = 5
RESMOTE_RADIUS_SCALE = 1.0
RESMOTE_MAX_TRIES_FACTOR = 50


# ---------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------
def require_smote_variants():
    try:
        import smote_variants as sv
        return sv
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'smote-variants'. Install with:\n"
            "    pip install smote-variants"
        ) from exc


def safe_gmean(y_true, y_pred, n_classes: int) -> float:
    eps = 1e-12
    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0,
    )
    recalls = np.asarray([max(float(r), eps) for r in recalls], dtype=float)
    if recalls.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def get_scores(model, X_te, n_classes: int):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_te)
        return p[:, 1] if n_classes == 2 else p

    if hasattr(model, "decision_function"):
        return model.decision_function(X_te)

    if n_classes == 2:
        return np.zeros(X_te.shape[0], dtype=float)

    return np.zeros((X_te.shape[0], n_classes), dtype=float)


def safe_avg_precision(y_true, scores, n_classes: int) -> float:
    try:
        if n_classes == 2:
            return float(average_precision_score(y_true, scores))

        y_bin = np.zeros((len(y_true), n_classes), dtype=int)
        y_bin[np.arange(len(y_true)), y_true] = 1

        if np.ndim(scores) != 2 or scores.shape[1] != n_classes:
            return np.nan

        return float(
            average_precision_score(
                y_bin,
                scores,
                average="macro",
            )
        )
    except Exception:
        return np.nan


def load_dataset(openml_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    ds = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )
    return X, y


def build_numeric_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def get_models() -> Dict[str, Any]:
    return {
        "LR": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        ),
        "LinearSVC": LinearSVC(),
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def evaluate_model(model, X_rs, y_rs, X_te, y_te, n_classes):
    model.fit(X_rs, y_rs)
    pred = model.predict(X_te)

    f1m = float(
        f1_score(
            y_te,
            pred,
            average="macro",
            zero_division=0,
        )
    )
    acc = float(accuracy_score(y_te, pred))
    gm = safe_gmean(y_te, pred, n_classes)
    scores = get_scores(model, X_te, n_classes)
    ap = safe_avg_precision(y_te, scores, n_classes)

    return f1m, acc, gm, ap


# ---------------------------------------------------------------------
# KWSMOTE-2025
# ---------------------------------------------------------------------
class KWSMOTE2025:
    def __init__(
        self,
        k: int = 5,
        ell: int = 3,
        tau: float = 0.01,
        sigma: Optional[float] = None,
        random_state: int = 42,
    ):
        self.k = int(k)
        self.ell = int(ell)
        self.tau = float(tau)
        self.sigma = sigma
        self.random_state = int(random_state)

        self.generated_ = 0
        self.sigma_ = np.nan
        self.skipped_ = 0

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("KWSMOTE-2025 is restricted to binary datasets.")

        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]

        X_min = X[y == minority]
        n_min = int(np.sum(y == minority))
        n_maj = int(np.sum(y == majority))
        n_generate = n_maj - n_min

        if n_generate <= 0:
            return X.copy(), y.copy()

        if len(X_min) < 2:
            raise ValueError("Too few minority samples for KWSMOTE.")

        k_eff = min(self.k, len(X_min) - 1)
        ell_eff = min(self.ell, k_eff)

        nn = NearestNeighbors(n_neighbors=k_eff + 1)
        nn.fit(X_min)
        neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]

        if self.sigma is None:
            var_scalar = float(np.mean(np.var(X_min, axis=0)))
            sigma = np.sqrt(
                max(var_scalar, 1e-12)
                * X_min.shape[1]
                / 2.0
            )
        else:
            sigma = float(self.sigma)

        sigma = max(float(sigma), 1e-12)
        self.sigma_ = sigma

        rng = np.random.RandomState(self.random_state)
        synth = []

        max_tries = max(n_generate * 100, 1000)
        tries = 0

        while len(synth) < n_generate and tries < max_tries:
            tries += 1

            i = int(rng.randint(0, len(X_min)))
            xi = X_min[i]
            cand = neigh[i]

            if len(cand) > ell_eff:
                chosen = rng.choice(cand, size=ell_eff, replace=False)
            else:
                chosen = cand

            Xn = X_min[np.asarray(chosen, dtype=int)]
            sq_dist = np.sum((Xn - xi) ** 2, axis=1)
            w = np.exp(-sq_dist / (2.0 * sigma ** 2))

            if len(w) == 0 or float(np.max(w)) < self.tau:
                self.skipped_ += 1
                continue

            x_new = (
                xi + np.sum(w[:, None] * Xn, axis=0)
            ) / (
                1.0 + np.sum(w)
            )

            synth.append(x_new)

        if len(synth) != n_generate:
            raise RuntimeError(
                "KWSMOTE-2025 could not generate the required sample count."
            )

        X_syn = np.vstack(synth)
        y_syn = np.full(len(synth), minority, dtype=y.dtype)

        self.generated_ = len(synth)

        return (
            np.vstack([X, X_syn]),
            np.concatenate([y, y_syn]),
        )


# ---------------------------------------------------------------------
# RE-SMOTE-2024 compatibility implementation
# ---------------------------------------------------------------------
class RESMOTE2024Compat:
    """
    Current-Python compatibility implementation based on the published
    RE-SMOTE idea and authors' public repository.

    Main steps:
      1) split minority samples into SAFE and BOUNDARY using local neighbors;
      2) SAFE: synthesize inside a radius determined by nearest majority;
      3) BOUNDARY: linear interpolation toward nearest majority;
      4) candidate-quality check with local inverse-distance weighted voting;
      5) continue until exact class balance is reached.

    No silent fallback to SMOTE/None is used.
    """

    def __init__(
        self,
        k: int = 5,
        radius_scale: float = 1.0,
        random_state: int = 42,
        max_tries_factor: int = 50,
    ):
        self.k = int(k)
        self.radius_scale = float(radius_scale)
        self.random_state = int(random_state)
        self.max_tries_factor = int(max_tries_factor)

        self.generated_ = 0
        self.safe_count_ = 0
        self.boundary_count_ = 0
        self.rejected_ = 0
        self.acceptance_rate_ = np.nan

    def _local_type(
        self,
        X,
        y,
        X_min,
        minority,
        k_eff,
    ):
        n_neighbors = min(k_eff + 1, len(X))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)

        indices = nn.kneighbors(
            X_min,
            return_distance=False,
        )

        safe_mask = np.zeros(len(X_min), dtype=bool)

        for i, row in enumerate(indices):
            neighbor_labels = []

            for idx in row:
                # remove self or an exact duplicated same-label point once
                if (
                    y[idx] == minority
                    and np.allclose(X[idx], X_min[i])
                    and len(neighbor_labels) == 0
                ):
                    continue

                neighbor_labels.append(y[idx])

                if len(neighbor_labels) == k_eff:
                    break

            if not neighbor_labels:
                safe_mask[i] = True
            else:
                maj_n = np.sum(
                    np.asarray(neighbor_labels) != minority
                )
                # no local majority -> safe; otherwise boundary
                safe_mask[i] = (maj_n == 0)

        return safe_mask

    def _nearest_majority(self, X_min, X_maj):
        nn_maj = NearestNeighbors(n_neighbors=1).fit(X_maj)
        d, idx = nn_maj.kneighbors(
            X_min,
            return_distance=True,
        )
        return d[:, 0], idx[:, 0]

    def _weighted_local_accept(
        self,
        candidate,
        X_reference,
        y_reference,
        minority,
        k_eff,
    ):
        """
        WENN-inspired local editing check:
        accept candidate when inverse-distance weighted kNN vote
        supports minority class.

        This intentionally avoids scoring a failed generation as a valid
        result and acts only as a quality filter on synthetic candidates.
        """
        k_use = min(k_eff, len(X_reference))
        nn = NearestNeighbors(n_neighbors=k_use).fit(X_reference)

        d, idx = nn.kneighbors(
            candidate.reshape(1, -1),
            return_distance=True,
        )

        d = d[0]
        idx = idx[0]

        w = 1.0 / np.maximum(d, 1e-12)
        labels = y_reference[idx]

        minority_weight = float(np.sum(w[labels == minority]))
        majority_weight = float(np.sum(w[labels != minority]))

        return minority_weight >= majority_weight

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)

        if len(classes) != 2:
            raise ValueError(
                "RE-SMOTE-2024 comparison is restricted to binary datasets."
            )

        minority = classes[np.argmin(counts)]
        majority = classes[np.argmax(counts)]

        X_min = X[y == minority]
        X_maj = X[y == majority]

        n_generate = len(X_maj) - len(X_min)

        if n_generate <= 0:
            return X.copy(), y.copy()

        if len(X_min) < 2:
            raise ValueError("Too few minority samples for RE-SMOTE.")

        k_eff = min(self.k, len(X_min) - 1)

        safe_mask = self._local_type(
            X,
            y,
            X_min,
            minority,
            k_eff,
        )

        boundary_mask = ~safe_mask

        # Degenerate guard: keep both generation modes available if one
        # group is empty, but do not switch to another oversampler.
        safe_idx = np.where(safe_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]

        self.safe_count_ = len(safe_idx)
        self.boundary_count_ = len(boundary_idx)

        nearest_maj_dist, nearest_maj_idx = self._nearest_majority(
            X_min,
            X_maj,
        )

        rng = np.random.RandomState(self.random_state)
        synth = []

        X_ref = X.copy()
        y_ref = y.copy()

        max_tries = max(
            n_generate * self.max_tries_factor,
            1000,
        )

        tries = 0
        rejected = 0

        while len(synth) < n_generate and tries < max_tries:
            tries += 1

            # Choose a minority seed uniformly.
            i = int(rng.randint(0, len(X_min)))
            xi = X_min[i]

            if safe_mask[i]:
                # Radius determined by nearest majority distance.
                radius = (
                    self.radius_scale
                    * max(float(nearest_maj_dist[i]), 1e-12)
                )

                # Uniform random direction.
                direction = rng.normal(size=X.shape[1])
                norm = float(np.linalg.norm(direction))
                if norm <= 1e-12:
                    rejected += 1
                    continue
                direction /= norm

                # Uniform-in-ball radial scaling.
                radial = float(
                    rng.random() ** (1.0 / max(X.shape[1], 1))
                )
                candidate = xi + direction * radius * radial

            else:
                # Published boundary generation: interpolate minority seed
                # with its nearest majority point.
                xmaj = X_maj[int(nearest_maj_idx[i])]
                lam = float(rng.random())
                candidate = xi + lam * (xmaj - xi)

            if not np.all(np.isfinite(candidate)):
                rejected += 1
                continue

            if not self._weighted_local_accept(
                candidate,
                X_ref,
                y_ref,
                minority,
                k_eff,
            ):
                rejected += 1
                continue

            synth.append(candidate)

            # Accepted candidate joins local reference set for subsequent
            # editing checks.
            X_ref = np.vstack([X_ref, candidate])
            y_ref = np.concatenate([
                y_ref,
                np.asarray([minority], dtype=y.dtype),
            ])

        if len(synth) != n_generate:
            raise RuntimeError(
                f"RE-SMOTE generated {len(synth)} of {n_generate} required "
                f"samples after {tries} attempts. No fallback was used."
            )

        X_syn = np.vstack(synth)
        y_syn = np.full(len(synth), minority, dtype=y.dtype)

        self.generated_ = len(synth)
        self.rejected_ = rejected
        self.acceptance_rate_ = (
            len(synth) / tries if tries > 0 else np.nan
        )

        return (
            np.vstack([X, X_syn]),
            np.concatenate([y, y_syn]),
        )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------
def main():
    sv = require_smote_variants()
    sv_version = getattr(sv, "__version__", "unknown")

    print("=" * 90)
    print("MoRO REVISION — THREE BASELINE ADD-ON")
    print("SMOTE-IPF + KWSMOTE-2025 + RE-SMOTE-2024")
    print("Historical MoRO result files/figures will NOT be modified.")
    print("=" * 90)

    methods = [
        "SMOTE-IPF",
        "KWSMOTE-2025",
        "RE-SMOTE-2024",
    ]

    rows: List[Dict[str, Any]] = []
    applicability: List[Dict[str, Any]] = []

    for ds_i, (dataset_name, openml_id) in enumerate(
        NUMERIC_DATASETS,
        start=1,
    ):
        print(
            f"\n[{ds_i}/{len(NUMERIC_DATASETS)}] "
            f"{dataset_name} (OpenML {openml_id})"
        )

        X_df, y_series = load_dataset(openml_id)

        non_numeric = [
            c for c in X_df.columns
            if not pd.api.types.is_numeric_dtype(X_df[c])
        ]

        if non_numeric:
            raise ValueError(
                f"{dataset_name} has non-numeric columns: {non_numeric}"
            )

        le = LabelEncoder()
        y = le.fit_transform(y_series.to_numpy())

        n_classes = len(np.unique(y))
        is_binary = (n_classes == 2)

        for method in methods:
            applicability.append({
                "dataset": dataset_name,
                "openml_id": openml_id,
                "n_classes": n_classes,
                "method": method,
                "applicable": int(is_binary),
                "reason": (
                    "common binary-numeric comparison subset"
                    if is_binary
                    else
                    "excluded to preserve a common binary-numeric comparison"
                ),
            })

        if not is_binary:
            print(f"  skipped: {n_classes} classes")
            continue

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for fold, (tr_idx, te_idx) in enumerate(
            skf.split(X_df, y),
            start=1,
        ):
            X_tr_df = X_df.iloc[tr_idx]
            X_te_df = X_df.iloc[te_idx]
            y_tr = y[tr_idx]
            y_te = y[te_idx]

            pre = build_numeric_preprocessor()
            X_tr = pre.fit_transform(X_tr_df)
            X_te = pre.transform(X_te_df)

            sampled_sets: Dict[str, Dict[str, Any]] = {}

            # ----------------------------------------------------------
            # 1) SMOTE-IPF
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                sampler = sv.SMOTE_IPF(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )

                X_rs, y_rs = sampler.sample(
                    np.asarray(X_tr, dtype=float),
                    np.asarray(y_tr),
                )

                sampled_sets["SMOTE-IPF"] = {
                    "X": np.asarray(X_rs, dtype=float),
                    "y": np.asarray(y_rs),
                    "ok": 1,
                    "error": "",
                    "time": time.time() - t0,
                    "detail1": np.nan,
                    "detail2": np.nan,
                    "detail3": np.nan,
                }

            except Exception as exc:
                sampled_sets["SMOTE-IPF"] = {
                    "X": None,
                    "y": None,
                    "ok": 0,
                    "error": repr(exc),
                    "time": time.time() - t0,
                    "detail1": np.nan,
                    "detail2": np.nan,
                    "detail3": np.nan,
                }

            # ----------------------------------------------------------
            # 2) KWSMOTE-2025
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                sampler = KWSMOTE2025(
                    k=KWSMOTE_K,
                    ell=KWSMOTE_L,
                    tau=KWSMOTE_TAU,
                    random_state=RANDOM_STATE,
                )

                X_rs, y_rs = sampler.fit_resample(
                    X_tr,
                    y_tr,
                )

                sampled_sets["KWSMOTE-2025"] = {
                    "X": X_rs,
                    "y": y_rs,
                    "ok": 1,
                    "error": "",
                    "time": time.time() - t0,
                    "detail1": sampler.sigma_,
                    "detail2": sampler.skipped_,
                    "detail3": np.nan,
                }

            except Exception as exc:
                sampled_sets["KWSMOTE-2025"] = {
                    "X": None,
                    "y": None,
                    "ok": 0,
                    "error": repr(exc),
                    "time": time.time() - t0,
                    "detail1": np.nan,
                    "detail2": np.nan,
                    "detail3": np.nan,
                }

            # ----------------------------------------------------------
            # 3) RE-SMOTE-2024
            # ----------------------------------------------------------
            t0 = time.time()
            try:
                sampler = RESMOTE2024Compat(
                    k=RESMOTE_K,
                    radius_scale=RESMOTE_RADIUS_SCALE,
                    random_state=RANDOM_STATE,
                    max_tries_factor=RESMOTE_MAX_TRIES_FACTOR,
                )

                X_rs, y_rs = sampler.fit_resample(
                    X_tr,
                    y_tr,
                )

                sampled_sets["RE-SMOTE-2024"] = {
                    "X": X_rs,
                    "y": y_rs,
                    "ok": 1,
                    "error": "",
                    "time": time.time() - t0,
                    "detail1": sampler.safe_count_,
                    "detail2": sampler.boundary_count_,
                    "detail3": sampler.acceptance_rate_,
                }

            except Exception as exc:
                sampled_sets["RE-SMOTE-2024"] = {
                    "X": None,
                    "y": None,
                    "ok": 0,
                    "error": repr(exc),
                    "time": time.time() - t0,
                    "detail1": np.nan,
                    "detail2": np.nan,
                    "detail3": np.nan,
                }

            # ----------------------------------------------------------
            # Models
            # ----------------------------------------------------------
            for method_name, info in sampled_sets.items():
                for model_name, model in get_models().items():

                    if info["ok"] != 1:
                        # Critical safety rule:
                        # failed sampling is NEVER scored using unsampled data.
                        f1m = acc = gm = ap = np.nan
                        model_ok = 0
                        model_error = (
                            "Model not evaluated because sampling failed."
                        )
                        before_n = len(y_tr)
                        after_n = np.nan
                        generated_n = np.nan

                    else:
                        model_ok = 1
                        model_error = ""
                        before_n = len(y_tr)
                        after_n = len(info["y"])
                        generated_n = after_n - before_n

                        try:
                            f1m, acc, gm, ap = evaluate_model(
                                model,
                                info["X"],
                                info["y"],
                                X_te,
                                y_te,
                                n_classes,
                            )
                        except Exception as exc:
                            model_ok = 0
                            model_error = repr(exc)
                            f1m = acc = gm = ap = np.nan

                    rows.append({
                        "dataset": dataset_name,
                        "openml_id": openml_id,
                        "detected_type": "numeric",
                        "n_classes": n_classes,
                        "fold": fold,
                        "sampler": method_name,
                        "classifier": model_name,
                        "f1_macro": f1m,
                        "accuracy": acc,
                        "gmean": gm,
                        "avg_precision": ap,
                        "BeforeN": before_n,
                        "AfterN": after_n,
                        "GeneratedN": generated_n,
                        "SamplingTime": info["time"],
                        "SamplingOK": info["ok"],
                        "ModelOK": model_ok,
                        "SamplingError": info["error"],
                        "ModelError": model_error,
                        "Detail1": info["detail1"],
                        "Detail2": info["detail2"],
                        "Detail3": info["detail3"],
                    })

            print(f"  fold {fold}/{N_SPLITS} completed")

    raw = pd.DataFrame(rows)
    raw.to_csv(
        "recent3_resmote_raw_folds.csv",
        index=False,
    )

    app = pd.DataFrame(applicability)
    app.to_csv(
        "recent3_resmote_applicability.csv",
        index=False,
    )

    if raw.empty:
        raise RuntimeError("No valid experiment rows were generated.")

    summary = (
        raw.groupby(
            [
                "dataset",
                "openml_id",
                "sampler",
                "classifier",
            ],
            as_index=False,
        )
        .agg(
            f1_macro=("f1_macro", "mean"),
            accuracy=("accuracy", "mean"),
            gmean=("gmean", "mean"),
            avg_precision=("avg_precision", "mean"),
            MeanGeneratedN=("GeneratedN", "mean"),
            MeanSamplingTime=("SamplingTime", "mean"),
            SamplingOKRate=("SamplingOK", "mean"),
            ModelOKRate=("ModelOK", "mean"),
            ValidFolds=("f1_macro", "count"),
        )
    )

    summary.to_csv(
        "recent3_resmote_summary_mean.csv",
        index=False,
    )

    overall = (
        summary.groupby(
            "sampler",
            as_index=False,
        )
        .agg(
            MeanF1=("f1_macro", "mean"),
            MeanAccuracy=("accuracy", "mean"),
            MeanGMean=("gmean", "mean"),
            MeanAvgPrecision=("avg_precision", "mean"),
            MeanSamplingTime=("MeanSamplingTime", "mean"),
            SamplingOKRate=("SamplingOKRate", "mean"),
            ModelOKRate=("ModelOKRate", "mean"),
            N=("f1_macro", "count"),
        )
        .sort_values(
            "MeanF1",
            ascending=False,
        )
    )

    overall.to_csv(
        "recent3_resmote_overall.csv",
        index=False,
    )

    info = f"""MoRO revision — SMOTE-IPF + KWSMOTE-2025 + RE-SMOTE-2024

Historical MoRO results/figures were NOT modified.

Common protocol
===============
- common binary numeric subset only
- Stratified {N_SPLITS}-fold CV
- random_state={RANDOM_STATE}
- preprocessing fitted only on training fold
- median imputation + StandardScaler
- LR / LinearSVC / RF
- F1-macro / Accuracy / G-mean / Average Precision

Methods
=======
1) SMOTE-IPF
   implementation: smote-variants
   package version: {sv_version}
   random_state={RANDOM_STATE}
   n_jobs=1
   other settings: package defaults

2) KWSMOTE-2025
   k={KWSMOTE_K}
   ell={KWSMOTE_L}
   tau={KWSMOTE_TAU}
   sigma=automatic formula
   random_state={RANDOM_STATE}

3) RE-SMOTE-2024
   paper:
   RE-SMOTE: A Novel Imbalanced Sampling Method Based on SMOTE
   with Radius Estimation.
   Computers, Materials & Continua 81(3), 3853-3880, 2024.
   DOI: 10.32604/cmc.2024.057538

   authors' public repository:
   https://github.com/blue9792/RE-SMOTE

   experiment implementation:
   independent current-Python compatibility implementation based on the
   published algorithm/repository; NOT claimed to be byte-for-byte execution
   of the original research scripts.

   k={RESMOTE_K}
   radius_scale={RESMOTE_RADIUS_SCALE}
   random_state={RANDOM_STATE}

Failure handling
================
If a sampler fails on a fold, classifier metrics are NaN.
The original unsampled training set is NEVER evaluated under the failed
sampler's name.

RE-SMOTE diagnostic columns in raw CSV
======================================
Detail1 = number of safe minority samples in training fold
Detail2 = number of boundary minority samples in training fold
Detail3 = synthetic-candidate acceptance rate

Outputs
=======
- recent3_resmote_raw_folds.csv
- recent3_resmote_summary_mean.csv
- recent3_resmote_overall.csv
- recent3_resmote_applicability.csv
- recent3_resmote_run_info.txt
"""

    Path(
        "recent3_resmote_run_info.txt"
    ).write_text(
        info,
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("Created:")
    print("- recent3_resmote_raw_folds.csv")
    print("- recent3_resmote_summary_mean.csv")
    print("- recent3_resmote_overall.csv")
    print("- recent3_resmote_applicability.csv")
    print("- recent3_resmote_run_info.txt")
    print("=" * 90)


if __name__ == "__main__":
    main()
