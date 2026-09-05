# -*- coding: utf-8 -*-
"""
MoRO-G one-at-a-time sensitivity analysis for revision.

IMPORTANT
---------
- This is a SEPARATE experiment script.
- It does NOT overwrite any existing main-result CSV/PNG/PDF.
- It evaluates ONLY MoRO-G on NUMERIC datasets.
- The published/default configuration is preserved:
      k_neighbors = 5
      k_borderline = 15
      tau_majority = 0.30
      alpha = 6.0
      epsilon = 1e-12
      mr_exponent = 2.0
      max_attempts_per_sample = 12
      p_floor = 0.15
      p_ceil = 0.98
      fallback_accept = True
      random_state = 42

One-at-a-time grids:
- MR exponent q: 1, 2, 3
- alpha: 2, 4, 6, 8, 10
- tau_majority: 0.20, 0.30, 0.40, 0.50
- epsilon: 1e-14, 1e-12, 1e-10, 1e-8

Outputs (new files only):
- sensitivity_raw_folds.csv
- sensitivity_summary.csv
- sensitivity_by_parameter.csv
- sensitivity_default_check.csv
- sensitivity_gate_diagnostics.csv
- sensitivity_run_info.txt

Run:
    python run_sensitivity_analysis.py

Run from the repository root:
    python revision/run_sensitivity_analysis.py

"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import openml

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


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

DEFAULTS = {
    "k_neighbors": 5,
    "k_borderline": 15,
    "tau_majority": 0.30,
    "alpha": 6.0,
    "epsilon": 1e-12,
    "mr_exponent": 2.0,
    "max_majority_for_gate": 2000,
    "max_attempts_per_sample": 12,
    "p_floor": 0.15,
    "p_ceil": 0.98,
    "fallback_accept": True,
}

SENSITIVITY_GRIDS = {
    "mr_exponent": [1.0, 2.0, 3.0],
    "alpha": [2.0, 4.0, 6.0, 8.0, 10.0],
    "tau_majority": [0.20, 0.30, 0.40, 0.50],
    "epsilon": [1e-14, 1e-12, 1e-10, 1e-8],
}

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def safe_gmean(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    eps = 1e-12
    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0
    )
    recalls = np.asarray([max(float(r), eps) for r in recalls], dtype=float)
    if recalls.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def get_scores(model, X_te: np.ndarray, n_classes: int) -> np.ndarray:
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
        return float(average_precision_score(y_bin, scores, average="macro"))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_dataset(openml_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    ds = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute
    )
    return X, y


def build_numeric_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------
# MoRO-G
# ---------------------------------------------------------------------
@dataclass
class MoRODiagnostics:
    candidate_attempts: int
    accepted_samples: int
    gated_attempts: int
    rejected_gated_attempts: int
    retry_samples: int
    fallback_accepts: int
    mean_mr_star: float
    mean_accept_prob: float

    @property
    def gate_active_rate(self) -> float:
        return (
            self.gated_attempts / self.candidate_attempts
            if self.candidate_attempts else 0.0
        )

    @property
    def gated_rejection_rate(self) -> float:
        return (
            self.rejected_gated_attempts / self.gated_attempts
            if self.gated_attempts else 0.0
        )

    @property
    def attempts_per_accepted_sample(self) -> float:
        return (
            self.candidate_attempts / self.accepted_samples
            if self.accepted_samples else np.nan
        )

    @property
    def retry_sample_rate(self) -> float:
        return (
            self.retry_samples / self.accepted_samples
            if self.accepted_samples else 0.0
        )

    @property
    def fallback_accept_rate(self) -> float:
        return (
            self.fallback_accepts / self.accepted_samples
            if self.accepted_samples else 0.0
        )


def sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


class MoROGSampler:
    def __init__(
        self,
        k_neighbors: int = 5,
        k_borderline: int = 15,
        tau_majority: float = 0.30,
        alpha: float = 6.0,
        epsilon: float = 1e-12,
        mr_exponent: float = 2.0,
        max_majority_for_gate: int = 2000,
        random_state: int = 42,
        max_attempts_per_sample: int = 12,
        p_floor: float = 0.15,
        p_ceil: float = 0.98,
        fallback_accept: bool = True,
    ):
        self.k_neighbors = int(k_neighbors)
        self.k_borderline = int(k_borderline)
        self.tau_majority = float(tau_majority)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.mr_exponent = float(mr_exponent)
        self.max_majority_for_gate = int(max_majority_for_gate)
        self.random_state = int(random_state)
        self.max_attempts_per_sample = int(max_attempts_per_sample)
        self.p_floor = float(p_floor)
        self.p_ceil = float(p_ceil)
        self.fallback_accept = bool(fallback_accept)

        self.diag_: Optional[MoRODiagnostics] = None

    def _borderline_flag(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cls: int,
        x_anchor: np.ndarray,
        nn_global: NearestNeighbors
    ) -> bool:
        k_use = min(self.k_borderline, X.shape[0])
        idx = nn_global.kneighbors(
            x_anchor.reshape(1, -1),
            n_neighbors=k_use,
            return_distance=False
        ).ravel()

        labels = y[idx]
        majority_ratio = float(np.mean(labels != cls))
        return majority_ratio >= self.tau_majority

    def _dm_do(
        self,
        Xm: np.ndarray,
        Xo: np.ndarray,
        x: np.ndarray
    ) -> Tuple[float, float]:
        if Xm.shape[0] < 2 or Xo.shape[0] < 2:
            return 0.0, 0.0

        dm = np.linalg.norm(Xm - x[None, :], axis=1)
        do = np.linalg.norm(Xo - x[None, :], axis=1)

        km = min(self.k_neighbors, dm.shape[0])
        ko = min(self.k_neighbors, do.shape[0])

        # Keep the historical numerical behaviour at the default epsilon.
        dm_mean = float(np.mean(np.partition(dm, km - 1)[:km])) + self.epsilon
        do_mean = float(np.mean(np.partition(do, ko - 1)[:ko])) + self.epsilon

        return dm_mean, do_mean

    def _mr_star(self, dm_mean: float, do_mean: float) -> float:
        if dm_mean <= 0:
            return 0.0
        ratio = do_mean / dm_mean
        return float(ratio ** self.mr_exponent)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y

        max_count = int(np.max(counts))

        nn_global = NearestNeighbors(
            n_neighbors=min(self.k_borderline, X.shape[0]),
            metric="euclidean",
            algorithm="auto",
        )
        nn_global.fit(X)

        X_new = [X]
        y_new = [y]

        candidate_attempts = 0
        accepted_samples = 0
        gated_attempts = 0
        rejected_gated_attempts = 0
        retry_samples = 0
        fallback_accepts = 0
        mr_values: List[float] = []
        p_values: List[float] = []

        for cls in classes:
            Xc = X[y == cls]
            n_c = Xc.shape[0]

            if n_c < 2:
                continue

            n_to_add = max_count - n_c
            if n_to_add <= 0:
                continue

            nn_min = NearestNeighbors(
                n_neighbors=min(self.k_neighbors + 1, n_c),
                metric="euclidean",
                algorithm="auto",
            )
            nn_min.fit(Xc)

            Xo_full = X[y != cls]
            if Xo_full.shape[0] > self.max_majority_for_gate:
                idx = rng.choice(
                    Xo_full.shape[0],
                    self.max_majority_for_gate,
                    replace=False
                )
                Xo = Xo_full[idx]
            else:
                Xo = Xo_full

            synth = []

            for _ in range(n_to_add):
                got_one = False
                last_candidate = None
                rejected_once = False

                for _attempt in range(self.max_attempts_per_sample):
                    i = rng.randint(0, n_c)
                    x_i = Xc[i]

                    neigh_idx = nn_min.kneighbors(
                        x_i.reshape(1, -1),
                        return_distance=False
                    ).ravel()
                    neigh_idx = neigh_idx[neigh_idx != i]

                    if neigh_idx.size == 0:
                        continue

                    j = int(rng.choice(neigh_idx))
                    x_j = Xc[j]

                    lam = rng.rand()
                    x_cand = x_i + lam * (x_j - x_i)

                    last_candidate = x_cand
                    candidate_attempts += 1

                    is_borderline = self._borderline_flag(
                        X, y, cls, x_i, nn_global
                    )

                    if not is_borderline:
                        synth.append(x_cand)
                        accepted_samples += 1
                        got_one = True
                        break

                    gated_attempts += 1

                    dm_mean, do_mean = self._dm_do(
                        Xc, Xo, x_cand
                    )
                    mr_star = self._mr_star(
                        dm_mean, do_mean
                    )

                    p_acc = sigmoid(
                        self.alpha * (mr_star - 1.0)
                    )
                    p_acc = float(
                        np.clip(
                            p_acc,
                            self.p_floor,
                            self.p_ceil
                        )
                    )

                    mr_values.append(mr_star)
                    p_values.append(p_acc)

                    if rng.rand() < p_acc:
                        synth.append(x_cand)
                        accepted_samples += 1
                        got_one = True
                        break

                    rejected_gated_attempts += 1
                    rejected_once = True

                if rejected_once:
                    retry_samples += 1

                if (
                    not got_one
                    and last_candidate is not None
                    and self.fallback_accept
                ):
                    synth.append(last_candidate)
                    accepted_samples += 1
                    fallback_accepts += 1

            if synth:
                X_new.append(np.vstack(synth))
                y_new.append(
                    np.full(
                        len(synth),
                        cls,
                        dtype=y.dtype
                    )
                )

        self.diag_ = MoRODiagnostics(
            candidate_attempts=candidate_attempts,
            accepted_samples=accepted_samples,
            gated_attempts=gated_attempts,
            rejected_gated_attempts=rejected_gated_attempts,
            retry_samples=retry_samples,
            fallback_accepts=fallback_accepts,
            mean_mr_star=(
                float(np.mean(mr_values))
                if mr_values else 0.0
            ),
            mean_accept_prob=(
                float(np.mean(p_values))
                if p_values else 1.0
            ),
        )

        return np.vstack(X_new), np.concatenate(y_new)


def get_models() -> Dict[str, Any]:
    return {
        "LR": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="auto",
        ),
        "LinearSVC": LinearSVC(),
        "RF": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------
def build_configs() -> List[Dict[str, Any]]:
    configs = []

    # One-at-a-time: each experiment changes exactly one parameter.
    for parameter, values in SENSITIVITY_GRIDS.items():
        for value in values:
            cfg = dict(DEFAULTS)
            cfg[parameter] = value

            configs.append({
                "parameter": parameter,
                "value": value,
                "is_default_value": bool(
                    np.isclose(
                        float(value),
                        float(DEFAULTS[parameter]),
                        rtol=0.0,
                        atol=0.0,
                    )
                ),
                "config": cfg,
            })

    return configs


def main() -> None:
    configs = build_configs()
    total_configs = len(configs)

    print("=" * 78)
    print("MoRO-G NUMERIC SENSITIVITY ANALYSIS")
    print("Existing main result files are NOT modified.")
    print(f"Datasets: {len(NUMERIC_DATASETS)}")
    print(f"Configurations: {total_configs}")
    print(f"CV: {N_SPLITS}-fold")
    print("=" * 78)

    raw_rows = []
    diagnostic_rows = []

    for ds_idx, (dataset_name, openml_id) in enumerate(
        NUMERIC_DATASETS, start=1
    ):
        print(
            f"\n[{ds_idx}/{len(NUMERIC_DATASETS)}] "
            f"{dataset_name} (OpenML {openml_id})"
        )

        X_df, y_series = load_dataset(openml_id)

        # Numeric sensitivity must truly be numeric.
        non_numeric = [
            c for c in X_df.columns
            if not pd.api.types.is_numeric_dtype(X_df[c])
        ]
        if non_numeric:
            raise ValueError(
                f"{dataset_name} contains non-numeric columns: {non_numeric}"
            )

        le = LabelEncoder()
        y = le.fit_transform(y_series.to_numpy())
        n_classes = len(np.unique(y))

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for fold, (tr_idx, te_idx) in enumerate(
            skf.split(X_df, y), start=1
        ):
            X_tr_df = X_df.iloc[tr_idx]
            X_te_df = X_df.iloc[te_idx]
            y_tr = y[tr_idx]
            y_te = y[te_idx]

            pre = build_numeric_preprocessor()
            X_tr = pre.fit_transform(X_tr_df)
            X_te = pre.transform(X_te_df)

            for cfg_idx, item in enumerate(configs, start=1):
                parameter = item["parameter"]
                value = item["value"]
                cfg = item["config"]

                t0 = time.time()

                sampler = MoROGSampler(
                    k_neighbors=cfg["k_neighbors"],
                    k_borderline=cfg["k_borderline"],
                    tau_majority=cfg["tau_majority"],
                    alpha=cfg["alpha"],
                    epsilon=cfg["epsilon"],
                    mr_exponent=cfg["mr_exponent"],
                    max_majority_for_gate=cfg["max_majority_for_gate"],
                    random_state=RANDOM_STATE,
                    max_attempts_per_sample=cfg["max_attempts_per_sample"],
                    p_floor=cfg["p_floor"],
                    p_ceil=cfg["p_ceil"],
                    fallback_accept=cfg["fallback_accept"],
                )

                X_rs, y_rs = sampler.fit_resample(X_tr, y_tr)
                sampling_time = time.time() - t0

                d = sampler.diag_

                diagnostic_rows.append({
                    "dataset": dataset_name,
                    "openml_id": openml_id,
                    "fold": fold,
                    "parameter": parameter,
                    "value": value,
                    "is_default_value": item["is_default_value"],
                    "GateActiveRate": (
                        d.gate_active_rate if d else np.nan
                    ),
                    "GatedRejectionRate": (
                        d.gated_rejection_rate if d else np.nan
                    ),
                    "RetrySampleRate": (
                        d.retry_sample_rate if d else np.nan
                    ),
                    "FallbackAcceptRate": (
                        d.fallback_accept_rate if d else np.nan
                    ),
                    "AttemptsPerAcceptedSample": (
                        d.attempts_per_accepted_sample
                        if d else np.nan
                    ),
                    "MeanMRStar": (
                        d.mean_mr_star if d else np.nan
                    ),
                    "MeanAcceptProb": (
                        d.mean_accept_prob if d else np.nan
                    ),
                    "SamplingTime": sampling_time,
                })

                for model_name, model in get_models().items():
                    model_t0 = time.time()
                    model.fit(X_rs, y_rs)

                    y_pred = model.predict(X_te)

                    f1m = float(
                        f1_score(
                            y_te,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    acc = float(
                        accuracy_score(y_te, y_pred)
                    )
                    gm = float(
                        safe_gmean(
                            y_te,
                            y_pred,
                            n_classes,
                        )
                    )

                    scores = get_scores(
                        model,
                        X_te,
                        n_classes,
                    )
                    ap = float(
                        safe_avg_precision(
                            y_te,
                            scores,
                            n_classes,
                        )
                    )

                    raw_rows.append({
                        "dataset": dataset_name,
                        "openml_id": openml_id,
                        "fold": fold,
                        "classifier": model_name,
                        "parameter": parameter,
                        "value": value,
                        "is_default_value": item["is_default_value"],
                        "f1_macro": f1m,
                        "accuracy": acc,
                        "gmean": gm,
                        "avg_precision": ap,
                        "SamplingTime": sampling_time,
                        "ModelTime": time.time() - model_t0,
                    })

            print(f"  fold {fold}/{N_SPLITS} completed")

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(
        "sensitivity_raw_folds.csv",
        index=False,
    )

    diag = pd.DataFrame(diagnostic_rows)
    diag.to_csv(
        "sensitivity_gate_diagnostics.csv",
        index=False,
    )

    # Each parameter/value across all datasets/classifiers/folds
    summary = (
        raw.groupby(
            ["parameter", "value"],
            as_index=False,
        )
        .agg(
            MeanF1=("f1_macro", "mean"),
            StdF1=("f1_macro", "std"),
            MedianF1=("f1_macro", "median"),
            MeanAccuracy=("accuracy", "mean"),
            MeanGMean=("gmean", "mean"),
            MeanAvgPrecision=("avg_precision", "mean"),
            N=("f1_macro", "count"),
        )
    )

    # Add delta relative to DEFAULT value within each sensitivity family.
    summary_parts = []

    for parameter, grp in summary.groupby(
        "parameter",
        sort=False
    ):
        grp = grp.copy()
        default_value = float(DEFAULTS[parameter])

        default_rows = grp[
            np.isclose(
                grp["value"].astype(float),
                default_value,
                rtol=0.0,
                atol=0.0,
            )
        ]

        if len(default_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one default row for {parameter}"
            )

        default_f1 = float(
            default_rows.iloc[0]["MeanF1"]
        )
        default_gm = float(
            default_rows.iloc[0]["MeanGMean"]
        )
        default_ap = float(
            default_rows.iloc[0]["MeanAvgPrecision"]
        )

        grp["DefaultValue"] = default_value
        grp["DeltaF1_vs_Default"] = (
            grp["MeanF1"] - default_f1
        )
        grp["DeltaGMean_vs_Default"] = (
            grp["MeanGMean"] - default_gm
        )
        grp["DeltaAvgPrecision_vs_Default"] = (
            grp["MeanAvgPrecision"] - default_ap
        )

        summary_parts.append(grp)

    summary = pd.concat(
        summary_parts,
        ignore_index=True,
    )

    summary.to_csv(
        "sensitivity_summary.csv",
        index=False,
    )
    summary.to_csv(
        "sensitivity_by_parameter.csv",
        index=False,
    )

    # The default setting occurs once inside EACH one-at-a-time family.
    # Since all four should be numerically identical at the same seed,
    # this file checks that explicitly.
    defaults = raw[
        raw["is_default_value"]
    ].copy()

    default_check = (
        defaults.groupby(
            ["parameter"],
            as_index=False,
        )
        .agg(
            MeanF1=("f1_macro", "mean"),
            MeanAccuracy=("accuracy", "mean"),
            MeanGMean=("gmean", "mean"),
            MeanAvgPrecision=("avg_precision", "mean"),
            N=("f1_macro", "count"),
        )
    )

    default_check["F1DifferenceFromFirst"] = (
        default_check["MeanF1"]
        - float(default_check.iloc[0]["MeanF1"])
    )

    default_check.to_csv(
        "sensitivity_default_check.csv",
        index=False,
    )

    diag_summary = (
        diag.groupby(
            ["parameter", "value"],
            as_index=False,
        )
        .agg(
            GateActiveRate=("GateActiveRate", "mean"),
            GatedRejectionRate=("GatedRejectionRate", "mean"),
            RetrySampleRate=("RetrySampleRate", "mean"),
            FallbackAcceptRate=("FallbackAcceptRate", "mean"),
            AttemptsPerAcceptedSample=("AttemptsPerAcceptedSample", "mean"),
            MeanMRStar=("MeanMRStar", "mean"),
            MeanAcceptProb=("MeanAcceptProb", "mean"),
            MeanSamplingTime=("SamplingTime", "mean"),
        )
    )

    diag_summary.to_csv(
        "sensitivity_gate_diagnostics_summary.csv",
        index=False,
    )

    info = [
        "MoRO-G sensitivity analysis completed.",
        "",
        "Existing main-result files were not modified.",
        "This experiment is numeric-only and one-at-a-time.",
        "",
        "Default configuration:",
        *(f"{k} = {v}" for k, v in DEFAULTS.items()),
        "",
        "Grids:",
        *(f"{k}: {v}" for k, v in SENSITIVITY_GRIDS.items()),
        "",
        "Outputs:",
        "- sensitivity_raw_folds.csv",
        "- sensitivity_summary.csv",
        "- sensitivity_by_parameter.csv",
        "- sensitivity_default_check.csv",
        "- sensitivity_gate_diagnostics.csv",
        "- sensitivity_gate_diagnostics_summary.csv",
    ]

    with open(
        "sensitivity_run_info.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(info))

    print("\n" + "=" * 78)
    print("DONE")
    print("New sensitivity files created; main results were not changed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
