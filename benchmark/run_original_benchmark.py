# -*- coding: utf-8 -*-
"""
Run:
  python pipeline_v3.py

Outputs:
  results_raw_folds.csv
  results_summary_mean.csv
  winner_report.csv
  average_rank_report.csv
  stats_friedman_posthoc.csv
  effectsize_winloss.csv
  runtime_report.csv
  gate_mechanism_report.csv
  distribution_shift_report.csv
  distribution_shift_summary.csv
  distribution_before_after_summary.csv
  distribution_shift_smote_vs_moro.png
  distribution_shift_smote_vs_moro.pdf
  delta_f1_distribution_gate_vs_nogate.png
  delta_f1_distribution_gate_vs_nogate.pdf
"""

import time
import warnings
import inspect
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import openml

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC

# Optional robust baselines (usually available in imblearn)
try:
    from imblearn.combine import SMOTEENN, SMOTETomek
    _HAS_COMBINE = True
except Exception:
    _HAS_COMBINE = False


# =========================
# Global config
# =========================
RANDOM_STATE = 42
N_SPLITS = 5

PRIMARY_METRIC = "f1_macro"  # winner selection metric
TIE_BREAK_METRIC = "accuracy"

# 20 dataset list (OpenML IDs)
DATASETS_20 = [
    ("adult", 1590),
    ("titanic", 40945),
    ("bank-marketing", 1558),
    ("banknote-authentication", 1462),
    ("car", 21),
    ("credit-g", 31),
    ("diabetes", 37),
    ("ecoli", 40671),
    ("glass", 41),
    ("haberman", 43),
    ("heart-statlog", 53),
    ("kr-vs-kp", 3),
    ("monks-problems-1", 333),
    ("nursery", 26),
    ("page-blocks", 30),
    ("parkinsons", 1488),
    ("vehicle", 54),
    ("vertebra-column", 1524),
    ("wdbc", 1510),
    ("australian", 40981),
]

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")


# =========================
# Utilities: metrics (robust)
# =========================
def safe_gmean(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Geometric mean of per-class recall. Works for binary & multiclass."""
    eps = 1e-12
    recalls = recall_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    recalls = np.array([max(float(r), eps) for r in recalls], dtype=float)
    if recalls.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def safe_avg_precision(y_true: np.ndarray, scores_or_proba: np.ndarray, n_classes: int) -> float:
    """
    Robust average precision:
      - binary: expects scores/proba for class 1 (n_samples,)
      - multiclass: expects (n_samples, n_classes)
    y_true must be integer encoded 0..C-1.
    """
    try:
        if n_classes == 2:
            return float(average_precision_score(y_true, scores_or_proba))
        else:
            y_bin = np.zeros((len(y_true), n_classes), dtype=int)
            y_bin[np.arange(len(y_true)), y_true] = 1
            if scores_or_proba.ndim != 2 or scores_or_proba.shape[1] != n_classes:
                return float("nan")
            return float(average_precision_score(y_bin, scores_or_proba, average="macro"))
    except Exception:
        return float("nan")


def get_scores(model, X_te: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Returns:
      - binary: (n_samples,) score/proba for class 1
      - multiclass: (n_samples, n_classes)
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)
        if n_classes == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X_te)
        if n_classes == 2:
            return dec
        return dec
    if n_classes == 2:
        return np.zeros((X_te.shape[0],), dtype=float)
    return np.zeros((X_te.shape[0], n_classes), dtype=float)


# =========================
# Dataset load + type detect
# =========================
def load_openml_dataset(openml_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    ds = openml.datasets.get_dataset(openml_id)
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    return X, y


def detect_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (num_cols, cat_cols) based on pandas dtypes."""
    # avoid deprecated is_categorical_dtype
    cat_cols = []
    for c in X.columns:
        dt = X[c].dtype
        if (str(dt) in ("object", "category", "bool")) or isinstance(dt, pd.CategoricalDtype):
            cat_cols.append(c)
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def detected_type_label(num_cols: List[str], cat_cols: List[str]) -> str:
    return "numeric" if len(cat_cols) == 0 else "mixed/nominal"


# =========================
# Preprocess: ordinal encode cats, impute, scale numerics
# =========================
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return pre


def get_cat_indices(num_cols: List[str], cat_cols: List[str]) -> List[int]:
    # after ColumnTransformer, numeric first then categorical
    return list(range(len(num_cols), len(num_cols) + len(cat_cols)))


# =========================
# KMeansSMOTE: version-safe builder + fallback
# =========================
def _kmeanssmote_supported_kwargs(**kwargs):
    sig = inspect.signature(KMeansSMOTE.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_kmeans_smote_safe(random_state: int) -> Tuple[Any, Any]:
    base_kwargs = _kmeanssmote_supported_kwargs(random_state=random_state)
    tuned_kwargs = _kmeanssmote_supported_kwargs(
        random_state=random_state,
        cluster_balance_threshold=0.001,
        k_neighbors=3,
        n_clusters=10,  # auto-removed if not supported
    )
    return (KMeansSMOTE(**base_kwargs), KMeansSMOTE(**tuned_kwargs))


def fit_resample_safe(sampler_name: str, sampler_obj: Any, X_tr: np.ndarray, y_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    KMeansSMOTE: (try1, try2) with fallback to SMOTE
    """
    if sampler_name != "KMeansSMOTE":
        return sampler_obj.fit_resample(X_tr, y_tr)

    try1, try2 = sampler_obj
    try:
        return try1.fit_resample(X_tr, y_tr)
    except Exception:
        try:
            return try2.fit_resample(X_tr, y_tr)
        except Exception:
            sm = SMOTE(random_state=RANDOM_STATE)
            return sm.fit_resample(X_tr, y_tr)


# =========================
# MoRO: helpers + samplers
# =========================
@dataclass
class MoRODiagV3:
    accept_rate: float
    gate_active_rate: float
    mean_mr_star: float
    mean_accept_prob: float
    maj_sample_used: float
    retry_rate: float
    fallback_accept_rate: float


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def _euclid_dm_do_means(Xm: np.ndarray, Xo: np.ndarray, x: np.ndarray, k: int) -> Tuple[float, float]:
    eps = 1e-12
    if Xm.shape[0] < 2 or Xo.shape[0] < 2:
        return 0.0, 0.0
    k_m = min(k, Xm.shape[0])
    k_o = min(k, Xo.shape[0])

    dm = np.linalg.norm(Xm - x[None, :], axis=1)
    do = np.linalg.norm(Xo - x[None, :], axis=1)

    dm_mean = float(np.mean(np.partition(dm, k_m - 1)[:k_m])) + eps
    do_mean = float(np.mean(np.partition(do, k_o - 1)[:k_o])) + eps
    return dm_mean, do_mean


def _mr_star_from_dm_do(dm_mean: float, do_mean: float) -> float:
    # MR* ≈ (do/dm)^2
    r = do_mean / dm_mean if dm_mean > 0 else 0.0
    return float(r * r)


def _borderline_flag_from_local_mix(
    X: np.ndarray, y: np.ndarray, cls: int, x_anchor: np.ndarray,
    nn_global: NearestNeighbors, k_bl: int, maj_ratio_thr: float
) -> bool:
    k_use = min(k_bl, X.shape[0])
    idx = nn_global.kneighbors(x_anchor.reshape(1, -1), n_neighbors=k_use, return_distance=False).ravel()
    lab = y[idx]
    maj_ratio = float(np.mean(lab != cls))
    return maj_ratio >= maj_ratio_thr


class MoROGSamplerV3:
    def __init__(
        self,
        k_neighbors: int = 5,
        k_borderline: int = 15,
        maj_ratio_thr: float = 0.30,
        alpha: float = 6.0,
        max_majority_for_gate: int = 2000,
        random_state: int = 42,
        use_gate: bool = True,
        # NEW:
        max_attempts_per_sample: int = 12,
        p_floor: float = 0.15,
        p_ceil: float = 0.98,
        fallback_accept: bool = True,
    ):
        self.k_neighbors = k_neighbors
        self.k_borderline = k_borderline
        self.maj_ratio_thr = maj_ratio_thr
        self.alpha = alpha
        self.max_majority_for_gate = max_majority_for_gate
        self.random_state = random_state
        self.use_gate = use_gate

        self.max_attempts_per_sample = max_attempts_per_sample
        self.p_floor = p_floor
        self.p_ceil = p_ceil
        self.fallback_accept = fallback_accept

        self.diag_: Optional[MoRODiagV3] = None

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            self.diag_ = MoRODiagV3(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            return X, y

        max_count = int(np.max(counts))

        nn_global = NearestNeighbors(
            n_neighbors=min(self.k_borderline, X.shape[0]),
            metric="euclidean",
            algorithm="auto"
        )
        nn_global.fit(X)

        X_new = [X]
        y_new = [y]

        attempted = 0
        accepted = 0
        gate_active_cnt = 0
        mr_star_list = []
        p_list = []
        maj_used_list = []

        retries_total = 0
        fallback_accept_cnt = 0

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
                algorithm="auto"
            )
            nn_min.fit(Xc)

            Xo_full = X[y != cls]
            if Xo_full.shape[0] > self.max_majority_for_gate:
                idx = rng.choice(Xo_full.shape[0], self.max_majority_for_gate, replace=False)
                Xo = Xo_full[idx]
            else:
                Xo = Xo_full
            Xm = Xc
            maj_used_list.append(float(Xo.shape[0]))

            synth = []
            for _ in range(n_to_add):
                # retry loop
                got_one = False
                last_candidate = None
                did_retry = False

                for att in range(self.max_attempts_per_sample):
                    i = rng.randint(0, n_c)
                    x_i = Xc[i]

                    neigh_idx = nn_min.kneighbors(x_i.reshape(1, -1), return_distance=False).ravel()
                    neigh_idx = neigh_idx[neigh_idx != i]
                    if neigh_idx.size == 0:
                        continue
                    j = int(rng.choice(neigh_idx))
                    x_j = Xc[j]

                    lam = rng.rand()
                    x_cand = x_i + lam * (x_j - x_i)

                    last_candidate = x_cand
                    attempted += 1

                    if not self.use_gate:
                        synth.append(x_cand)
                        accepted += 1
                        got_one = True
                        break

                    is_borderline = _borderline_flag_from_local_mix(
                        X, y, cls, x_i, nn_global, self.k_borderline, self.maj_ratio_thr
                    )

                    if is_borderline:
                        gate_active_cnt += 1
                        dm_mean, do_mean = _euclid_dm_do_means(Xm, Xo, x_cand, self.k_neighbors)
                        mr_star = _mr_star_from_dm_do(dm_mean, do_mean)
                        p = _sigmoid(self.alpha * (mr_star - 1.0))
                        # clip for stability
                        p = float(np.clip(p, self.p_floor, self.p_ceil))

                        mr_star_list.append(mr_star)
                        p_list.append(p)

                        if rng.rand() < p:
                            synth.append(x_cand)
                            accepted += 1
                            got_one = True
                            break
                        else:
                            did_retry = True
                            continue
                    else:
                        # non-borderline: accept directly
                        synth.append(x_cand)
                        accepted += 1
                        got_one = True
                        break

                if did_retry:
                    retries_total += 1

                if (not got_one) and (last_candidate is not None) and self.fallback_accept:
                    # safety: still accept something to keep balancing
                    synth.append(last_candidate)
                    accepted += 1
                    fallback_accept_cnt += 1

            if len(synth) > 0:
                X_new.append(np.vstack(synth))
                y_new.append(np.full((len(synth),), cls, dtype=y.dtype))

        accept_rate = float(accepted / attempted) if attempted > 0 else 0.0
        gate_active_rate = float(gate_active_cnt / attempted) if attempted > 0 else 0.0
        mean_mr_star = float(np.mean(mr_star_list)) if len(mr_star_list) else 0.0
        mean_accept_prob = float(np.mean(p_list)) if len(p_list) else 1.0
        maj_sample_used = float(np.mean(maj_used_list)) if len(maj_used_list) else 0.0

        retry_rate = float(retries_total / max_count) if max_count > 0 else 0.0
        fallback_accept_rate = float(fallback_accept_cnt / max(accepted, 1))

        self.diag_ = MoRODiagV3(
            accept_rate=accept_rate,
            gate_active_rate=gate_active_rate,
            mean_mr_star=mean_mr_star,
            mean_accept_prob=mean_accept_prob,
            maj_sample_used=maj_sample_used,
            retry_rate=retry_rate,
            fallback_accept_rate=fallback_accept_rate
        )

        return np.vstack(X_new), np.concatenate(y_new)


def _gower_prepare_ranges(X_tr: np.ndarray, num_indices: List[int]) -> np.ndarray:
    if len(num_indices) == 0:
        return np.array([], dtype=float)
    mins = np.nanmin(X_tr[:, num_indices], axis=0)
    maxs = np.nanmax(X_tr[:, num_indices], axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return ranges


def _gower_distance_single(
    x: np.ndarray,
    X: np.ndarray,
    num_indices: List[int],
    cat_indices: List[int],
    num_ranges: np.ndarray
) -> np.ndarray:
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    d = np.zeros((n,), dtype=float)
    has_num = len(num_indices) > 0
    has_cat = len(cat_indices) > 0

    if has_num:
        xn = x[num_indices]
        Xn = X[:, num_indices]
        dn = np.abs(Xn - xn[None, :]) / num_ranges[None, :]
        d += np.mean(dn, axis=1)

    if has_cat:
        xc = x[cat_indices]
        Xc = X[:, cat_indices]
        dc = (Xc != xc[None, :]).astype(float)
        d += np.mean(dc, axis=1)

    denom = (1 if has_num else 0) + (1 if has_cat else 0)
    return d / max(denom, 1)


def _gower_dm_do_means(
    Xm: np.ndarray, Xo: np.ndarray, x: np.ndarray,
    num_indices: List[int], cat_indices: List[int], num_ranges: np.ndarray,
    k: int
) -> Tuple[float, float]:
    eps = 1e-12
    if Xm.shape[0] < 2 or Xo.shape[0] < 2:
        return 0.0, 0.0

    dm = _gower_distance_single(x, Xm, num_indices, cat_indices, num_ranges)
    do = _gower_distance_single(x, Xo, num_indices, cat_indices, num_ranges)

    k_m = min(k, dm.shape[0])
    k_o = min(k, do.shape[0])

    dm_mean = float(np.mean(np.partition(dm, k_m - 1)[:k_m])) + eps
    do_mean = float(np.mean(np.partition(do, k_o - 1)[:k_o])) + eps
    return dm_mean, do_mean


class MoROMixSamplerV3:
    def __init__(
        self,
        num_indices: List[int],
        cat_indices: List[int],
        k_neighbors: int = 5,
        k_borderline: int = 15,
        maj_ratio_thr: float = 0.30,
        alpha: float = 6.0,
        max_majority_for_gate: int = 2000,
        random_state: int = 42,
        use_gate: bool = True,
        # NEW:
        max_attempts_per_sample: int = 12,
        p_floor: float = 0.15,
        p_ceil: float = 0.98,
        fallback_accept: bool = True,
    ):
        self.num_indices = list(num_indices)
        self.cat_indices = list(cat_indices)
        self.k_neighbors = k_neighbors
        self.k_borderline = k_borderline
        self.maj_ratio_thr = maj_ratio_thr
        self.alpha = alpha
        self.max_majority_for_gate = max_majority_for_gate
        self.random_state = random_state
        self.use_gate = use_gate

        self.max_attempts_per_sample = max_attempts_per_sample
        self.p_floor = p_floor
        self.p_ceil = p_ceil
        self.fallback_accept = fallback_accept

        self.diag_: Optional[MoRODiagV3] = None

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            self.diag_ = MoRODiagV3(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            return X, y

        max_count = int(np.max(counts))
        num_ranges = _gower_prepare_ranges(X, self.num_indices)

        nn_global = NearestNeighbors(
            n_neighbors=min(self.k_borderline, X.shape[0]),
            metric="euclidean",
            algorithm="auto"
        )
        nn_global.fit(X)

        X_new = [X]
        y_new = [y]

        attempted = 0
        accepted = 0
        gate_active_cnt = 0
        mr_star_list = []
        p_list = []
        maj_used_list = []

        retries_total = 0
        fallback_accept_cnt = 0

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
                algorithm="auto"
            )
            nn_min.fit(Xc)

            Xo_full = X[y != cls]
            if Xo_full.shape[0] > self.max_majority_for_gate:
                idx = rng.choice(Xo_full.shape[0], self.max_majority_for_gate, replace=False)
                Xo = Xo_full[idx]
            else:
                Xo = Xo_full
            Xm = Xc
            maj_used_list.append(float(Xo.shape[0]))

            synth = []
            for _ in range(n_to_add):
                got_one = False
                last_candidate = None
                did_retry = False

                for att in range(self.max_attempts_per_sample):
                    i = rng.randint(0, n_c)
                    x_i = Xc[i]

                    neigh_idx = nn_min.kneighbors(x_i.reshape(1, -1), return_distance=False).ravel()
                    neigh_idx = neigh_idx[neigh_idx != i]
                    if neigh_idx.size == 0:
                        continue
                    j = int(rng.choice(neigh_idx))
                    x_j = Xc[j]
                    lam = rng.rand()

                    x_cand = x_i.copy()

                    if len(self.num_indices) > 0:
                        x_cand[self.num_indices] = (
                            x_i[self.num_indices] + lam * (x_j[self.num_indices] - x_i[self.num_indices])
                        )

                    if len(self.cat_indices) > 0:
                        choose_i = (rng.rand(len(self.cat_indices)) < 0.5)
                        cats = x_cand[self.cat_indices].copy()
                        cats[choose_i] = x_i[self.cat_indices][choose_i]
                        cats[~choose_i] = x_j[self.cat_indices][~choose_i]
                        x_cand[self.cat_indices] = cats

                    last_candidate = x_cand
                    attempted += 1

                    if not self.use_gate:
                        synth.append(x_cand)
                        accepted += 1
                        got_one = True
                        break

                    is_borderline = _borderline_flag_from_local_mix(
                        X, y, cls, x_i, nn_global, self.k_borderline, self.maj_ratio_thr
                    )

                    if is_borderline:
                        gate_active_cnt += 1
                        dm_mean, do_mean = _gower_dm_do_means(
                            Xm, Xo, x_cand,
                            self.num_indices, self.cat_indices, num_ranges,
                            self.k_neighbors
                        )
                        mr_star = _mr_star_from_dm_do(dm_mean, do_mean)
                        p = _sigmoid(self.alpha * (mr_star - 1.0))
                        p = float(np.clip(p, self.p_floor, self.p_ceil))

                        mr_star_list.append(mr_star)
                        p_list.append(p)

                        if rng.rand() < p:
                            synth.append(x_cand)
                            accepted += 1
                            got_one = True
                            break
                        else:
                            did_retry = True
                            continue
                    else:
                        synth.append(x_cand)
                        accepted += 1
                        got_one = True
                        break

                if did_retry:
                    retries_total += 1

                if (not got_one) and (last_candidate is not None) and self.fallback_accept:
                    synth.append(last_candidate)
                    accepted += 1
                    fallback_accept_cnt += 1

            if len(synth) > 0:
                X_new.append(np.vstack(synth))
                y_new.append(np.full((len(synth),), cls, dtype=y.dtype))

        accept_rate = float(accepted / attempted) if attempted > 0 else 0.0
        gate_active_rate = float(gate_active_cnt / attempted) if attempted > 0 else 0.0
        mean_mr_star = float(np.mean(mr_star_list)) if len(mr_star_list) else 0.0
        mean_accept_prob = float(np.mean(p_list)) if len(p_list) else 1.0
        maj_sample_used = float(np.mean(maj_used_list)) if len(maj_used_list) else 0.0

        retry_rate = float(retries_total / max_count) if max_count > 0 else 0.0
        fallback_accept_rate = float(fallback_accept_cnt / max(accepted, 1))

        self.diag_ = MoRODiagV3(
            accept_rate=accept_rate,
            gate_active_rate=gate_active_rate,
            mean_mr_star=mean_mr_star,
            mean_accept_prob=mean_accept_prob,
            maj_sample_used=maj_sample_used,
            retry_rate=retry_rate,
            fallback_accept_rate=fallback_accept_rate
        )

        return np.vstack(X_new), np.concatenate(y_new)


# =========================
# Distribution-shift helpers
# =========================
def _featurewise_mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=0)


def _minority_classes_to_augment(y: np.ndarray) -> List[int]:
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts)) if counts.size else 0
    return [int(c) for c, n in zip(classes, counts) if int(n) < max_count]


def _distribution_snapshot(
    X_before: np.ndarray,
    y_before: np.ndarray,
    X_after: np.ndarray,
    y_after: np.ndarray,
) -> Dict[str, float]:
    """
    Summarize distribution change for the classes that were augmented.
    To keep the report compact and comparable across datasets with different
    dimensions, statistics are computed on feature-wise means/stds and then
    averaged across augmented classes and features.
    """
    aug_classes = _minority_classes_to_augment(y_before)
    if len(aug_classes) == 0:
        return {
            'BeforeMean': np.nan,
            'AfterMean': np.nan,
            'BeforeStd': np.nan,
            'AfterStd': np.nan,
            'MeanShiftAbs': np.nan,
            'StdShiftAbs': np.nan,
            'GeneratedCount': 0.0,
            'AugmentedClasses': 0.0,
        }

    before_mean_vals = []
    after_mean_vals = []
    before_std_vals = []
    after_std_vals = []
    mean_shift_vals = []
    std_shift_vals = []
    gen_counts = []

    for cls in aug_classes:
        Xb = X_before[y_before == cls]
        Xa = X_after[y_after == cls]
        if Xb.shape[0] == 0 or Xa.shape[0] == 0:
            continue

        mb, sb = _featurewise_mean_std(Xb)
        ma, sa = _featurewise_mean_std(Xa)
        if mb.size == 0 or ma.size == 0:
            continue

        before_mean_vals.append(float(np.nanmean(mb)))
        after_mean_vals.append(float(np.nanmean(ma)))
        before_std_vals.append(float(np.nanmean(sb)))
        after_std_vals.append(float(np.nanmean(sa)))
        mean_shift_vals.append(float(np.nanmean(np.abs(ma - mb))))
        std_shift_vals.append(float(np.nanmean(np.abs(sa - sb))))
        gen_counts.append(float(max(0, Xa.shape[0] - Xb.shape[0])))

    if len(before_mean_vals) == 0:
        return {
            'BeforeMean': np.nan,
            'AfterMean': np.nan,
            'BeforeStd': np.nan,
            'AfterStd': np.nan,
            'MeanShiftAbs': np.nan,
            'StdShiftAbs': np.nan,
            'GeneratedCount': 0.0,
            'AugmentedClasses': float(len(aug_classes)),
        }

    return {
        'BeforeMean': float(np.mean(before_mean_vals)),
        'AfterMean': float(np.mean(after_mean_vals)),
        'BeforeStd': float(np.mean(before_std_vals)),
        'AfterStd': float(np.mean(after_std_vals)),
        'MeanShiftAbs': float(np.mean(mean_shift_vals)),
        'StdShiftAbs': float(np.mean(std_shift_vals)),
        'GeneratedCount': float(np.mean(gen_counts)),
        'AugmentedClasses': float(len(before_mean_vals)),
    }


def _sampler_family_for_distribution(sampler_name: str) -> str:
    if sampler_name == 'None':
        return 'Original'
    if sampler_name == 'SMOTE':
        return 'SMOTE'
    if sampler_name.startswith('MoRO-G'):
        return 'MoRO-G'
    if sampler_name.startswith('MoRO-Mix'):
        return 'MoRO-Mix'
    return 'Other'


# =========================
# Sampler factory
# =========================
def get_samplers(detected_type: str, cat_indices: List[int], num_indices: List[int]) -> Dict[str, Any]:
    samplers: Dict[str, Any] = {}

    samplers["None"] = None
    samplers["SMOTE"] = SMOTE(random_state=RANDOM_STATE)
    samplers["ADASYN"] = ADASYN(random_state=RANDOM_STATE)
    samplers["BorderlineSMOTE"] = BorderlineSMOTE(random_state=RANDOM_STATE)
    samplers["SVMSMOTE"] = SVMSMOTE(random_state=RANDOM_STATE)
    samplers["KMeansSMOTE"] = make_kmeans_smote_safe(random_state=RANDOM_STATE)

    if _HAS_COMBINE:
        samplers["SMOTE+Tomek"] = SMOTETomek(random_state=RANDOM_STATE)
        samplers["SMOTE+ENN"] = SMOTEENN(random_state=RANDOM_STATE)

    if detected_type == "mixed/nominal" and len(cat_indices) > 0:
        samplers["SMOTENC"] = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)

    # =========
    # ONLY CHANGE: MoRO(v3)
    # =========
    samplers["MoRO-G(v2)"] = MoROGSamplerV3(
        k_neighbors=5,
        k_borderline=15,
        maj_ratio_thr=0.30,
        alpha=6.0,
        max_majority_for_gate=2000,
        random_state=RANDOM_STATE,
        use_gate=True,
        max_attempts_per_sample=12,
        p_floor=0.15,
        p_ceil=0.98,
        fallback_accept=True,
    )
    samplers["MoRO-G(v2,no_gate)"] = MoROGSamplerV3(
        k_neighbors=5,
        k_borderline=15,
        maj_ratio_thr=0.30,
        alpha=6.0,
        max_majority_for_gate=2000,
        random_state=RANDOM_STATE,
        use_gate=False,  # no gate
        max_attempts_per_sample=1,
        p_floor=0.0,
        p_ceil=1.0,
        fallback_accept=True,
    )

    if detected_type == "mixed/nominal" and len(cat_indices) > 0:
        samplers["MoRO-Mix(full,v2)"] = MoROMixSamplerV3(
            num_indices=num_indices,
            cat_indices=cat_indices,
            k_neighbors=5,
            k_borderline=15,
            maj_ratio_thr=0.30,
            alpha=6.0,
            max_majority_for_gate=2000,
            random_state=RANDOM_STATE,
            use_gate=True,
            max_attempts_per_sample=12,
            p_floor=0.15,
            p_ceil=0.98,
            fallback_accept=True,
        )
        samplers["MoRO-Mix(full,v2,no_gate)"] = MoROMixSamplerV3(
            num_indices=num_indices,
            cat_indices=cat_indices,
            k_neighbors=5,
            k_borderline=15,
            maj_ratio_thr=0.30,
            alpha=6.0,
            max_majority_for_gate=2000,
            random_state=RANDOM_STATE,
            use_gate=False,
            max_attempts_per_sample=1,
            p_floor=0.0,
            p_ceil=1.0,
            fallback_accept=True,
        )

    return samplers


# =========================
# Models
# =========================
def get_models(n_classes: int) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    models["LR"] = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto")
    models["LinearSVC"] = LinearSVC()
    models["RF"] = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)
    return models


# =========================
# Evaluation loop
# =========================
def main():
    print("=" * 70)
    print(f"RUN START | {len(DATASETS_20)} datasets | {N_SPLITS}-fold CV")
    print(f"Primary metric: {PRIMARY_METRIC} | Tie-break: {TIE_BREAK_METRIC}")
    print("=" * 70)

    raw_rows: List[Dict[str, Any]] = []
    distribution_rows: List[Dict[str, Any]] = []

    for ds_name, ds_id in DATASETS_20:
        print(f"\n=== Dataset: {ds_name} (OpenML ID={ds_id}) ===")
        X_df, y_sr = load_openml_dataset(ds_id)

        num_cols, cat_cols = detect_feature_types(X_df)
        d_type = detected_type_label(num_cols, cat_cols)
        print(f"Detected type: {d_type} | n={X_df.shape[0]} | d={X_df.shape[1]} | cat={len(cat_cols)}")

        le = LabelEncoder()
        y = le.fit_transform(y_sr.values)
        n_classes = len(np.unique(y))

        pre = build_preprocessor(num_cols, cat_cols)
        cat_indices = get_cat_indices(num_cols, cat_cols)
        num_indices = list(range(0, len(num_cols)))

        samplers = get_samplers(d_type, cat_indices, num_indices)
        models = get_models(n_classes)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        fold_idx = 0
        for tr_idx, te_idx in skf.split(X_df, y):
            fold_idx += 1
            X_tr_df, X_te_df = X_df.iloc[tr_idx], X_df.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            X_tr = pre.fit_transform(X_tr_df)
            X_te = pre.transform(X_te_df)

            for sampler_name, sampler in samplers.items():
                t0 = time.time()

                X_rs, y_rs = X_tr, y_tr

                diag = {
                    "AcceptRate": np.nan,
                    "GateActiveRate": np.nan,
                    "MeanMRStar": np.nan,
                    "MeanAcceptProb": np.nan,
                    "MajSampleUsed": np.nan,
                    "RetryRate": np.nan,
                    "FallbackAcceptRate": np.nan,
                }

                try:
                    if sampler is None:
                        pass
                    elif sampler_name == "KMeansSMOTE":
                        X_rs, y_rs = fit_resample_safe(sampler_name, sampler, X_tr, y_tr)
                    else:
                        X_rs, y_rs = sampler.fit_resample(X_tr, y_tr)

                    if hasattr(sampler, "diag_") and sampler.diag_ is not None:
                        diag["AcceptRate"] = getattr(sampler.diag_, "accept_rate", np.nan)
                        diag["GateActiveRate"] = getattr(sampler.diag_, "gate_active_rate", np.nan)
                        diag["MeanMRStar"] = getattr(sampler.diag_, "mean_mr_star", np.nan)
                        diag["MeanAcceptProb"] = getattr(sampler.diag_, "mean_accept_prob", np.nan)
                        diag["MajSampleUsed"] = getattr(sampler.diag_, "maj_sample_used", np.nan)
                        diag["RetryRate"] = getattr(sampler.diag_, "retry_rate", np.nan)
                        diag["FallbackAcceptRate"] = getattr(sampler.diag_, "fallback_accept_rate", np.nan)

                except Exception:
                    X_rs, y_rs = X_tr, y_tr

                dist_family = _sampler_family_for_distribution(sampler_name)
                if dist_family in {"Original", "SMOTE", "MoRO-G", "MoRO-Mix"}:
                    dist_snapshot = _distribution_snapshot(X_tr, y_tr, X_rs, y_rs)
                    distribution_rows.append({
                        "dataset": ds_name,
                        "openml_id": ds_id,
                        "detected_type": d_type,
                        "fold": fold_idx,
                        "sampler": sampler_name,
                        "family": dist_family,
                        **dist_snapshot,
                    })

                for model_name, model in models.items():
                    ok = 1
                    try:
                        model.fit(X_rs, y_rs)
                        y_pred = model.predict(X_te)

                        f1m = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
                        acc = float(accuracy_score(y_te, y_pred))
                        gm = float(safe_gmean(y_te, y_pred, n_classes))

                        scores = get_scores(model, X_te, n_classes)
                        ap = float(safe_avg_precision(y_te, scores, n_classes))

                        total_time = float(time.time() - t0)

                        raw_rows.append({
                            "dataset": ds_name,
                            "openml_id": ds_id,
                            "detected_type": d_type,
                            "fold": fold_idx,
                            "sampler": sampler_name,
                            "classifier": model_name,
                            "f1_macro": f1m,
                            "accuracy": acc,
                            "gmean": gm,
                            "avg_precision": ap,
                            **diag,
                            "TotalTime": total_time,
                            "OK": ok
                        })
                    except Exception:
                        ok = 0
                        raw_rows.append({
                            "dataset": ds_name,
                            "openml_id": ds_id,
                            "detected_type": d_type,
                            "fold": fold_idx,
                            "sampler": sampler_name,
                            "classifier": model_name,
                            "f1_macro": np.nan,
                            "accuracy": np.nan,
                            "gmean": np.nan,
                            "avg_precision": np.nan,
                            **diag,
                            "TotalTime": float(time.time() - t0),
                            "OK": ok
                        })

            print(f"  Fold {fold_idx}/{N_SPLITS} done.")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv("results_raw_folds.csv", index=False)

    summary = raw_df.groupby(
        ["dataset", "openml_id", "detected_type", "sampler", "classifier"],
        as_index=False
    ).agg({
        "f1_macro": "mean",
        "accuracy": "mean",
        "gmean": "mean",
        "avg_precision": "mean",
        "AcceptRate": "mean",
        "GateActiveRate": "mean",
        "MeanMRStar": "mean",
        "MeanAcceptProb": "mean",
        "MajSampleUsed": "mean",
        "RetryRate": "mean",
        "FallbackAcceptRate": "mean",
        "TotalTime": "mean",
        "OK": "mean"
    }).rename(columns={"TotalTime": "MeanTotalTime", "OK": "OKRate"})

    summary.to_csv("results_summary_mean.csv", index=False)

    distribution_df = pd.DataFrame(distribution_rows)
    if not distribution_df.empty:
        distribution_df.to_csv("distribution_shift_report.csv", index=False)

        distribution_summary = distribution_df.groupby(
            ["dataset", "openml_id", "detected_type", "family"], as_index=False
        ).agg({
            "BeforeMean": "mean",
            "AfterMean": "mean",
            "BeforeStd": "mean",
            "AfterStd": "mean",
            "MeanShiftAbs": "mean",
            "StdShiftAbs": "mean",
            "GeneratedCount": "mean",
            "AugmentedClasses": "mean",
        })
        distribution_summary.to_csv("distribution_shift_summary.csv", index=False)

        before_after_summary = distribution_summary[[
            "dataset", "detected_type", "family",
            "BeforeMean", "AfterMean", "BeforeStd", "AfterStd",
            "MeanShiftAbs", "StdShiftAbs", "GeneratedCount"
        ]].copy()
        before_after_summary.to_csv("distribution_before_after_summary.csv", index=False)
    else:
        distribution_summary = pd.DataFrame()

    def pick_winner(df_block: pd.DataFrame) -> pd.Series:
        dfb = df_block.copy()
        dfb = dfb.dropna(subset=[PRIMARY_METRIC])
        if dfb.empty:
            return pd.Series({"winner": "None"})
        dfb = dfb.sort_values([PRIMARY_METRIC, TIE_BREAK_METRIC], ascending=[False, False])
        return pd.Series({"winner": dfb.iloc[0]["sampler"]})

    winners = summary.groupby(["dataset", "openml_id", "detected_type", "classifier"]).apply(pick_winner).reset_index()
    winner_pivot = winners.pivot_table(
        index=["dataset", "openml_id", "detected_type"],
        columns="classifier",
        values="winner",
        aggfunc="first"
    ).reset_index()

    print("\n" + "=" * 70)
    print("BEST OVERSAMPLER PER DATASET x CLASSIFIER (paper-style table)")
    print(f"Primary metric: {PRIMARY_METRIC}, tie-break: {TIE_BREAK_METRIC}")
    print("=" * 70)
    print(winner_pivot.to_string(index=False))
    winner_pivot.to_csv("winner_report.csv", index=False)

    rank_df = summary.dropna(subset=["f1_macro"]).copy()
    rank_df["rank"] = rank_df.groupby(["dataset", "classifier"])["f1_macro"].rank(ascending=False, method="average")

    avg_rank = rank_df.groupby("sampler", as_index=False).agg(
        AvgRank=("rank", "mean"),
        MeanF1=("f1_macro", "mean"),
        MeanAcc=("accuracy", "mean"),
        MeanTime=("MeanTotalTime", "mean"),
        SamplerOKRate=("OKRate", "mean"),
        ModelOKRate=("OKRate", "mean"),
        N=("rank", "count")
    ).sort_values("AvgRank", ascending=True)

    print("\n" + "=" * 70)
    print("AVERAGE RANK (lower is better)")
    print("=" * 70)
    print(avg_rank.to_string(index=False))
    avg_rank.to_csv("average_rank_report.csv", index=False)

    print("\n" + "=" * 70)
    print("SAVED OUTPUT FILES")
    print("- results_raw_folds.csv")
    print("- results_summary_mean.csv")
    print("- winner_report.csv")
    print("- average_rank_report.csv")
    print("=" * 70)

    try:
        generate_eswa_reports(raw_df=raw_df, summary_df=summary, primary_metric=PRIMARY_METRIC, distribution_df=distribution_df if not distribution_df.empty else None)
        print("\n" + "=" * 70)
        print("EXTRA ESWA REPORTS SAVED")
        print("- stats_friedman_posthoc.csv")
        print("- effectsize_winloss.csv")
        print("- runtime_report.csv")
        print("- gate_mechanism_report.csv")
        print("- distribution_shift_report.csv")
        print("- distribution_shift_summary.csv")
        print("- distribution_before_after_summary.csv")
        print("- distribution_shift_smote_vs_moro.png")
        print("- distribution_shift_smote_vs_moro.pdf")
        print("- delta_f1_distribution_gate_vs_nogate.png")
        print("- delta_f1_distribution_gate_vs_nogate.pdf")
        print("=" * 70)
    except Exception as e:
        print("\n[WARN] ESWA extra reports failed:", repr(e))

# =====================  ESWA REPORTING MODULE  =========================

def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for i, idx in enumerate(order):
        factor = (m - i)
        val = min(1.0, factor * pvals[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def _paired_win_tie_loss(a: np.ndarray, b: np.ndarray, tol: float = 0.0) -> Tuple[int, int, int]:
    d = a - b
    win = int(np.sum(d > tol))
    tie = int(np.sum(np.abs(d) <= tol))
    loss = int(np.sum(d < -tol))
    return win, tie, loss


def _rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs != 0.0]
    n = diffs.size
    if n == 0:
        return float("nan")

    abs_d = np.abs(diffs)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j + 1)
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    Wpos = float(np.sum(ranks[diffs > 0]))
    Wneg = float(np.sum(ranks[diffs < 0]))
    denom = n * (n + 1) / 2.0
    return (Wpos - Wneg) / denom


def _build_block_matrix(summary_df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, List[str]]:
    df = summary_df.copy()
    df = df.dropna(subset=[metric])
    df["block"] = df["dataset"].astype(str) + "||" + df["classifier"].astype(str)

    pivot = df.pivot_table(index="block", columns="sampler", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="any")
    pivot = pivot.dropna(axis=0, how="any")

    samplers = list(pivot.columns)
    return pivot, samplers


def generate_eswa_reports(raw_df: pd.DataFrame, summary_df: pd.DataFrame, primary_metric: str = "f1_macro", distribution_df: Optional[pd.DataFrame] = None) -> None:
    M, samplers = _build_block_matrix(summary_df, primary_metric)
    blocks_n = int(M.shape[0])

    ranks = M.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0).sort_values()
    ref = str(avg_rank.index[0])

    friedman_stat = float("nan")
    friedman_p = float("nan")
    try:
        from scipy.stats import friedmanchisquare
        arrays = [M[c].values for c in M.columns]
        friedman_stat, friedman_p = friedmanchisquare(*arrays)
        friedman_stat = float(friedman_stat)
        friedman_p = float(friedman_p)
    except Exception:
        pass

    rows_stats = []
    rows_effect = []

    try:
        from scipy.stats import wilcoxon
        has_wilcoxon = True
    except Exception:
        has_wilcoxon = False

    ref_vals = M[ref].values.astype(float)

    for s in samplers:
        s = str(s)
        vals = M[s].values.astype(float)
        diffs = vals - ref_vals

        win, tie, loss = _paired_win_tie_loss(vals, ref_vals, tol=0.0)
        rows_effect.append({
            "Reference": ref,
            "Sampler": s,
            "Blocks": blocks_n,
            "Win": win,
            "Tie": tie,
            "Loss": loss,
            "WinRate": win / max(blocks_n, 1),
            "MeanDelta": float(np.mean(diffs)),
            "MedianDelta": float(np.median(diffs)),
        })

        if s == ref:
            rows_stats.append({
                "Reference": ref,
                "Sampler": s,
                "Blocks": blocks_n,
                "Friedman_Chi2": friedman_stat,
                "Friedman_p": friedman_p,
                "Wilcoxon_p": 1.0,
                "Wilcoxon_p_Holm": 1.0,
                "RankBiserial": float("nan"),
                "AvgRank": float(avg_rank[s]),
            })
            continue

        p = float("nan")
        if has_wilcoxon:
            d = diffs[np.isfinite(diffs)]
            if d.size > 0:
                try:
                    p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    p = float("nan")

        r_rb = _rank_biserial_from_diffs(diffs)

        rows_stats.append({
            "Reference": ref,
            "Sampler": s,
            "Blocks": blocks_n,
            "Friedman_Chi2": friedman_stat,
            "Friedman_p": friedman_p,
            "Wilcoxon_p": p,
            "Wilcoxon_p_Holm": float("nan"),
            "RankBiserial": r_rb,
            "AvgRank": float(avg_rank[s]),
        })

    stats_df = pd.DataFrame(rows_stats)
    mask = (stats_df["Sampler"] != ref) & np.isfinite(stats_df["Wilcoxon_p"].values)
    pvals = stats_df.loc[mask, "Wilcoxon_p"].values.astype(float)
    if pvals.size > 0:
        stats_df.loc[mask, "Wilcoxon_p_Holm"] = _holm_adjust(pvals)
    stats_df.to_csv("stats_friedman_posthoc.csv", index=False)

    effect_df = pd.DataFrame(rows_effect)
    effect_df = effect_df.sort_values(["Reference", "WinRate", "MeanDelta"], ascending=[True, False, False])
    effect_df.to_csv("effectsize_winloss.csv", index=False)

    rt = raw_df.copy()
    rt = rt[np.isfinite(rt["TotalTime"].values)]
    rt_overall = rt.groupby("sampler", as_index=False).agg(
        MeanTime=("TotalTime", "mean"),
        MedianTime=("TotalTime", "median"),
        P90Time=("TotalTime", lambda x: float(np.quantile(x, 0.90))),
        MaxTime=("TotalTime", "max"),
        N=("TotalTime", "count")
    )

    smote_row = rt_overall[rt_overall["sampler"] == "SMOTE"]
    sm_mean = float(smote_row["MeanTime"].values[0]) if len(smote_row) else float("nan")
    sm_med = float(smote_row["MedianTime"].values[0]) if len(smote_row) else float("nan")
    rt_overall["MeanTime_vs_SMOTE"] = rt_overall["MeanTime"] / sm_mean if np.isfinite(sm_mean) else np.nan
    rt_overall["MedianTime_vs_SMOTE"] = rt_overall["MedianTime"] / sm_med if np.isfinite(sm_med) else np.nan

    outlier_rows = []
    target = "MoRO-G(v2)"
    if target in rt["sampler"].unique():
        per_ds = rt[rt["sampler"] == target].groupby(["dataset"], as_index=False).agg(
            MedianTime=("TotalTime", "median"),
            MeanTime=("TotalTime", "mean"),
            N=("TotalTime", "count")
        ).sort_values("MedianTime", ascending=False).head(10)
        for _, r in per_ds.iterrows():
            outlier_rows.append({
                "sampler": target,
                "dataset": r["dataset"],
                "MedianTime": float(r["MedianTime"]),
                "MeanTime": float(r["MeanTime"]),
                "N": int(r["N"]),
                "Tag": "Top10_median_time"
            })

    runtime_report = rt_overall.copy()
    if outlier_rows:
        runtime_report = pd.concat([runtime_report, pd.DataFrame(outlier_rows)], ignore_index=True)
    runtime_report.to_csv("runtime_report.csv", index=False)

    mech_rows = []

    def _paired_delta_blocks(df: pd.DataFrame, gated: str, nogate: str) -> pd.DataFrame:
        d = df.copy()
        d["block"] = d["dataset"].astype(str) + "||" + d["classifier"].astype(str)
        pvt = d.pivot_table(index="block", columns="sampler", values=primary_metric, aggfunc="mean")
        if gated not in pvt.columns or nogate not in pvt.columns:
            return pd.DataFrame(columns=["block", "Delta"])
        pvt = pvt[[gated, nogate]].dropna()
        return pd.DataFrame({"block": pvt.index.values, "Delta": (pvt[gated] - pvt[nogate]).values})

    diag_cols = ["AcceptRate", "GateActiveRate", "MeanMRStar", "MeanAcceptProb",
                 "MajSampleUsed", "RetryRate", "FallbackAcceptRate"]
    diag = summary_df.copy()
    keep_cols = ["dataset", "classifier", "sampler", primary_metric] + [c for c in diag_cols if c in diag.columns]
    diag = diag[keep_cols].copy()

    g, ng = "MoRO-G(v2)", "MoRO-G(v2,no_gate)"
    deltas_g = _paired_delta_blocks(summary_df, g, ng)
    if not deltas_g.empty:
        diag_g = diag[diag["sampler"] == g].copy()
        diag_g["block"] = diag_g["dataset"].astype(str) + "||" + diag_g["classifier"].astype(str)
        merged = deltas_g.merge(diag_g[["block"] + [c for c in diag_cols if c in diag_g.columns]],
                                on="block", how="left")
        merged["Pair"] = f"{g} - {ng}"
        mech_rows.append(merged)

    g2, ng2 = "MoRO-Mix(full,v2)", "MoRO-Mix(full,v2,no_gate)"
    deltas_m = _paired_delta_blocks(summary_df, g2, ng2)
    if not deltas_m.empty:
        diag_m = diag[diag["sampler"] == g2].copy()
        diag_m["block"] = diag_m["dataset"].astype(str) + "||" + diag_m["classifier"].astype(str)
        merged = deltas_m.merge(diag_m[["block"] + [c for c in diag_cols if c in diag_m.columns]],
                                on="block", how="left")
        merged["Pair"] = f"{g2} - {ng2}"
        mech_rows.append(merged)

    gate_report = pd.DataFrame()
    if mech_rows:
        gate_report = pd.concat(mech_rows, ignore_index=True)

        try:
            from scipy.stats import spearmanr
            corr_rows = []
            for pair_name, grp in gate_report.groupby("Pair"):
                if "GateActiveRate" in grp.columns:
                    x = grp["GateActiveRate"].values.astype(float)
                    yv = grp["Delta"].values.astype(float)
                    mask2 = np.isfinite(x) & np.isfinite(yv)
                    if int(np.sum(mask2)) >= 5:
                        rho, p = spearmanr(x[mask2], yv[mask2])
                        corr_rows.append({
                            "Pair": pair_name,
                            "N": int(np.sum(mask2)),
                            "SpearmanRho(GateActiveRate,Delta)": float(rho),
                            "Spearman_p": float(p)
                        })
            if corr_rows:
                gate_report = pd.concat([gate_report, pd.DataFrame(corr_rows)], ignore_index=True)
        except Exception:
            pass

    # =========================
    # ΔF1 distribution plot (gate vs no_gate)
    # =========================
    try:
        import matplotlib.pyplot as plt

        pair_to_deltas = {}

        if not deltas_g.empty:
            pair_to_deltas[f"{g} - {ng}"] = deltas_g["Delta"].values.astype(float)

        if not deltas_m.empty:
            pair_to_deltas[f"{g2} - {ng2}"] = deltas_m["Delta"].values.astype(float)

        if pair_to_deltas:
            plt.figure(figsize=(8, 5), dpi=150)

            for pair_name, deltas in pair_to_deltas.items():
                deltas = deltas[np.isfinite(deltas)]
                if deltas.size == 0:
                    continue
                plt.hist(deltas, bins="auto", alpha=0.45, density=True, label=pair_name)

            plt.axvline(0.0, linestyle="--", linewidth=1.2)
            plt.title("ΔF1-macro Distribution (gate − no_gate)")
            plt.xlabel("ΔF1-macro")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()

            plt.savefig("delta_f1_distribution_gate_vs_nogate.png")
            plt.savefig("delta_f1_distribution_gate_vs_nogate.pdf")
            plt.close()
    except Exception:
        pass

    # =========================
    # Distribution shift comparison plot (SMOTE vs MoRO)
    # =========================
    if distribution_df is not None and not distribution_df.empty:
        try:
            import matplotlib.pyplot as plt

            dist = distribution_df.groupby(["dataset", "family"], as_index=False).agg({
                "MeanShiftAbs": "mean",
                "StdShiftAbs": "mean"
            })
            keep = dist[dist["family"].isin(["SMOTE", "MoRO-G", "MoRO-Mix"])].copy()
            if not keep.empty:
                fam_summary = keep.groupby("family", as_index=False).agg({
                    "MeanShiftAbs": "mean",
                    "StdShiftAbs": "mean"
                })

                plt.figure(figsize=(8, 5), dpi=150)
                x = np.arange(fam_summary.shape[0])
                width = 0.36
                plt.bar(x - width / 2, fam_summary["MeanShiftAbs"].values, width=width, label="Mean shift")
                plt.bar(x + width / 2, fam_summary["StdShiftAbs"].values, width=width, label="Std shift")
                plt.xticks(x, fam_summary["family"].tolist(), rotation=0)
                plt.ylabel("Average absolute shift")
                plt.title("Distribution shift after augmentation: SMOTE vs MoRO")
                plt.legend()
                plt.tight_layout()
                plt.savefig("distribution_shift_smote_vs_moro.png")
                plt.savefig("distribution_shift_smote_vs_moro.pdf")
                plt.close()
        except Exception:
            pass

    gate_report.to_csv("gate_mechanism_report.csv", index=False)


if __name__ == "__main__":
    main()
