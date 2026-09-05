"""
MoRO revision — corrected sampler-only runtime benchmark
========================================================

This script measures ONLY oversampling/resampling time.

It uses the exact implementations already used in the revision:
    run_recent_baselines.py
        - MoROGSampler
        - make_kmeans_smote_safe
        - fit_resample_safe

    run_recent_baselines.py
        - KWSMOTE2025
        - RESMOTE2024Compat
        - require_smote_variants
        - published experiment constants

The three newly added baselines are included:
    - SMOTE-IPF
    - KWSMOTE-2025
    - RE-SMOTE-2024

Common binary numerical subset:
    - banknote-authentication (OpenML 1462)
    - diabetes (OpenML 37)
    - heart-statlog (OpenML 53)
    - parkinsons (OpenML 1488)
    - vertebra-column (OpenML 1524)
    - wdbc (OpenML 1510)

Important design choice
-----------------------
The supplementary predictive comparison contains:
    6 datasets x 3 classifiers = 18 dataset-classifier blocks.

However, the sampler is executed BEFORE classifier fitting. Therefore sampler
runtime does not depend on whether the downstream classifier is LR, LinearSVC
or RF. Re-running the identical sampler three times merely to attach classifier
names would artificially triple the same timing observation.

For computational-efficiency reporting this script therefore times the unique:
    6 datasets x 5 CV folds = 30 sampler-fold units.

It ALSO produces an "18-block aligned" completeness column for manuscript
bookkeeping:
    CompleteDatasetClassifierBlocks = (# datasets with all 5 folds successful) x 3

Classifier training and preprocessing are excluded from timing.

Previous result files are NEVER overwritten. All outputs are timestamped.

Dependencies
------------
pip install numpy pandas scikit-learn imbalanced-learn openml smote-variants

Place this file in the SAME DIRECTORY as:
    revised_code.py
    moro_smoteipf_kwsmote_resmote.py
"""

from __future__ import annotations

import copy
import sys
import time
import platform
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import sklearn
import imblearn
import openml

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
)
from imblearn.combine import SMOTETomek, SMOTEENN


# ---------------------------------------------------------------------
# Local project imports — exact class/function names
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from run_revised_benchmark import (
        MoROGSampler,
        make_kmeans_smote_safe,
        fit_resample_safe,
    )
except Exception as exc:
    raise ImportError(
        "\nCould not import MoRO implementation from revised_code.py.\n"
        "Put this runtime script in the same folder as revised_code.py.\n"
        f"Original error: {repr(exc)}"
    ) from exc

try:
    from run_recent_baselines import (
        KWSMOTE2025,
        RESMOTE2024Compat,
        require_smote_variants,
        KWSMOTE_K,
        KWSMOTE_L,
        KWSMOTE_TAU,
        RESMOTE_K,
        RESMOTE_RADIUS_SCALE,
        RESMOTE_MAX_TRIES_FACTOR,
    )
except Exception as exc:
    raise ImportError(
        "\nCould not import recent-baseline implementations from "
        "moro_smoteipf_kwsmote_resmote.py.\n"
        "Put this runtime script in the same folder as that file.\n"
        f"Original error: {repr(exc)}"
    ) from exc


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5

# Repeated timing reduces one-off system noise.
# Each fold/method is executed N_REPEATS times and the median is retained.
N_REPEATS = 5

# One untimed warm-up per method/fold.
WARMUP_RUNS = 1

DATASETS: List[Tuple[str, int]] = [
    ("banknote-authentication", 1462),
    ("diabetes", 37),
    ("heart-statlog", 53),
    ("parkinsons", 1488),
    ("vertebra-column", 1524),
    ("wdbc", 1510),
]

METHODS = [
    "None",
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "SVMSMOTE",
    "KMeansSMOTE",
    "SMOTE+Tomek",
    "SMOTE+ENN",
    "MoRO-G",
    "MoRO-G(no_gate)",
    "SMOTE-IPF",
    "KWSMOTE-2025",
    "RE-SMOTE-2024",
]

OUTPUT_DIR = Path("runtime_revision_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_dataset(openml_id: int) -> Tuple[pd.DataFrame, np.ndarray]:
    ds = openml.datasets.get_dataset(openml_id)

    X, y, _, _ = ds.get_data(
        dataset_format="dataframe",
        target=ds.default_target_attribute,
    )

    # This benchmark is intentionally numerical-only.
    non_numeric = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c])
    ]
    if non_numeric:
        raise ValueError(
            f"OpenML {openml_id} contains non-numeric columns: {non_numeric}"
        )

    le = LabelEncoder()
    y_enc = le.fit_transform(pd.Series(y).astype(str).to_numpy())

    if len(np.unique(y_enc)) != 2:
        raise ValueError(
            f"OpenML {openml_id} is not binary in the current loading setup."
        )

    return X, np.asarray(y_enc)


def make_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# ---------------------------------------------------------------------
# Exact sampler factories
# ---------------------------------------------------------------------
def make_moro_g(use_gate: bool):
    """
    Exact defaults used in revised_code.py.
    """
    return MoROGSampler(
        k_neighbors=5,
        k_borderline=15,
        maj_ratio_thr=0.30,
        alpha=6.0,
        epsilon=1e-12,
        mr_exponent=2.0,
        max_majority_for_gate=2000,
        random_state=RANDOM_STATE,
        use_gate=use_gate,
        max_attempts_per_sample=(12 if use_gate else 1),
        p_floor=(0.15 if use_gate else 0.0),
        p_ceil=(0.98 if use_gate else 1.0),
        fallback_accept=True,
    )


def make_runtime_sampler(method: str):
    """
    Returns an object describing exactly how the method is executed.

    Returned dictionary:
        kind:
            "none"
            "fit_resample"
            "kmeans_safe"
            "smote_ipf"
        sampler:
            method-specific object
    """

    if method == "None":
        return {"kind": "none", "sampler": None}

    if method == "SMOTE":
        return {
            "kind": "fit_resample",
            "sampler": SMOTE(random_state=RANDOM_STATE),
        }

    if method == "ADASYN":
        return {
            "kind": "fit_resample",
            "sampler": ADASYN(random_state=RANDOM_STATE),
        }

    if method == "BorderlineSMOTE":
        return {
            "kind": "fit_resample",
            "sampler": BorderlineSMOTE(random_state=RANDOM_STATE),
        }

    if method == "SVMSMOTE":
        return {
            "kind": "fit_resample",
            "sampler": SVMSMOTE(random_state=RANDOM_STATE),
        }

    if method == "KMeansSMOTE":
        # Exact safe chain used in the main pipeline:
        # default KMeansSMOTE -> tuned KMeansSMOTE -> SMOTE fallback.
        return {
            "kind": "kmeans_safe",
            "sampler": make_kmeans_smote_safe(random_state=RANDOM_STATE),
        }

    if method == "SMOTE+Tomek":
        return {
            "kind": "fit_resample",
            "sampler": SMOTETomek(random_state=RANDOM_STATE),
        }

    if method == "SMOTE+ENN":
        return {
            "kind": "fit_resample",
            "sampler": SMOTEENN(random_state=RANDOM_STATE),
        }

    if method == "MoRO-G":
        return {
            "kind": "fit_resample",
            "sampler": make_moro_g(use_gate=True),
        }

    if method == "MoRO-G(no_gate)":
        return {
            "kind": "fit_resample",
            "sampler": make_moro_g(use_gate=False),
        }

    if method == "SMOTE-IPF":
        sv = require_smote_variants()
        return {
            "kind": "smote_ipf",
            "sampler": sv.SMOTE_IPF(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        }

    if method == "KWSMOTE-2025":
        return {
            "kind": "fit_resample",
            "sampler": KWSMOTE2025(
                k=KWSMOTE_K,
                ell=KWSMOTE_L,
                tau=KWSMOTE_TAU,
                random_state=RANDOM_STATE,
            ),
        }

    if method == "RE-SMOTE-2024":
        return {
            "kind": "fit_resample",
            "sampler": RESMOTE2024Compat(
                k=RESMOTE_K,
                radius_scale=RESMOTE_RADIUS_SCALE,
                random_state=RANDOM_STATE,
                max_tries_factor=RESMOTE_MAX_TRIES_FACTOR,
            ),
        }

    raise KeyError(f"Unknown method: {method}")


def execute_sampler(spec: Dict[str, Any], X: np.ndarray, y: np.ndarray):
    kind = spec["kind"]
    sampler = spec["sampler"]

    if kind == "none":
        return X, y

    if kind == "fit_resample":
        return sampler.fit_resample(X, y)

    if kind == "kmeans_safe":
        return fit_resample_safe(
            "KMeansSMOTE",
            sampler,
            X,
            y,
        )

    if kind == "smote_ipf":
        # smote-variants uses .sample(), not .fit_resample().
        return sampler.sample(
            np.asarray(X, dtype=float),
            np.asarray(y),
        )

    raise RuntimeError(f"Unsupported execution kind: {kind}")


# ---------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------
def one_timed_execution(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    A NEW sampler object is created for every timed execution.
    Object construction itself is excluded from timing.
    Only the resampling call is timed.
    """
    spec = make_runtime_sampler(method)

    if method == "None":
        return 0.0

    t0 = time.perf_counter()
    execute_sampler(spec, X, y)
    return float(time.perf_counter() - t0)


def benchmark_method_fold(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:

    if method == "None":
        return {
            "ok": 1,
            "error": "",
            "median_sec": 0.0,
            "mean_sec": 0.0,
            "std_sec": 0.0,
            "repeat_times": [0.0] * N_REPEATS,
        }

    # Warm-up(s) are deliberately not included.
    try:
        for _ in range(WARMUP_RUNS):
            spec = make_runtime_sampler(method)
            execute_sampler(spec, X, y)
    except Exception as exc:
        return {
            "ok": 0,
            "error": f"warmup: {repr(exc)}",
            "median_sec": np.nan,
            "mean_sec": np.nan,
            "std_sec": np.nan,
            "repeat_times": [],
        }

    times = []

    try:
        for _ in range(N_REPEATS):
            elapsed = one_timed_execution(method, X, y)
            times.append(elapsed)
    except Exception as exc:
        return {
            "ok": 0,
            "error": f"timed run: {repr(exc)}",
            "median_sec": np.nan,
            "mean_sec": np.nan,
            "std_sec": np.nan,
            "repeat_times": times,
        }

    a = np.asarray(times, dtype=float)

    return {
        "ok": 1,
        "error": "",
        "median_sec": float(np.median(a)),
        "mean_sec": float(np.mean(a)),
        "std_sec": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "repeat_times": times,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    raw_rows: List[Dict[str, Any]] = []

    print("=" * 88)
    print("CORRECTED MATCHED SAMPLER-ONLY RUNTIME BENCHMARK")
    print("=" * 88)
    print(f"Datasets       : {len(DATASETS)}")
    print(f"CV folds       : {N_SPLITS}")
    print(f"Unique units   : {len(DATASETS) * N_SPLITS} sampler-fold units")
    print(f"Methods        : {len(METHODS)}")
    print(f"Repeats        : {N_REPEATS} (+ {WARMUP_RUNS} untimed warm-up)")
    print("Timed scope    : resampling call only")
    print("Classifiers    : excluded (sampler runtime is classifier-independent)")
    print("=" * 88)

    for ds_idx, (dataset_name, openml_id) in enumerate(DATASETS, start=1):
        print(f"\n[{ds_idx}/{len(DATASETS)}] {dataset_name} (OpenML {openml_id})")

        X_df, y = load_dataset(openml_id)

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for fold, (tr_idx, _) in enumerate(skf.split(X_df, y), start=1):
            X_tr_df = X_df.iloc[tr_idx]
            y_tr = y[tr_idx]

            # Preprocessing is explicitly outside the timed region.
            pre = make_preprocessor()
            X_tr = pre.fit_transform(X_tr_df)
            X_tr = np.asarray(X_tr, dtype=float)

            for method in METHODS:
                result = benchmark_method_fold(
                    method,
                    X_tr,
                    np.asarray(y_tr),
                )

                raw_rows.append(
                    {
                        "Dataset": dataset_name,
                        "OpenML_ID": openml_id,
                        "Fold": fold,
                        "Method": method,
                        "OK": result["ok"],
                        "RuntimeMedianSec": result["median_sec"],
                        "RuntimeMeanSec": result["mean_sec"],
                        "RuntimeStdSec": result["std_sec"],
                        "RepeatTimesSec": ";".join(
                            f"{x:.10f}"
                            for x in result["repeat_times"]
                        ),
                        "Error": result["error"],
                        "TrainN": int(len(y_tr)),
                        "Features": int(X_tr.shape[1]),
                    }
                )

                status = (
                    f"{result['median_sec']:.6f}s"
                    if result["ok"]
                    else f"FAILED: {result['error']}"
                )

                print(
                    f"  fold {fold} | "
                    f"{method:20s} | {status}"
                )

    raw_df = pd.DataFrame(raw_rows)

    # -------------------------------------------------------------
    # Raw output
    # -------------------------------------------------------------
    raw_path = OUTPUT_DIR / f"runtime_corrected_raw_{timestamp}.csv"
    raw_df.to_csv(raw_path, index=False)

    errors_df = raw_df.loc[
        raw_df["OK"] == 0,
        ["Dataset", "OpenML_ID", "Fold", "Method", "Error"]
    ].copy()

    error_path = OUTPUT_DIR / f"runtime_corrected_errors_{timestamp}.csv"
    errors_df.to_csv(error_path, index=False)

    # -------------------------------------------------------------
    # Method summary
    # AvgTimeSec = mean of successful fold-level MEDIAN timings.
    # -------------------------------------------------------------
    summary_rows = []

    for method in METHODS:
        m = raw_df[raw_df["Method"] == method].copy()
        ok = m[m["OK"] == 1].copy()

        successful_folds = int(len(ok))
        total_folds = int(len(m))

        # Number of datasets for which all five folds completed.
        per_dataset_success = (
            m.groupby("Dataset")["OK"]
            .agg(["sum", "count"])
            .reset_index()
        )
        complete_datasets = int(
            np.sum(
                (per_dataset_success["sum"] == N_SPLITS)
                & (per_dataset_success["count"] == N_SPLITS)
            )
        )

        # Maps directly to the 6 x 3 = 18 predictive-block bookkeeping.
        complete_18_blocks = complete_datasets * 3

        summary_rows.append(
            {
                "Method": method,
                "AvgTimeSec": (
                    float(ok["RuntimeMedianSec"].mean())
                    if successful_folds else np.nan
                ),
                "MedianTimeSec": (
                    float(ok["RuntimeMedianSec"].median())
                    if successful_folds else np.nan
                ),
                "StdAcrossSuccessfulFoldsSec": (
                    float(ok["RuntimeMedianSec"].std(ddof=1))
                    if successful_folds > 1 else 0.0
                ),
                "SuccessfulSamplerFolds": successful_folds,
                "TotalSamplerFolds": total_folds,
                "CompleteDatasets": complete_datasets,
                "CompleteDatasetClassifierBlocks": complete_18_blocks,
                "FailureCount": total_folds - successful_folds,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / f"runtime_corrected_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    # -------------------------------------------------------------
    # Paper-ready table data
    # -------------------------------------------------------------
    paper_df = summary_df[
        [
            "Method",
            "AvgTimeSec",
            "SuccessfulSamplerFolds",
            "TotalSamplerFolds",
            "CompleteDatasetClassifierBlocks",
        ]
    ].copy()

    paper_df["AvgTimeSec"] = paper_df["AvgTimeSec"].round(6)

    paper_path = OUTPUT_DIR / f"runtime_corrected_paper_table_{timestamp}.csv"
    paper_df.to_csv(paper_path, index=False)

    # LaTeX rows
    latex_path = OUTPUT_DIR / f"runtime_corrected_table_rows_{timestamp}.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% Method & Avg Time (s) & Successful folds & 18-block completeness \\\\\n")
        for _, row in paper_df.iterrows():
            method_tex = str(row["Method"]).replace("_", r"\_")
            avg = row["AvgTimeSec"]
            avg_text = "--" if pd.isna(avg) else f"{float(avg):.6f}"
            f.write(
                f"{method_tex} & {avg_text} & "
                f"{int(row['SuccessfulSamplerFolds'])}/"
                f"{int(row['TotalSamplerFolds'])} & "
                f"{int(row['CompleteDatasetClassifierBlocks'])}/18 \\\\\n"
            )

    # -------------------------------------------------------------
    # Reproducibility metadata
    # -------------------------------------------------------------
    info_path = OUTPUT_DIR / f"runtime_corrected_run_info_{timestamp}.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Corrected sampler-only runtime benchmark\n")
        f.write("=======================================\n")
        f.write(f"timestamp={timestamp}\n")
        f.write(f"python={platform.python_version()}\n")
        f.write(f"numpy={np.__version__}\n")
        f.write(f"pandas={pd.__version__}\n")
        f.write(f"scikit-learn={sklearn.__version__}\n")
        f.write(f"imbalanced-learn={imblearn.__version__}\n")
        f.write(f"openml={getattr(openml, '__version__', 'unknown')}\n")
        try:
            sv = require_smote_variants()
            f.write(
                f"smote-variants="
                f"{getattr(sv, '__version__', 'unknown')}\n"
            )
        except Exception as exc:
            f.write(f"smote-variants=ERROR {repr(exc)}\n")

        f.write(f"random_state={RANDOM_STATE}\n")
        f.write(f"n_splits={N_SPLITS}\n")
        f.write(f"n_repeats={N_REPEATS}\n")
        f.write(f"warmup_runs={WARMUP_RUNS}\n")
        f.write("timing_scope=resampling call only\n")
        f.write("preprocessing_timed=False\n")
        f.write("classifier_training_timed=False\n")
        f.write(
            "RE-SMOTE implementation="
            "RESMOTE2024Compat from moro_smoteipf_kwsmote_resmote.py\n"
        )
        f.write(
            "SMOTE-IPF implementation="
            "smote_variants.SMOTE_IPF\n"
        )

        f.write("\nDatasets:\n")
        for name, oid in DATASETS:
            f.write(f"{name}: OpenML {oid}\n")

        f.write("\nMethods:\n")
        for method in METHODS:
            f.write(f"{method}\n")

    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(summary_df.to_string(index=False))

    print("\nFiles written:")
    print(f"  {raw_path}")
    print(f"  {error_path}")
    print(f"  {summary_path}")
    print(f"  {paper_path}")
    print(f"  {latex_path}")
    print(f"  {info_path}")

    print("\nIMPORTANT:")
    print(
        "Use runtime_corrected_paper_table_*.csv for Section 5.6. "
        "Do not combine these timings with the old historical runtime table."
    )


if __name__ == "__main__":
    main()
