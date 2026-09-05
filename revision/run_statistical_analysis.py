# -*- coding: utf-8 -*-
"""
MoRO revision analysis-only script.

PURPOSE
-------
This script DOES NOT rerun the 20-dataset experiments and DOES NOT overwrite
any existing result, CSV, PDF, or PNG.

It reads the already-produced:
    results_summary_mean*.csv

and creates ONLY new revision reports with the prefix:
    revision_

Scientific comparison rule used here:
- Numeric datasets:
    MoRO-G / MoRO-G(no_gate) are evaluated.
    MoRO-Mix / MoRO-Mix(no_gate) / SMOTENC are excluded because they are
    mixed-data-specific.
- Mixed/nominal datasets:
    MoRO-Mix / MoRO-Mix(no_gate) / SMOTENC are evaluated.
    MoRO-G / MoRO-G(no_gate) are excluded because MoRO-G is defined as the
    numeric-data variant.

This prevents the "two different AvgRank values" issue caused by mixing
methods with different applicability domains.

NO FIGURES ARE GENERATED OR MODIFIED.
Existing figures and reports remain untouched.

Run:
    python moro_revision_analysis_only.py

Outputs:
    revision_rank_numeric.csv
    revision_rank_mixed.csv
    revision_rank_combined_variantwise.csv

    revision_stats_numeric.csv
    revision_stats_mixed.csv

    revision_win_tie_loss_numeric.csv
    revision_win_tie_loss_mixed.csv

    revision_analysis_summary.txt
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PRIMARY_METRIC = "f1_macro"
TIE_BREAK_METRIC = "accuracy"

NUMERIC_EXCLUDE = {
    "MoRO-Mix",
    "MoRO-Mix(no_gate)",
    "SMOTENC",
}

MIXED_EXCLUDE = {
    "MoRO-G",
    "MoRO-G(no_gate)",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _find_latest(pattern: str) -> Path:
    """Return most recently modified file matching pattern in current folder."""
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' was found in:\n{Path.cwd()}\n\n"
            "Place this script in the same folder as results_summary_mean.csv "
            "(or the timestamped version) and run it again."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def _normalise_sampler_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise historical display labels only.
    Numerical results are not changed.
    """
    d = df.copy()

    replacements = {
        # historical labels -> final manuscript labels
        "MoRO-G(v2)": "MoRO-G",
        "MoRO-G(v3)": "MoRO-G",
        "MoRO-G(v2,no_gate)": "MoRO-G(no_gate)",
        "MoRO-G(v3,no_gate)": "MoRO-G(no_gate)",

        "MoRO-Mix(full,v2)": "MoRO-Mix",
        "MoRO-Mix(full,v3)": "MoRO-Mix",
        "MoRO-Mix(full,v2,no_gate)": "MoRO-Mix(no_gate)",
        "MoRO-Mix(full,v3,no_gate)": "MoRO-Mix(no_gate)",
    }
    d["sampler"] = d["sampler"].replace(replacements)
    return d


def _rank_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank methods independently within each dataset x classifier block.
    All methods in `df` are applicable to the same data regime.
    """
    d = df.dropna(subset=[PRIMARY_METRIC]).copy()

    d["rank"] = d.groupby(
        ["dataset", "classifier"]
    )[PRIMARY_METRIC].rank(
        ascending=False,
        method="average"
    )

    agg_dict = {
        "AvgRank": ("rank", "mean"),
        "MeanF1": ("f1_macro", "mean"),
        "N": ("rank", "count"),
    }

    if "accuracy" in d.columns:
        agg_dict["MeanAcc"] = ("accuracy", "mean")
    if "gmean" in d.columns:
        agg_dict["MeanGMean"] = ("gmean", "mean")
    if "avg_precision" in d.columns:
        agg_dict["MeanAvgPrecision"] = ("avg_precision", "mean")

    out = (
        d.groupby("sampler", as_index=False)
         .agg(**agg_dict)
         .sort_values(["AvgRank", "MeanF1"], ascending=[True, False])
         .reset_index(drop=True)
    )
    return out


def _complete_block_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Build a complete block matrix for Friedman/Wilcoxon analyses.

    Blocks are dataset x classifier.
    Only methods observed in ALL remaining blocks are retained.
    Then blocks with any missing value are removed.
    """
    d = df.dropna(subset=[metric]).copy()
    d["block"] = d["dataset"].astype(str) + "||" + d["classifier"].astype(str)

    pvt = d.pivot_table(
        index="block",
        columns="sampler",
        values=metric,
        aggfunc="mean"
    )

    # Keep methods available in all blocks in this regime
    pvt = pvt.dropna(axis=1, how="any")
    # Safety
    pvt = pvt.dropna(axis=0, how="any")
    return pvt


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals

    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running_max = 0.0

    for i, idx in enumerate(order):
        val = min(1.0, (m - i) * pvals[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max

    return adjusted


def _rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    """
    Paired rank-biserial effect size consistent with Wilcoxon signed-rank logic.
    Positive value means the REFERENCE method tends to perform better because
    diffs are defined as reference - comparator.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = d.size

    if n == 0:
        return np.nan

    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    w_pos = float(np.sum(ranks[d > 0]))
    w_neg = float(np.sum(ranks[d < 0]))
    denom = n * (n + 1) / 2.0
    return (w_pos - w_neg) / denom


def _win_tie_loss(reference: np.ndarray, comparator: np.ndarray,
                  tol: float = 1e-12) -> Tuple[int, int, int]:
    """
    Win/Tie/Loss from REFERENCE method's perspective.
    """
    delta = reference - comparator
    win = int(np.sum(delta > tol))
    tie = int(np.sum(np.abs(delta) <= tol))
    loss = int(np.sum(delta < -tol))
    return win, tie, loss


def _stats_and_wtl(
    df: pd.DataFrame,
    reference: str,
    regime_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Friedman omnibus + pairwise Wilcoxon/Holm against a prespecified reference,
    plus Win/Tie/Loss.

    IMPORTANT:
    Reference is NOT selected as "the winner" after seeing ranks.
    Numeric analysis uses MoRO-G.
    Mixed analysis uses MoRO-Mix.
    """
    M = _complete_block_matrix(df, PRIMARY_METRIC)

    if reference not in M.columns:
        raise ValueError(
            f"Reference '{reference}' is not available in the complete "
            f"{regime_name} block matrix. Available: {list(M.columns)}"
        )

    # Friedman
    friedman_chi2 = np.nan
    friedman_p = np.nan
    if M.shape[1] >= 3 and M.shape[0] >= 2:
        try:
            from scipy.stats import friedmanchisquare
            arrs = [M[c].to_numpy(dtype=float) for c in M.columns]
            friedman_chi2, friedman_p = friedmanchisquare(*arrs)
            friedman_chi2 = float(friedman_chi2)
            friedman_p = float(friedman_p)
        except Exception:
            pass

    # Ranks on EXACT SAME complete matrix used for significance testing
    avg_ranks = M.rank(axis=1, ascending=False, method="average").mean(axis=0)

    try:
        from scipy.stats import wilcoxon
        has_wilcoxon = True
    except Exception:
        has_wilcoxon = False

    ref_vals = M[reference].to_numpy(dtype=float)

    stats_rows = []
    wtl_rows = []

    for sampler in M.columns:
        vals = M[sampler].to_numpy(dtype=float)

        # Reference - comparator: positive means MoRO variant is better
        diffs = ref_vals - vals
        win, tie, loss = _win_tie_loss(ref_vals, vals)

        wtl_rows.append({
            "Regime": regime_name,
            "Reference": reference,
            "Comparator": sampler,
            "Blocks": int(M.shape[0]),
            "Win": win,
            "Tie": tie,
            "Loss": loss,
            "WinRate": win / max(int(M.shape[0]), 1),
            "MeanDeltaF1": float(np.mean(diffs)),
            "MedianDeltaF1": float(np.median(diffs)),
        })

        if sampler == reference:
            p = 1.0
            effect = np.nan
        else:
            p = np.nan
            if has_wilcoxon:
                finite = diffs[np.isfinite(diffs)]
                if finite.size and not np.allclose(finite, 0.0):
                    try:
                        p = float(
                            wilcoxon(
                                finite,
                                zero_method="wilcox",
                                alternative="two-sided"
                            ).pvalue
                        )
                    except Exception:
                        pass
                elif finite.size:
                    p = 1.0

            effect = _rank_biserial_from_diffs(diffs)

        stats_rows.append({
            "Regime": regime_name,
            "Reference": reference,
            "Comparator": sampler,
            "Blocks": int(M.shape[0]),
            "MethodsInCommonMatrix": int(M.shape[1]),
            "Friedman_Chi2": friedman_chi2,
            "Friedman_p": friedman_p,
            "Wilcoxon_p": p,
            "Wilcoxon_p_Holm": np.nan,
            "RankBiserial_ReferenceMinusComparator": effect,
            "ReferenceAvgRank_CommonMatrix": float(avg_ranks[reference]),
            "ComparatorAvgRank_CommonMatrix": float(avg_ranks[sampler]),
            "MeanDeltaF1_ReferenceMinusComparator": float(np.mean(diffs)),
        })

    stats_df = pd.DataFrame(stats_rows)

    # Holm only across actual pairwise comparisons, excluding self
    mask = (
        (stats_df["Comparator"] != reference)
        & np.isfinite(stats_df["Wilcoxon_p"].to_numpy(dtype=float))
    )
    pvals = stats_df.loc[mask, "Wilcoxon_p"].to_numpy(dtype=float)
    if pvals.size:
        stats_df.loc[mask, "Wilcoxon_p_Holm"] = _holm_adjust(pvals)

    stats_df.loc[
        stats_df["Comparator"] == reference, "Wilcoxon_p_Holm"
    ] = 1.0

    wtl_df = pd.DataFrame(wtl_rows).sort_values(
        ["MeanDeltaF1", "WinRate"],
        ascending=[False, False]
    )

    return stats_df, wtl_df, int(M.shape[0])


def _prepare_regimes(summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric = summary[
        summary["detected_type"].astype(str).str.lower().eq("numeric")
    ].copy()
    numeric = numeric[~numeric["sampler"].isin(NUMERIC_EXCLUDE)].copy()

    mixed = summary[
        summary["detected_type"].astype(str).str.lower().eq("mixed/nominal")
    ].copy()
    mixed = mixed[~mixed["sampler"].isin(MIXED_EXCLUDE)].copy()

    return numeric, mixed


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    src = _find_latest("results_summary_mean*.csv")
    print(f"[READ ONLY] Source: {src.name}")

    original = pd.read_csv(src)
    summary = _normalise_sampler_names(original)

    required = {
        "dataset", "detected_type", "sampler",
        "classifier", "f1_macro", "accuracy"
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            "results_summary_mean file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    numeric, mixed = _prepare_regimes(summary)

    # -----------------------------
    # Applicability-aware ranks
    # -----------------------------
    rank_numeric = _rank_report(numeric)
    rank_mixed = _rank_report(mixed)

    rank_numeric.insert(0, "Regime", "Numeric")
    rank_mixed.insert(0, "Regime", "Mixed/Nominal")

    rank_numeric.to_csv("revision_rank_numeric.csv", index=False)
    rank_mixed.to_csv("revision_rank_mixed.csv", index=False)

    # A single convenience file; ranks remain regime-specific.
    combined = pd.concat(
        [rank_numeric, rank_mixed],
        ignore_index=True,
        sort=False
    )
    combined.to_csv(
        "revision_rank_combined_variantwise.csv",
        index=False
    )

    # -----------------------------
    # Statistical analyses
    # -----------------------------
    stats_num, wtl_num, blocks_num = _stats_and_wtl(
        numeric,
        reference="MoRO-G",
        regime_name="Numeric"
    )
    stats_mix, wtl_mix, blocks_mix = _stats_and_wtl(
        mixed,
        reference="MoRO-Mix",
        regime_name="Mixed/Nominal"
    )

    stats_num.to_csv("revision_stats_numeric.csv", index=False)
    stats_mix.to_csv("revision_stats_mixed.csv", index=False)

    wtl_num.to_csv("revision_win_tie_loss_numeric.csv", index=False)
    wtl_mix.to_csv("revision_win_tie_loss_mixed.csv", index=False)

    # -----------------------------
    # Human-readable summary
    # -----------------------------
    def first_rank(rank_df: pd.DataFrame, method: str):
        x = rank_df[rank_df["sampler"] == method]
        if x.empty:
            return None
        return x.iloc[0]

    g = first_rank(rank_numeric, "MoRO-G")
    m = first_rank(rank_mixed, "MoRO-Mix")

    lines: List[str] = []
    lines.append("MoRO REVISION ANALYSIS — EXISTING RESULTS ONLY")
    lines.append("=" * 64)
    lines.append(f"Input file: {src.name}")
    lines.append("")
    lines.append(
        "No classifier was refit, no resampling experiment was rerun, "
        "and no existing output file or figure was modified."
    )
    lines.append("")
    lines.append("Comparison policy:")
    lines.append(
        "- Numeric datasets: MoRO-G is compared only in the numeric regime."
    )
    lines.append(
        "- Mixed datasets: MoRO-Mix is compared only in the mixed/nominal regime."
    )
    lines.append(
        "- Statistical ranks are calculated from the exact same complete block "
        "matrix used by Friedman/Wilcoxon."
    )
    lines.append("")

    if g is not None:
        lines.append(
            f"MoRO-G numeric regime: AvgRank={g['AvgRank']:.6f}, "
            f"MeanF1={g['MeanF1']:.6f}, N={int(g['N'])}"
        )
    if m is not None:
        lines.append(
            f"MoRO-Mix mixed regime: AvgRank={m['AvgRank']:.6f}, "
            f"MeanF1={m['MeanF1']:.6f}, N={int(m['N'])}"
        )

    lines.append("")
    lines.append(
        f"Complete statistical blocks: Numeric={blocks_num}, "
        f"Mixed/Nominal={blocks_mix}"
    )
    lines.append("")
    lines.append("New files:")
    for name in [
        "revision_rank_numeric.csv",
        "revision_rank_mixed.csv",
        "revision_rank_combined_variantwise.csv",
        "revision_stats_numeric.csv",
        "revision_stats_mixed.csv",
        "revision_win_tie_loss_numeric.csv",
        "revision_win_tie_loss_mixed.csv",
    ]:
        lines.append(f"- {name}")

    Path("revision_analysis_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "\n".join(lines))
    print("\nDONE. Existing results/figures were not changed.")


if __name__ == "__main__":
    main()
