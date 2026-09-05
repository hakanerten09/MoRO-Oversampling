# MoRO: Manifold-oriented Risk-aware Oversampling

This repository contains the implementation and reproducibility code associated with the manuscript on **Manifold-oriented Risk-aware Oversampling (MoRO)** for imbalanced classification.

MoRO extends interpolation-based oversampling by introducing a candidate-level risk evaluation mechanism. Instead of automatically retaining every generated synthetic candidate, the proposed approach evaluates the candidate with respect to its local relationship with the target class and competing classes before acceptance.

Two variants are considered:

- **MoRO-G** for numerical data
- **MoRO-Mix** for mixed numerical/categorical data

The repository preserves the original benchmark implementation separately from the additional experiments and analyses conducted during manuscript revision.

---

## Repository Structure

```text
MoRO-Oversampling/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── benchmark/
│   └── run_original_benchmark.py
│
└── revision/
    ├── run_revised_benchmark.py
    ├── run_statistical_analysis.py
    ├── run_sensitivity_analysis.py
    ├── run_recent_baselines.py
    └── run_runtime_analysis.py
```

The `benchmark/` directory contains the original experimental pipeline used for the main benchmark.

The `revision/` directory contains the additional analyses and experiments introduced during manuscript revision. This separation is intentional and preserves the provenance of the original experimental results.

---

## Original Benchmark

The original benchmark pipeline is provided in:

```text
benchmark/run_original_benchmark.py
```

The script contains the main experimental workflow, including:

- retrieval of benchmark datasets from OpenML;
- fold-wise data preprocessing;
- oversampling of the training data;
- training of the evaluated classifiers;
- computation of predictive performance metrics;
- candidate-gate diagnostics;
- distributional analyses;
- generation of fold-level and aggregate experimental reports.

The original benchmark evaluates 20 publicly available OpenML datasets.

---

## MoRO Variants

The final method names used in the manuscript are:

```text
MoRO-G
MoRO-G(no_gate)
MoRO-Mix
MoRO-Mix(no_gate)
```

**MoRO-G** is the numerical-data variant of the proposed method.

**MoRO-Mix** is designed for datasets containing mixed numerical and categorical attributes.

The corresponding `no_gate` variants retain the candidate-generation mechanism while disabling candidate-level gate-based rejection. They are used to examine the contribution of the gating mechanism.

Historical internal labels may still occur in the original benchmark script because that file is retained to preserve the original experimental pipeline. The final terminology used in the manuscript and revision analyses is the terminology listed above.

---

## Datasets

The main benchmark uses the following 20 datasets obtained from OpenML:

| Dataset | OpenML ID |
|---|---:|
| Adult | 1590 |
| Titanic | 40945 |
| Bank Marketing | 1558 |
| Banknote Authentication | 1462 |
| Car Evaluation | 21 |
| Credit-G | 31 |
| Diabetes | 37 |
| Ecoli | 40671 |
| Glass | 41 |
| Haberman | 43 |
| Heart Statlog | 53 |
| KR-vs-KP | 3 |
| Monks Problems 1 | 333 |
| Nursery | 26 |
| Page Blocks | 30 |
| Parkinsons | 1488 |
| Vehicle | 54 |
| Vertebra Column | 1524 |
| WDBC | 1510 |
| Australian Credit | 40981 |

The OpenML identifiers are specified explicitly to support reproducible dataset retrieval.

---

## Experimental Protocol

The main benchmark uses stratified five-fold cross-validation with a fixed random seed:

```python
random_state = 42
```

Three classifiers are evaluated:

- Logistic Regression
- LinearSVC
- Random Forest

Preprocessing is fitted exclusively on the training partition of each cross-validation fold.

For numerical attributes, the main preprocessing steps include median imputation and standardization. Categorical attributes, where applicable, are processed within the corresponding mixed-data pipeline.

Oversampling is applied only to the training data. The test partition is not included in fitting the preprocessing transformations or in the resampling procedure.

---

## Compared Oversampling Methods

Depending on dataset applicability, the benchmark includes the following methods:

- None
- SMOTE
- BorderlineSMOTE
- SVMSMOTE
- ADASYN
- KMeansSMOTE
- SMOTE+Tomek
- SMOTE+ENN
- SMOTENC
- MoRO-G
- MoRO-G(no_gate)
- MoRO-Mix
- MoRO-Mix(no_gate)

Additional recent oversampling methods are evaluated separately in the revision experiments.

---

## Default MoRO Configuration

The principal MoRO configuration used in the revised experiments is:

| Parameter | Default value |
|---|---:|
| `k_neighbors` | 5 |
| `k_borderline` | 15 |
| `tau_majority` | 0.30 |
| `alpha` | 6.0 |
| `mr_exponent (q)` | 2.0 |
| `epsilon` | 1e-12 |
| `max_majority_for_gate` | 2000 |
| `max_attempts_per_sample` | 12 |
| `p_floor` | 0.15 |
| `p_ceil` | 0.98 |
| `fallback_accept` | True |
| `random_state` | 42 |

Candidate generation uses a bounded retry mechanism.

When gating is active, a candidate is evaluated according to its relative local proximity to samples from the target and competing classes. The resulting acceptance probability is bounded by `p_floor` and `p_ceil`.

If no candidate is accepted after `max_attempts_per_sample` attempts and fallback acceptance is enabled, the **last generated candidate** is retained as the fallback sample.

The fallback mechanism therefore guarantees termination of the bounded candidate-generation procedure.

---

## Evaluation Metrics

The primary predictive performance measure used in the study is:

- **Macro F1-score**

Complementary measures include:

- Accuracy
- G-mean
- Average Precision (AP)

Fold-level measurements are used for the matched statistical analyses and stability-related evaluations.

---

## Revision Experiments

Additional experiments and analyses introduced during manuscript revision are provided under:

```text
revision/
```

The revision scripts are kept separate from the original benchmark to avoid altering the provenance of the original experimental results.

### Revised benchmark

```text
revision/run_revised_benchmark.py
```

This script contains the revised implementation used to support the additional analyses and the final MoRO terminology.

### Statistical analysis

```text
revision/run_statistical_analysis.py
```

This script performs applicability-aware statistical analyses using previously generated benchmark results.

The numerical and mixed/nominal regimes are evaluated separately:

- MoRO-G is evaluated within the numerical-data regime.
- MoRO-Mix is evaluated within the mixed/nominal-data regime.

The statistical analysis includes:

- average-rank analysis;
- Friedman omnibus testing;
- paired Wilcoxon signed-rank tests;
- Holm correction for multiple comparisons;
- paired rank-biserial effect sizes;
- Win/Tie/Loss analysis.

The statistical-analysis script does not refit the classifiers or replace the original benchmark results.

### Sensitivity analysis

```text
revision/run_sensitivity_analysis.py
```

This script performs a one-factor-at-a-time sensitivity analysis for MoRO-G.

The investigated values include:

```text
q            = {1, 2, 3}
alpha        = {2, 4, 6, 8, 10}
tau_majority = {0.20, 0.30, 0.40, 0.50}
epsilon      = {1e-14, 1e-12, 1e-10, 1e-8}
```

The sensitivity experiment evaluates the local behavior of the fixed MoRO configuration. It is not used to perform dataset-specific hyperparameter optimization or to retrospectively select a new default configuration.

### Recent-baseline comparison

```text
revision/run_recent_baselines.py
```

This experiment extends the comparison with three additional oversampling approaches:

- SMOTE-IPF
- KWSMOTE-2025
- RE-SMOTE-2024

For a common and methodologically compatible comparison, the experiment is conducted on six binary numerical datasets:

- Banknote Authentication
- Diabetes
- Heart Statlog
- Parkinsons
- Vertebra Column
- WDBC

Together with the three classifiers, this produces 18 matched dataset-classifier blocks.

The RE-SMOTE experiment uses an independent compatibility implementation based on the published method and the authors' publicly available implementation. It should therefore not be interpreted as a byte-for-byte reproduction of the original authors' software environment.

### Runtime analysis

```text
revision/run_runtime_analysis.py
```

The runtime experiment measures the computational cost of the resampling procedures separately from classifier training.

It uses the same six binary numerical datasets employed in the recent-method comparison.

For each dataset, five cross-validation training folds are considered. Since sampler runtime does not depend on the downstream classifier, timing is performed once for each dataset-fold combination rather than redundantly repeating the same resampling operation for each classifier.

Each method/fold combination uses:

```text
1 untimed warm-up
5 timed repetitions
```

The median of the repeated measurements is retained for each fold.

Only the resampling call is timed. Preprocessing and classifier training are excluded from the measured interval.

---

## Installation

Python 3 is required.

Clone the repository:

```bash
git clone https://github.com/hakanerten09/MoRO-Oversampling.git
cd MoRO-Oversampling
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The exact package versions used in the experimental environment can be specified in `requirements.txt` to support environment-level reproducibility.

---

## Running the Original Benchmark

The original benchmark can be executed with:

```bash
python benchmark/run_original_benchmark.py
```

The script retrieves the required datasets from OpenML and generates the corresponding experimental outputs.

---

## Running the Revision Experiments

### Revised benchmark

```bash
python revision/run_revised_benchmark.py
```

### Statistical analysis

```bash
python revision/run_statistical_analysis.py
```

### Sensitivity analysis

```bash
python revision/run_sensitivity_analysis.py
```

### Recent-baseline comparison

```bash
python revision/run_recent_baselines.py
```

### Runtime analysis

```bash
python revision/run_runtime_analysis.py
```

Some revision scripts operate on outputs generated by the benchmark or other revision experiments. The comments at the beginning of each script describe the required inputs and generated outputs.

---

## Generated Outputs

This repository intentionally focuses on the source code required for reproducibility rather than storing the complete collection of generated experimental files.

Running the provided scripts generates the corresponding CSV reports and intermediate outputs, including:

- fold-level predictive results;
- aggregate performance summaries;
- average-rank reports;
- statistical comparison reports;
- Win/Tie/Loss analyses;
- gate diagnostics;
- sensitivity-analysis outputs;
- recent-baseline comparison results;
- sampler-only runtime measurements.

This organization keeps the repository compact while retaining the code required to regenerate the experimental analyses.

---

## Reproducibility Notes

The repository intentionally distinguishes between the **original benchmark** and the **additional analyses introduced during manuscript revision**.

The original benchmark implementation is retained under:

```text
benchmark/
```

Additional statistical, sensitivity, recent-baseline, and runtime experiments are maintained under:

```text
revision/
```

This separation preserves the experimental provenance of the study and avoids presenting revision-stage analyses as part of the original benchmark.

The principal random seed used throughout the experiments is `42`.

---

## Data Availability

All benchmark datasets used in the study are publicly available through OpenML.

The OpenML dataset identifiers required to retrieve the benchmark data are provided in this README, in the source code, and in the accompanying manuscript.

No private dataset is required to reproduce the reported benchmark experiments.

---

## Code Availability

The source code required for the original benchmark and the additional experiments conducted during manuscript revision is provided in this repository.

The complete collection of generated result files is not duplicated in the repository; the corresponding outputs can be regenerated using the provided experimental scripts.

---

## Citation

If you use MoRO or this implementation in academic work, please cite the associated manuscript.

The final bibliographic information will be added after publication.

---

## License

This project is distributed under the MIT License.
