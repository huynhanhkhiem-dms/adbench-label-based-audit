# Auditing Label-Based Evaluation in Unsupervised Anomaly Detection

Reproduction materials for *Auditing Label-Based Evaluation in Unsupervised Anomaly Detection: Identifiability Limits and Evidence from ADBench*, by Huynh Anh Khiem (ORCID 0009-0007-7210-174X).

The primary analysis compares a supervised reference and a training-selected density ranking on the held-out half of each of 47 Classical ADBench datasets. The original paired comparisons give 38 Benjamini–Hochberg, 33 Benjamini–Yekutieli and 29 Bonferroni flags. These decisions apply to the fitted scores in the specified representation and split. They describe an operational comparison with a finite density family.

## Inputs and environment

Use Python 3.12 and install the pinned packages.

```sh
python -m pip install -r requirements.txt
```

Obtain the 47 Classical NPZ files from [the official ADBench repository](https://github.com/Minqi824/ADBench), under `adbench/datasets/Classical`. Set the path to your local copy before running commands that fit models.

```sh
export ADBENCH_ROOT="/path/to/ADBench/adbench/datasets/Classical"
```

The repository contains derived tables and run records. The benchmark files and observation-level labels and scores are obtained or generated locally. [`data_manifest.csv`](data_manifest.csv) records SHA-256 hashes for the input files and the primary sample digests used by the archived analysis. The selected-pair check validates each sample digest before fitting.

## Verify the deposited results

From this directory:

```sh
python code/tests/test_core.py
python code/summarize_revision.py
python code/run_float64_tie_sensitivity.py --data-root "$ADBENCH_ROOT" --jobs 4
python code/make_submission_figures.py
python code/make_submission_outputs.py --source-dir manuscript_tables
```

The first summary command uses the archived primary records and the 141 derived diagnostic CSVs. It reproduces the per-dataset traceability, split sensitivity, ties by detector and held-out candidate AUC quartiles. The float64 check refits the archived selected model identities on the primary splits with one thread per fit. It recomputes the three adjustments and performs 20 random order-preserving tie breaks. The underlying files and software environment are recorded in `results/float64_tie_sensitivity/`. Numerical implementations can differ on very close scores. The original table is the reference for manuscript AUCs and probabilities.

## Rerun selection and analysis from inputs

Use a separate copy of this directory for a from-data rerun, since the following commands regenerate its results. The full nested audit is a CPU workload and can take several hours. Use the same input files and package versions as above.

```sh
AUDIT_ROBUSTNESS_SPLITS=5 python code/run_audit.py --force
python code/analyse.py
python code/run_theory.py
python code/run_label_budget.py
python code/reviewer_reanalysis.py --data-root "$ADBENCH_ROOT" --out results/reviewer_raw_rerun --all-density
python code/summarize_revision.py
python code/make_submission_figures.py
python code/make_submission_outputs.py --source-dir manuscript_tables
```

Replicate 0 and outer seed 0 define the primary tests. The five alternative outer seeds repeat selection on replicate 0. The four extra subsamples for datasets above 20,000 rows are a separate sensitivity. The audit writes local score copies for the label-budget curves. Fit-level environment differences can affect individual numeric values on a from-data rerun.

For the remaining sensitivity outputs:

```sh
python code/additional/run_additional_sensitivity.py --stage known_lda
python code/additional/run_additional_sensitivity.py --stage non_gaussian_lda
python code/additional/run_additional_sensitivity.py --stage duplicates
python code/additional/run_additional_sensitivity.py --stage small_counts
python code/additional/run_additional_sensitivity.py --stage synbench
```

## Result files

| Path | Content |
| --- | --- |
| `results/density_gap_audit.csv` | Original 47 primary comparisons, paired confidence intervals and adjustments |
| `results/per_dataset/` | Dataset- and replicate-level run records, selection AUCs and fixed-split sensitivity |
| [`results/revision_summaries/`](results/revision_summaries/) | Full-precision inputs for Tables B1–B4, split counts, and fixed supervised parameters. [Primary traceability CSV](results/revision_summaries/primary_dataset_traceability.csv) |
| `results/float64_tie_sensitivity/` | Selected-pair refits, exact float64 ties and all 20 tie-breaking repeats |
| `results/reviewer_raw_rerun/` | 141 diagnostic CSVs, including the 16-detector tie inventory and 1,244 candidate AUCs |
| `results/additional_sensitivity/` | Known-LDA calibration, duplicate exclusion, low-count checks and SynBench-aligned stress tests |

The post-selection test AUCs for unchosen density members describe the family. Selection in the primary audit uses only outer-training validation.
