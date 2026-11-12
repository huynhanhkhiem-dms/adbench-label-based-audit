"""Generate dataset traceability, split stability and reviewer diagnostic summaries."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from detectors import DETECTORS, density_candidate_names
import stats as ST
from supervised import candidate_models


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 11, 23, 37, 53, 71]
ADJUSTMENTS = {
    'bh': ST.benjamini_hochberg, 'by': ST.benjamini_yekutieli, 'bonf': ST.bonferroni,
}


def read_raw(root, suffix):
    paths = sorted(root.glob(f'*__{suffix}.csv'))
    if len(paths) != 47:
        raise ValueError(f'{suffix}: expected 47 CSV files, found {len(paths)}')
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def split_summaries(primary, split_results, out):
    columns = ['dataset', 'delta', 'p']
    records = []
    summaries = []
    for seed in SEEDS:
        rows = (primary[columns] if seed == 0 else split_results.loc[split_results.seed == seed, columns]).copy()
        rows = rows.sort_values('dataset').reset_index(drop=True)
        if not rows.dataset.equals(primary.dataset):
            raise ValueError(f'Seed {seed}: expected one comparison per dataset')
        flags, adjusted = ST.benjamini_hochberg(rows.p.to_numpy())
        rows = rows.assign(seed=seed, q_bh=adjusted, flag_bh=flags)
        summaries.append({'seed': seed, 'n_datasets': len(rows), 'bh_flags': int(flags.sum()),
                          'positive_gaps': int((rows.delta > 0).sum()), 'median_gap': float(rows.delta.median()),
                          'changed_from_primary': int((flags != primary.flag_bh.to_numpy()).sum())})
        records.append(rows)
    all_splits = pd.concat(records, ignore_index=True)
    all_splits.to_csv(out / 'outer_seed_dataset_bh.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'outer_seed_counts.csv', index=False)
    return summaries, int(all_splits.groupby('dataset').flag_bh.all().sum())


def raw_summaries(raw_root, primary, out):
    ties = read_raw(raw_root, '16_detector_ties')
    candidates = read_raw(raw_root, 'all_density_candidates_posthoc')
    refit = read_raw(raw_root, 'outer_float64_ties').set_index('dataset').sort_index()
    tie_rate = 'cross_class_tie_pair_fraction'
    auc = 'heldout_auc_post_selection_diagnostic'
    if not ties.status.eq('ok').all() or not candidates.status.eq('ok').all():
        raise ValueError('At least one raw reviewer diagnostic failed')
    if len(ties) != 47 * len(DETECTORS) or ties.duplicated(['dataset', 'detector']).any():
        raise ValueError('Incomplete detector tie inventory')
    frozen = primary.set_index('dataset')
    if not (refit.index.equals(frozen.index) and refit.selected_density.equals(frozen.dens_member)
            and refit.supervised_model.equals(frozen.sup_model)):
        raise ValueError('The raw reviewer run selected different candidate names')
    for method, adjustment in ADJUSTMENTS.items():
        flags, _ = adjustment(refit.one_sided_p.to_numpy())
        if not np.array_equal(flags, frozen[f'flag_{method}'].to_numpy()):
            raise ValueError(f'The raw reviewer run changed at least one {method} decision')
    selected = candidates[candidates.selected_in_train_cv].set_index('dataset').sort_index()
    if not selected.candidate.equals(frozen.dens_member) or not np.allclose(
        selected[auc], refit.auc_density_float64, rtol=0, atol=1e-13,
    ):
        raise ValueError('Post-selection candidate diagnostics disagree with the selected refit')
    for name, group in candidates.groupby('dataset'):
        record = frozen.loc[name]
        expected = density_candidate_names(int(record.n - record.n_test), int(record.d_original))
        if sorted(group.candidate) != sorted(expected):
            raise ValueError(f'{name}: post-selection candidate list differs from eligibility rule')
    per_detector = ties.groupby('detector').agg(
        n_datasets=('dataset', 'size'), n_nonzero_cross_ties=(tie_rate, lambda v: int((v > 0).sum())),
        median_cross_tie=(tie_rate, 'median'), max_cross_tie=(tie_rate, 'max'),
        median_duplicate_score=('duplicate_score_fraction', 'median'),
    ).loc[list(DETECTORS)]
    per_detector.to_csv(out / 'detector_tie_summary.csv')
    candidate_stats = candidates.groupby('candidate').agg(
        eligible=('dataset', 'size'), selected=('selected_in_train_cv', 'sum'),
        auc_q1=(auc, lambda v: v.quantile(.25)), auc_median=(auc, 'median'),
        auc_q3=(auc, lambda v: v.quantile(.75)),
    )
    candidate_stats.to_csv(out / 'density_candidate_test_auc_summary.csv')
    gap = (candidates.groupby('dataset')[auc].max() - selected[auc]).rename('posthoc_best_minus_selected_auc')
    gap.to_csv(out / 'density_candidate_posthoc_gap.csv')
    return {
        'detector_dataset_pairs': len(ties), 'nonzero_cross_label_ties': int((ties[tie_rate] > 0).sum()),
        'candidate_dataset_pairs': len(candidates), 'min_eligible': int(candidates.groupby('dataset').size().min()),
        'max_eligible': int(candidates.groupby('dataset').size().max()),
        'median_posthoc_gap': float(gap.median()), 'mean_posthoc_gap': float(gap.mean()),
        'n_posthoc_gap_above_0.05': int((gap > .05).sum()),
        'max_posthoc_gap_dataset': gap.idxmax(), 'max_posthoc_gap': float(gap.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'revision_summaries')
    parser.add_argument('--raw-results', type=Path, default=ROOT / 'results' / 'reviewer_raw_rerun')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = ROOT / 'results'
    primary = pd.read_csv(results / 'density_gap_audit.csv').sort_values('dataset').reset_index(drop=True)
    if len(primary) != 47 or not primary.dataset.is_unique:
        raise ValueError('Expected exactly 47 primary dataset comparisons')
    for name, adjustment in ADJUSTMENTS.items():
        flags, adjusted = adjustment(primary.p.to_numpy())
        if not np.array_equal(flags, primary[f'flag_{name}']) or not np.allclose(
            adjusted, primary[f'q_{name}'], atol=0, rtol=1e-12,
        ):
            raise ValueError(f'Frozen {name} values do not match the one-sided p-values')
    columns = ['dataset', 'n', 'd', 'n_test', 'n_anom_test', 'dens_member', 'sup_model',
               'auc_dens', 'auc_sup', 'delta', 'se', 'p', 'q_bh', 'flag_bh', 'q_by', 'flag_by', 'q_bonf', 'flag_bonf']
    trace = primary[columns].copy()
    trace.insert(11, 'delong_z', trace.delta.div(trace.se.where(trace.se > 0)))
    trace.to_csv(args.out / 'primary_dataset_traceability.csv', index=False, na_rep='NA')
    seeds, stable = split_summaries(primary, pd.read_csv(results / 'split_robustness.csv'), args.out)
    raw = raw_summaries(args.raw_results, primary, args.out)
    configs = {name: model.get_params() for name, model in candidate_models(seed=0).items()}
    (args.out / 'supervised_fixed_parameters.json').write_text(json.dumps(configs, indent=2) + '\n')
    summary = {'outer_seed_counts': seeds, 'datasets_flagged_at_all_six_seeds': stable, **raw}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
