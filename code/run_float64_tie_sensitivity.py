"""Refit the selected primary pairs and break exact float64 ties at random."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

import data as D
import detectors as DET
from reviewer_reanalysis import tie_stats
from run_audit import _split
import stats as ST
from supervised import candidate_models, _scores


ROOT = Path(__file__).resolve().parents[1]
N_REPEATS = 20
REFIT_SEED = 99_991


def strict_ranks(scores, rng):
    """Randomly order equal scores, retaining the order of unequal scores."""
    order = np.lexsort((rng.random(len(scores)), scores))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(len(scores))
    return ranks


def refit_pair(item, data_root):
    index, record = item
    name = record['dataset']
    ds = D.load(name, root=data_root)
    expected = json.loads((ROOT / 'results' / 'per_dataset' / f'{name}__rep0.json').read_text())
    if ds.digest() != expected['digest']:
        raise ValueError(f'{name}: primary sample content differs from the archived digest')
    train, test = _split(ds.y, 0)
    labels = ds.y[test]
    if len(labels) != record['n_test'] or labels.sum() != record['n_anom_test']:
        raise ValueError(f'{name}: outer-test counts differ from the primary record')
    with threadpool_limits(limits=1):
        x_train, x_test, _ = D.train_standardise(ds.X[train], ds.X[test])
        density = DET.score_density_candidate(record['dens_member'], x_train, x_test, seed=REFIT_SEED)
        model = candidate_models(REFIT_SEED)[record['sup_model']]
        if 'n_jobs' in model.get_params():
            model.set_params(n_jobs=1)
        model.fit(x_train, ds.y[train])
        supervised = _scores(model, x_test)

    density = np.asarray(density, dtype=np.float64)
    supervised = np.asarray(supervised, dtype=np.float64)
    delta, se, z, p = ST.delong_test(supervised, density, labels)
    d_ties = tie_stats(density, labels)
    s_ties = tie_stats(supervised, labels)
    refit = {
        'dataset': name, 'dens_member': record['dens_member'], 'sup_model': record['sup_model'],
        'auc_dens': ST.auc(density, labels), 'auc_sup': ST.auc(supervised, labels),
        'delta': delta, 'se': se, 'z': z, 'p': p,
        'duplicate_scores_dens': d_ties['duplicate_score_fraction'],
        'duplicate_scores_sup': s_ties['duplicate_score_fraction'],
        'cross_ties_dens': d_ties['cross_class_tie_pair_fraction'],
        'cross_ties_sup': s_ties['cross_class_tie_pair_fraction'],
    }
    trials = []
    for seed in range(N_REPEATS):
        rng = np.random.default_rng(10_000 + 1009 * seed + index)
        d_rank = strict_ranks(density, rng)
        s_rank = strict_ranks(supervised, rng)
        gap, _, _, p_tie = ST.delong_test(s_rank, d_rank, labels)
        trials.append({'dataset': name, 'seed': seed, 'delta': gap, 'p': p_tie})
    return refit, trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'float64_tie_sensitivity')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    primary = pd.read_csv(ROOT / 'results' / 'density_gap_audit.csv').sort_values('dataset')
    if len(primary) != 47 or not primary.dataset.is_unique:
        raise ValueError('Expected exactly 47 primary dataset comparisons')
    items = list(enumerate(primary.to_dict(orient='records')))
    fit = partial(refit_pair, data_root=args.data_root)
    if args.jobs == 1:
        outputs = list(map(fit, items))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            outputs = list(pool.map(fit, items))
    results = pd.DataFrame(row for row, _ in outputs)
    trials = pd.DataFrame(row for _, group in outputs for row in group)
    names = primary.dataset.to_numpy()
    counts = {}
    changed = {}
    for method, adjustment in (
        ('bh', ST.benjamini_hochberg), ('by', ST.benjamini_yekutieli), ('bonf', ST.bonferroni),
    ):
        flags, q_values = adjustment(results.p.to_numpy())
        results[f'q_{method}'] = q_values
        results[f'flag_{method}'] = flags
        counts[method] = int(flags.sum())
        changed[method] = names[flags != primary[f'flag_{method}'].to_numpy()].tolist()
    trial_counts = []
    trials['q_bh'] = np.nan
    trials['flag_bh'] = False
    for seed, rows in trials.groupby('seed'):
        flags, q_values = ST.benjamini_hochberg(rows.p.to_numpy())
        trials.loc[rows.index, 'q_bh'] = q_values
        trials.loc[rows.index, 'flag_bh'] = flags
        trial_counts.append({
            'seed': int(seed), 'bh_count': int(flags.sum()),
            'changed_from_primary': names[flags != primary.flag_bh.to_numpy()].tolist(),
        })
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / 'selected_pair_refits.csv', index=False)
    trials.sort_values(['seed', 'dataset']).to_csv(args.out / 'tiebreak_20_repeats.csv', index=False)
    summary = {
        'n_datasets': len(results), 'adjusted_flag_counts': counts, 'changed_flags': changed,
        'n_cross_ties_dens': int((results.cross_ties_dens > 0).sum()),
        'n_cross_ties_sup': int((results.cross_ties_sup > 0).sum()),
        'max_abs_auc_diff_dens': float(np.max(np.abs(results.auc_dens - primary.auc_dens.to_numpy()))),
        'max_abs_auc_diff_sup': float(np.max(np.abs(results.auc_sup - primary.auc_sup.to_numpy()))),
        'trials': trial_counts,
        'environment': {
            'python': platform.python_version(), 'system': platform.system(), 'architecture': platform.machine(),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__,
            'numerical_libraries': sorted({f"{x['internal_api']} {x.get('version')}" for x in threadpool_info()}),
            'threads_per_fit': 1,
        },
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{len(results)} refits. Adjusted flag counts: {counts}. Changed from primary: {changed}.")
    print('BH counts across exact-float64 tie breaks:', [row['bh_count'] for row in trial_counts])


if __name__ == '__main__':
    main()
