"""Render the manuscript's result tables from the archived CSV and JSON outputs."""

from pathlib import Path
import argparse
import json
import math

import pandas as pd

from detectors import density_candidate_names


RESULTS = Path(__file__).resolve().parents[1] / 'results'
END_ROW = r' \\'
MARK_ROW = r'\revrow '


def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%')


def row(values, marked=False):
    return (MARK_ROW if marked else '') + ' & '.join(map(str, values)) + END_ROW


def probability(value):
    if value == 0:
        return '0'
    if value >= .001:
        return f'{value:.4f}'
    mantissa, exponent = f'{value:.2e}'.split('e')
    return rf'${mantissa}\!\times\!10^{{{int(exponent)}}}$'


def tie_rate(value):
    value *= 100
    return f'{value:.3f}' if value == 0 or value >= .001 else probability(value)


def short_table(path, columns, header, rows):
    path.write_text('\n'.join([
        rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}', r'\toprule', row(header), r'\midrule',
        *rows, r'\bottomrule', r'\end{tabular}', '',
    ]))


def long_table(path, columns, caption, label, header, rows):
    header_row = row(header)
    path.write_text('\n'.join([
        r'\begingroup\small',
        rf'\begin{{longtable}}{{@{{}}{columns}@{{}}}}',
        rf'\caption{{{caption}}}\label{{{label}}}' + END_ROW,
        r'\toprule', header_row, r'\midrule', r'\endfirsthead',
        r'\toprule', header_row, r'\midrule', r'\endhead',
        r'\bottomrule', r'\endfoot', *rows, r'\end{longtable}', r'\endgroup', '',
    ]))


def theory_table(out):
    t = json.loads((RESULTS / 'theory.json').read_text())
    identification, lda, worlds, rare = [t[key] for key in ('theorem2', 'theorem7', 'proposition10', 'proposition12')]
    short_table(out / 'theory_results.tex', r'lp{.48\linewidth}l', ['Result', 'Quantity', 'Measured'], [
        row(['Theorem 2', 'Width of identified AUC set', f"{identification['identified_set_width']:.3f}"]),
        row(['Theorem 2', 'Maximum numerical construction error', probability(identification['max_abs_error'])]),
        row(['Theorem 7', 'LDA/optimality mismatches', f"{lda['mismatches']} of {lda['n_grids']:,}"]),
        row(['Theorem 7', r'Bayes gap of $-p$ when LDA holds', f"{lda['max_gap_when_monotone']:.1f}"]),
        row(['Theorem 7', 'Mean Bayes gap when LDA fails', f"{lda['mean_gap_when_not_monotone']:.3f}"]),
        row(['Proposition 10', 'Mass-volume regret, worlds A/B', f"{worlds['regret_A_mass_volume']:.3f} / {worlds['regret_B_mass_volume']:.3f}"]),
        row(['Proposition 10', 'Excess-mass regret, worlds A/B', f"{worlds['regret_A_excess_mass']:.3f} / {worlds['regret_B_excess_mass']:.3f}"]),
        row(['Proposition 12', 'Log--log slope of SE vs. anomaly count', rf"${rare['log_log_slope']:.3f}$"]),
    ])


def main_result_tables(out):
    seeds = pd.read_csv(RESULTS / 'revision_summaries' / 'outer_seed_counts.csv')
    short_table(out / 'outer_seed_counts.tex', 'lrrrr', ['Outer seed', 'BH flags', 'Positive gaps', 'Median gap', 'Flags changed'], [
        row([f'{x.seed} (primary)' if x.seed == 0 else x.seed, x.bh_flags, x.positive_gaps,
             f'{x.median_gap:.4f}', x.changed_from_primary], marked=True)
        for x in seeds.itertuples()
    ])
    dups = pd.read_csv(RESULTS / 'additional_sensitivity' / 'duplicate_exclusion_sensitivity.csv').set_index('sensitivity')
    names = {
        'all': 'All datasets',
        'exclude_dup_ge_10pct': r'Exclude duplicate rate $\ge10\%$',
        'exclude_dup_ge_5pct': r'Exclude duplicate rate $\ge5\%$',
        'exclude_dup_ge_1pct': r'Exclude duplicate rate $\ge1\%$',
        'exclude_any_recorded_duplicate': 'Exclude any recorded duplicate',
        'confirmed_zero_duplicates_only': 'Confirmed zero duplicates',
    }
    short_table(out / 'duplicates.tex', 'lrrrrr', ['Analysis', 'Datasets', 'BH', 'BY', 'Bonf.', 'Median gap'], [
        row([names[key], int(x.n_datasets), int(x.bh_flags), int(x.by_flags), int(x.bonf_flags), f'{x.median_delta:.3f}'])
        for key, x in dups.reindex(names).iterrows()
    ])
    all_detectors = pd.read_csv(RESULTS / 'leaderboard_all.csv').set_index('detector')
    flagged = pd.read_csv(RESULTS / 'leaderboard_flagged.csv').set_index('detector')
    unflagged = pd.read_csv(RESULTS / 'leaderboard_not_flagged.csv').set_index('detector')
    def pair(frame, model):
        return [f'{frame.loc[model, "mean_auc"]:.3f}', f'{frame.loc[model, "mean_rank"]:.1f}']
    grouped_header = r' & & \multicolumn{3}{c}{All datasets} & \multicolumn{2}{c}{Flagged} & \multicolumn{2}{c}{Nonflagged} \\'
    short_table(out / 'leaderboard.tex', 'llrrrrrrr',
                ['Detector', 'Family', 'AUC', 'Rank', 'Wins', 'AUC', 'Rank', 'AUC', 'Rank'], [
        row([esc(model), x.family, *pair(all_detectors, model), int(x.n_wins), *pair(flagged, model), *pair(unflagged, model)])
        for model, x in all_detectors.iterrows()
    ])
    table = out / 'leaderboard.tex'
    table.write_text(table.read_text().replace(r'\toprule', r'\toprule' + '\n' + grouped_header, 1))


def appendix_tables(out):
    summary = RESULTS / 'revision_summaries'
    ties = pd.read_csv(summary / 'detector_tie_summary.csv')
    long_table(out / 'detector_ties.tex', 'lrrrr', r'Descriptive tie inventory by detector. Rates are percentages; counts have denominator 47.',
        'tab:ties', ['Detector', 'Datasets with ties', 'Median cross-label', 'Max. cross-label', 'Median duplicate'], [
        row([esc(x.detector), x.n_nonzero_cross_ties, tie_rate(x.median_cross_tie),
             tie_rate(x.max_cross_tie), tie_rate(x.median_duplicate_score)], marked=True)
        for x in ties.itertuples()
    ])
    detail = pd.read_csv(summary / 'primary_dataset_traceability.csv')
    detail = detail.sort_values('dataset', key=lambda s: s.str.split('_').str[0].astype(int))
    long_table(out / 'primary_auc.tex', 'lrrllrrr', 'Primary selected models, test sample sizes and held-out AUCs.',
        'tab:primary-auc', ['Dataset', 'Test', 'Anom.', 'Density member', 'Ref.', 'AUC D', 'AUC S', 'Gap'], [
        row([esc(x.dataset), x.n_test, x.n_anom_test, esc(x.dens_member), x.sup_model,
             f'{x.auc_dens:.4f}', f'{x.auc_sup:.4f}', f'{x.delta:+.4f}'], marked=True)
        for x in detail.itertuples()
    ])
    long_table(out / 'primary_tests.tex', 'lrrrrrl', 'Primary paired DeLong tests and adjusted probabilities. Flags are ordered BH/BY/Bonferroni.',
        'tab:primary-tests', ['Dataset', r'$z$', r'Raw $p$', 'BH adj.', 'BY adj.', 'Bonf. adj.', 'Flags'], [
        row([esc(x.dataset), 'NA' if math.isnan(x.delong_z) else f'{x.delong_z:.2f}',
             *[probability(q) for q in (x.p, x.q_bh, x.q_by, x.q_bonf)],
             '/'.join('Y' if flag else 'N' for flag in (x.flag_bh, x.flag_by, x.flag_bonf))], marked=True)
        for x in detail.itertuples()
    ])
    candidates = pd.read_csv(summary / 'density_candidate_test_auc_summary.csv').set_index('candidate')
    long_table(out / 'density_candidate_summary.tex', 'lrrrrr', 'Density-family behavior. Quartiles summarize held-out AUCs over eligible datasets.',
        'tab:candidates', ['Density candidate', 'Eligible', 'Selected', 'Q1', 'Median', 'Q3'], [
        row([esc(k), int(x.eligible), int(x.selected), f'{x.auc_q1:.3f}', f'{x.auc_median:.3f}', f'{x.auc_q3:.3f}'], marked=True)
        for k, x in candidates.reindex(density_candidate_names(10000, 10)).iterrows()
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.source_dir / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    theory_table(out)
    main_result_tables(out)
    appendix_tables(out)


if __name__ == '__main__':
    main()
