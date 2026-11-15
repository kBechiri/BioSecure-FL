# -*- coding: utf-8 -*-
"""
friedman_wilcoxon_test.py - Statistical significance testing for the
metaheuristic client-selection comparison (Reviewer 2's R2-4, Reviewer 3's
R3-6).

METHODOLOGY NOTE (read before trusting the output):
participation_rate_comparison_with_fedmccs.csv stores AGGREGATED
Accuracy_Avg per (Participation_Rate, Selection_Method, Fitness_Mode)
combination -- it does NOT contain the raw per-run accuracy values that
went into each average. Following the standard nonparametric
algorithm-comparison methodology (Demsar, 2006 -- the same approach
you've already used for the CCM/HHCRSPTW Friedman/Wilcoxon results),
this script treats each (Participation_Rate x Fitness_Mode) combination
as one BLOCK, and each method's mean accuracy in that block as ONE
observation. That gives 4 rates x 3 modes = 12 blocks per method, which
is enough for a valid (if conservative) Friedman test.

FedMCCS has no Fitness_Mode (only 4 rows total, one per rate) -- it is
EXCLUDED from the Friedman/Wilcoxon comparison by default, since it
can't be aligned to the same 12-block structure as the metaheuristics.
It's already >6pp below every metaheuristic on raw accuracy (Table 2),
so a significance test isn't really the open question there anyway --
if you want it included formally, see the --include-fedmccs note in
the code below.

IF YOU HAVE RAW PER-RUN DATA (e.g. a pickled results_by_combo with each
run's final accuracy, not just the mean): swap `build_block_matrix()`
for a version that reads per-run values directly. This gives you many
more observations per method (n_runs x n_blocks instead of n_blocks)
and a much more powerful test. Ask me to write that version if you have
the pickle -- it's a straightforward variant of this script.

Usage:
    python friedman_wilcoxon_test.py --csv participation_rate_comparison_with_fedmccs.csv

Outputs:
    - Friedman chi-square statistic, p-value, Kendall's W (effect size)
    - Pairwise Wilcoxon signed-rank tests (Holm-corrected p-values)
    - A results table saved to friedman_wilcoxon_results.csv
    - (optional) a critical difference diagram, matching your existing
      CCM-paper plotting style, saved as critical_difference_diagram.png
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests


def build_block_matrix(df, metric='Accuracy_Avg', exclude_fedmccs=True):
    """
    Pivot the comparison CSV into a (n_blocks x n_methods) matrix, where
    each block = one (Participation_Rate, Fitness_Mode) combination and
    each cell = that method's `metric` value in that block.

    Returns:
        matrix: pd.DataFrame, index=block id, columns=method names
    """
    d = df.copy()
    if exclude_fedmccs:
        d = d[d['Selection_Method'] != 'FedMCCS']

    d['block'] = d['Participation_Rate'].astype(str) + '_' + d['Fitness_Mode'].astype(str)
    matrix = d.pivot_table(index='block', columns='Selection_Method', values=metric)

    # Friedman requires a complete matrix (no missing method x block cells)
    n_missing = matrix.isna().sum().sum()
    if n_missing > 0:
        missing_report = matrix.isna().sum()
        raise ValueError(
            f"Block matrix has {n_missing} missing cells -- Friedman test "
            f"requires every method to have a value in every block. "
            f"Missing counts per method:\n{missing_report[missing_report > 0]}"
        )
    return matrix


def kendalls_w(matrix):
    """
    Kendall's coefficient of concordance -- effect size for the Friedman
    test. W ranges 0 (no agreement across blocks on method ranking) to 1
    (perfect agreement). Rule of thumb: 0.1 small, 0.3 moderate, 0.5+ large.
    """
    n_blocks, n_methods = matrix.shape
    ranks = matrix.apply(lambda row: rankdata(row), axis=1, result_type='expand')
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = ((rank_sums - mean_rank_sum) ** 2).sum()
    w = 12 * ss / (n_blocks ** 2 * (n_methods ** 3 - n_methods))
    return w


def run_friedman(matrix):
    """Friedman test across all methods (columns) using the blocks (rows)."""
    stat, p = friedmanchisquare(*[matrix[col].values for col in matrix.columns])
    w = kendalls_w(matrix)
    return {'chi2': stat, 'df': matrix.shape[1] - 1, 'p_value': p, 'kendalls_w': w}


def run_pairwise_wilcoxon(matrix, holm_correct=True):
    """
    Pairwise Wilcoxon signed-rank tests between every pair of methods,
    using the same block-level values. Holm-corrected by default (more
    powerful than Bonferroni while still controlling family-wise error).

    Returns:
        pd.DataFrame with columns: method_a, method_b, statistic,
        p_raw, p_holm, significant_at_0.05, rank_biserial_r (effect size)
    """
    methods = list(matrix.columns)
    pairs = list(itertools.combinations(methods, 2))

    results = []
    for a, b in pairs:
        x, y = matrix[a].values, matrix[b].values
        diffs = x - y
        # wilcoxon errors out if all diffs are 0 (identical) -- guard for that
        if np.allclose(diffs, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y, zero_method='wilcox')
        n = np.sum(diffs != 0)
        # matched-pairs rank-biserial correlation (effect size)
        r = 1 - (2 * stat) / (n * (n + 1)) if n > 0 and not np.isnan(stat) else np.nan
        results.append({'method_a': a, 'method_b': b, 'statistic': stat,
                         'p_raw': p, 'rank_biserial_r': r})

    res_df = pd.DataFrame(results)

    if holm_correct:
        reject, p_holm, _, _ = multipletests(res_df['p_raw'], alpha=0.05, method='holm')
        res_df['p_holm'] = p_holm
        res_df['significant_at_0.05'] = reject
    else:
        res_df['p_holm'] = res_df['p_raw']
        res_df['significant_at_0.05'] = res_df['p_raw'] < 0.05

    return res_df.sort_values('p_holm').reset_index(drop=True)


def summarize_method_ranks(matrix):
    """Average rank per method across blocks (lower rank = better, i.e. rank 1 = highest accuracy)."""
    ranks = matrix.apply(lambda row: rankdata(-row), axis=1, result_type='expand')  # negate: higher accuracy -> rank 1
    ranks.columns = matrix.columns
    avg_ranks = ranks.mean(axis=0).sort_values()
    return avg_ranks


def plot_critical_difference(avg_ranks, n_blocks, alpha=0.05, output_path='critical_difference_diagram.png'):
    """
    Critical difference diagram -- matches the style you've used for the
    CCM/HHCRSPTW paper's optimizer comparisons. Methods within one CD of
    each other are not significantly different (Nemenyi post-hoc CD,
    shown as a horizontal bar connecting them).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import studentized_range

    k = len(avg_ranks)
    n = n_blocks
    # Nemenyi critical value (q_alpha for k methods), CD formula
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    fig, ax = plt.subplots(figsize=(10, 0.6 * k + 2))
    methods = avg_ranks.index.tolist()
    ranks = avg_ranks.values

    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)
    ax.set_ylim(0, k + 1)
    ax.invert_xaxis()  # rank 1 (best) on the right, matches convention
    ax.set_yticks([])
    ax.set_xlabel('Average Rank (lower = better)', fontsize=11)
    ax.set_title(f'Critical Difference Diagram (Nemenyi, CD={cd:.3f}, \u03b1={alpha})', fontsize=12)

    for i, (m, r) in enumerate(zip(methods, ranks)):
        y = k - i
        ax.plot([r, r], [y, 0.3], color='black', linewidth=0.8)
        ax.plot(r, 0.3, 'o', color='black', markersize=4)
        ax.text(r, y + 0.15, f"{m}  ({r:.2f})", ha='center', fontsize=9)

    # Draw CD bars connecting groups of methods not significantly different
    sorted_ranks = sorted(zip(methods, ranks), key=lambda t: t[1])
    groups = []
    i = 0
    while i < len(sorted_ranks):
        j = i
        while j + 1 < len(sorted_ranks) and sorted_ranks[j + 1][1] - sorted_ranks[i][1] <= cd:
            j += 1
        if j > i:
            groups.append((sorted_ranks[i][1], sorted_ranks[j][1]))
        i = j + 1

    for level, (r_lo, r_hi) in enumerate(groups):
        y = 0.3 - 0.15 * (level + 1)
        ax.plot([r_lo, r_hi], [y, y], color='red', linewidth=2)

    ax.axis('on')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return cd


def main():
    parser = argparse.ArgumentParser(description="Friedman + Wilcoxon (Holm) significance test")
    parser.add_argument('--csv', default='participation_rate_comparison_with_fedmccs.csv')
    parser.add_argument('--metric', default='Accuracy_Avg',
                         help="Column to test (e.g. Accuracy_Avg, Efficiency_Score)")
    parser.add_argument('--include-fedmccs', action='store_true',
                         help="Include FedMCCS (will fail unless you've given it a Fitness_Mode "
                              "matching the other rows -- not recommended, see module docstring)")
    parser.add_argument('--exclude', nargs='*', default=[],
                         help="Method names to drop before testing, e.g. --exclude SA")
    parser.add_argument('--output-prefix', default='')
    parser.add_argument('--no-cd-diagram', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.exclude:
        df = df[~df['Selection_Method'].isin(args.exclude)]
        print(f"Excluded methods: {args.exclude}\n")
    matrix = build_block_matrix(df, metric=args.metric, exclude_fedmccs=not args.include_fedmccs)
    n_blocks, n_methods = matrix.shape
    print(f"Block matrix: {n_blocks} blocks x {n_methods} methods (metric={args.metric})\n")

    # --- Friedman omnibus test ---
    friedman_result = run_friedman(matrix)
    print("=== Friedman Test ===")
    print(f"chi2({friedman_result['df']}) = {friedman_result['chi2']:.4f}, "
          f"p = {friedman_result['p_value']:.6f}")
    print(f"Kendall's W (effect size) = {friedman_result['kendalls_w']:.4f}")
    if friedman_result['p_value'] < 0.05:
        print("-> Reject H0: at least one method differs significantly. Proceed to pairwise tests.\n")
    else:
        print("-> Fail to reject H0: no significant difference detected among methods overall.\n"
              "   (Pairwise tests below are still reported but should be interpreted cautiously --\n"
              "   a non-significant omnibus test means the manuscript should say the accuracy\n"
              "   differences are NOT statistically significant, e.g. for the ACO-vs-GWO or\n"
              "   GWO-vs-RSELECT questions raised by R2-4/R3-6.)\n")

    # --- Average ranks per method ---
    avg_ranks = summarize_method_ranks(matrix)
    print("=== Average Rank per Method (1 = best) ===")
    print(avg_ranks.to_string())
    print()

    # --- Pairwise Wilcoxon, Holm-corrected ---
    pairwise = run_pairwise_wilcoxon(matrix, holm_correct=True)
    print("=== Pairwise Wilcoxon (Holm-corrected) ===")
    print(pairwise.to_string(index=False))
    print()

    sig_pairs = pairwise[pairwise['significant_at_0.05']]
    print(f"{len(sig_pairs)} of {len(pairwise)} pairs are significant at alpha=0.05 after Holm correction.")
    if len(sig_pairs) > 0:
        print(sig_pairs[['method_a', 'method_b', 'p_holm']].to_string(index=False))
    print()

    # --- Save results ---
    prefix = args.output_prefix
    pairwise.to_csv(f'{prefix}friedman_wilcoxon_results.csv', index=False)
    avg_ranks.to_csv(f'{prefix}average_ranks.csv', header=['avg_rank'])
    print(f"Saved: {prefix}friedman_wilcoxon_results.csv, {prefix}average_ranks.csv")

    # --- Critical difference diagram ---
    if not args.no_cd_diagram:
        cd = plot_critical_difference(avg_ranks, n_blocks, output_path=f'{prefix}critical_difference_diagram.png')
        print(f"Saved: {prefix}critical_difference_diagram.png (CD = {cd:.4f})")


if __name__ == "__main__":
    main()
