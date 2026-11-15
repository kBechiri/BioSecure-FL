# -*- coding: utf-8 -*-
"""
================================================================================
fedmccs_plots.py - Plotting for the FedMCCS baseline comparison
================================================================================
Self-contained (does not depend on ResultBS.py, which was not available
when this module was written). Reads the CSV schema produced by
fedmccs_experiment.py and BioSecure.py's run_participation_rate_comparison.

Three plots, matching the paper's existing visual language:
  1. plot_accuracy_with_fedmccs   -- Figure 2-style: accuracy vs
     participation rate, metaheuristics (mean +/- std) vs FedMCCS.
  2. plot_fedmccs_fulfillment     -- new diagnostic: how often FedMCCS's
     greedy filter actually reaches its target client count, per rate.
  3. plot_pareto_with_fedmccs     -- Figure 6-style: accuracy vs energy
     per selected client, all methods plus FedMCCS marked distinctly.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def _rate_label(rate):
    return f"{int(round(rate * 100))}%"


def plot_accuracy_with_fedmccs(df_comparison, df_fedmccs, output_dir,
                                filename='accuracy_with_fedmccs.png'):
    """
    Bar chart: mean accuracy (+/- std) per participation rate, comparing
    the metaheuristic methods (averaged across method x fitness_mode) to
    the FedMCCS baseline. Mirrors the paper's Figure 2 framing.
    """
    rates = sorted(df_comparison['Participation_Rate'].unique())

    meta_means, meta_stds = [], []
    fed_means, fed_stds = [], []

    for rate in rates:
        meta_rows = df_comparison[df_comparison['Participation_Rate'] == rate]
        meta_means.append(meta_rows['Accuracy_Avg'].mean())
        meta_stds.append(meta_rows['Accuracy_Avg'].std())

        fed_row = df_fedmccs[df_fedmccs['Participation_Rate'] == rate]
        if len(fed_row):
            fed_means.append(fed_row['Accuracy_Avg'].values[0])
            fed_stds.append(fed_row['Accuracy_Std'].values[0])
        else:
            fed_means.append(np.nan)
            fed_stds.append(np.nan)

    x = np.arange(len(rates))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, meta_means, width, yerr=meta_stds, capsize=4,
           label='Metaheuristics (mean across methods)', color='#4C72B0')
    ax.bar(x + width / 2, fed_means, width, yerr=fed_stds, capsize=4,
           label='FedMCCS (baseline)', color='#DD8452')

    ax.set_xlabel('Participation Rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Participation Rate: Metaheuristics vs FedMCCS')
    ax.set_xticks(x)
    ax.set_xticklabels([_rate_label(r) for r in rates])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_fedmccs_fulfillment(df_fedmccs, output_dir,
                              filename='fedmccs_fulfillment.png'):
    """
    Line + bar diagnostic: FedMCCS's selection fulfillment rate (actual
    selected / target ceiling) per participation rate. This is the plot
    that surfaces FedMCCS's core behavioral difference from the
    metaheuristics -- it may under-fill its target as the rate rises,
    since it only admits clients whose predicted resources clear budget.
    """
    df = df_fedmccs.sort_values('Participation_Rate')
    rates = df['Participation_Rate'].values
    fulfillment = df['Fulfillment_Rate_Avg'].values
    fulfillment_std = df['Fulfillment_Rate_Std'].values
    skipped = df['Skipped_Resources_Avg'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.bar([_rate_label(r) for r in rates], fulfillment,
            yerr=fulfillment_std, capsize=4, color='#55A868')
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Target (100%)')
    ax1.set_ylabel('Fulfillment Rate (selected / target)')
    ax1.set_xlabel('Participation Rate')
    ax1.set_title('FedMCCS: Selection Fulfillment Rate')
    ax1.set_ylim(0, 1.15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar([_rate_label(r) for r in rates], skipped, color='#C44E52')
    ax2.set_ylabel('Avg. Clients Skipped (insufficient resources)')
    ax2.set_xlabel('Participation Rate')
    ax2.set_title('FedMCCS: Candidates Excluded by Resource Budget')
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_pareto_with_fedmccs(df_comparison, df_fedmccs, output_dir,
                              filename='pareto_with_fedmccs.png'):
    """
    Scatter: Accuracy vs Energy per selected client, all metaheuristic
    method/fitness_mode/rate combinations plotted as small points, with
    FedMCCS overlaid as larger, distinctly marked points per rate.
    Mirrors the paper's Figure 6 Pareto-frontier framing.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = df_comparison['Selection_Method'].unique()
    cmap = plt.get_cmap('tab10')
    for i, method in enumerate(methods):
        rows = df_comparison[df_comparison['Selection_Method'] == method]
        ax.scatter(rows['Energy_Per_Selected_Client'], rows['Accuracy_Avg'],
                   s=40, alpha=0.6, color=cmap(i % 10), label=method)

    ax.scatter(df_fedmccs['Energy_Per_Selected_Client'], df_fedmccs['Accuracy_Avg'],
               s=180, marker='*', color='black', edgecolor='gold', linewidth=1.5,
               label='FedMCCS', zorder=5)

    for _, row in df_fedmccs.iterrows():
        ax.annotate(_rate_label(row['Participation_Rate']),
                    (row['Energy_Per_Selected_Client'], row['Accuracy_Avg']),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel('Energy per Selected Client (Wh) -- lower is better')
    ax.set_ylabel('Accuracy -- higher is better')
    ax.set_title('Pareto Frontier: Accuracy vs Energy (Metaheuristics vs FedMCCS)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def generate_all_fedmccs_plots(df_comparison, df_fedmccs, output_dir):
    """Convenience wrapper: generate all three FedMCCS comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    paths.append(plot_accuracy_with_fedmccs(df_comparison, df_fedmccs, output_dir))
    paths.append(plot_fedmccs_fulfillment(df_fedmccs, output_dir))
    paths.append(plot_pareto_with_fedmccs(df_comparison, df_fedmccs, output_dir))
    return paths
