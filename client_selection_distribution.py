# -*- coding: utf-8 -*-
"""
client_selection_distribution.py - Per-client selection distribution and
fairness analysis across fitness modes, addressing Reviewer 1's R1-8
("show the per-client selection distribution under Exploration mode;
does it actually achieve more uniform/fair selection than the other
modes, as claimed?").

*** REQUIRES A ONE-LINE CODE CHANGE BEFORE RUNNING -- see PATCH below ***

METHODOLOGY
-----------
Runs GWO (the method retained for Part II, per Step 6) under each of the
three fitness modes (Resource, Learning, Exploration) at one
representative participation rate, N_RUNS times each. For every run,
records which specific client IDs were selected every round, then:
  1. Aggregates into a per-client selection COUNT (how many times each
     of the N clients was selected, summed across all rounds and runs).
  2. Computes Jain's Fairness Index per mode: J = (sum(x_i))^2 /
     (n * sum(x_i^2)), where x_i is client i's total selection count.
     J in (0, 1], with J=1 meaning perfectly uniform selection across
     all clients and lower values indicating selection concentrated on
     a few clients. This directly tests Exploration mode's stated
     design goal (avoid repeatedly selecting the same high-performing
     clients) rather than assuming it from the fitness formula alone.
  3. Produces a bar chart of per-client selection count for all three
     modes side by side, plus a summary table of Jain's index and
     selection-count spread (min/max/std) per mode.

REQUIRED CODE CHANGE (apply once, before running):
In FederatedLearningBS.py:
  (a) Near where self.training_history is initialized (~line 793), add:
        self.training_history['selected_client_ids_per_round'] = []
  (b) Right after `selected_ids` is computed inside the per-round
      selection logic (~line 1391, where you see
      `selected_ids = [c.client_id for c in selected_clients]`), add:
        self.training_history['selected_client_ids_per_round'].append(selected_ids)

Without this change, train_federated()'s returned history has no
per-round client-ID record and this script cannot compute anything --
it will fail loudly (see _check_history_has_client_ids below) rather
than silently producing empty results.

Usage:
    python client_selection_distribution.py --dry-run
    python client_selection_distribution.py --csv-dir ... --demographics-path ...
"""

import argparse
import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES = ['resource', 'learning', 'exploration']
MODE_LABELS = {'resource': 'Resource', 'learning': 'Learning', 'exploration': 'Exploration'}
MODE_COLORS = {'resource': '#2ca02c', 'learning': '#1f77b4', 'exploration': '#d62728'}


# =============================================================================
# 1. Running the FL pipeline and collecting per-round selected client IDs
# =============================================================================

def _check_history_has_client_ids(history):
    if 'selected_client_ids_per_round' not in history or not history['selected_client_ids_per_round']:
        raise RuntimeError(
            "train_federated()'s history has no 'selected_client_ids_per_round' entries. "
            "Apply the one-line instrumentation patch described in this script's module "
            "docstring to FederatedLearningBS.py before running for real (use --dry-run "
            "to test this script's analysis/plotting logic without that patch)."
        )


def run_configuration(fitness_mode, prepared_client_data, prepared_scaler, prepared_fl_clients,
                       n_features, device='cpu', selection_method='GWO',
                       participation_rate=0.2, n_runs=10, n_rounds=100,
                       local_epochs=5, lr=0.002, mu=0.2,
                       use_edge_sim=True, edge_sim_file=None, seed_offset=0):
    """
    Run the FL pipeline N_RUNS times under one fitness mode and return the
    aggregated per-client selection counts across all rounds and runs.

    Returns:
        dict: {client_id: total_selection_count}
    """
    import copy
    import torch
    from FederatedLearningBS import EnhancedFLSystem

    client_counts = {}

    for run in range(n_runs):
        seed = 42 + (seed_offset + run) * 100
        np.random.seed(seed)
        torch.manual_seed(seed)

        fl = EnhancedFLSystem(n_features, device, fitness_mode=fitness_mode)
        fl.client_data = copy.deepcopy(prepared_client_data)
        fl.scaler = copy.deepcopy(prepared_scaler)
        fl.fl_clients = copy.deepcopy(prepared_fl_clients)
        fl.client_alphas = {cid: 0.5 for cid in fl.client_data}

        fl.initialize_models()
        if use_edge_sim:
            fl.setup_edge_simulation(sim_file=edge_sim_file)

        history = fl.train_federated(
            selection_method=selection_method,
            participation_rate=participation_rate,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            lr=lr,
            mu=mu,
            verbose=False,
            show_progress=True,
            progress_position=0,
        )

        _check_history_has_client_ids(history)

        for round_ids in history['selected_client_ids_per_round']:
            for cid in round_ids:
                client_counts[cid] = client_counts.get(cid, 0) + 1

    return client_counts


def _synthetic_client_counts(fitness_mode, n_clients=32, n_rounds=100, n_runs=10, seed=None):
    """Deterministic synthetic per-client counts for --dry-run testing.
    Models Exploration mode as more uniform (lower concentration) than
    Resource/Learning, consistent with its stated design intent -- but
    this is only a plausibility stub; the real run may not confirm it."""
    if seed is None:
        seed = {'resource': 1, 'learning': 2, 'exploration': 3}[fitness_mode]
    rng = np.random.RandomState(seed)
    total_selections = n_rounds * n_runs * 6  # ~6 clients/round assumed for the synthetic stub
    if fitness_mode == 'exploration':
        # closer to uniform: Dirichlet with high concentration parameter
        probs = rng.dirichlet(np.ones(n_clients) * 8)
    else:
        # more concentrated: a few clients favored much more than others
        probs = rng.dirichlet(np.ones(n_clients) * 1.2)
    raw_counts = rng.multinomial(total_selections, probs)
    return {i: int(c) for i, c in enumerate(raw_counts)}


# =============================================================================
# 2. Fairness metric and reporting
# =============================================================================

def jains_fairness_index(counts_dict):
    """J = (sum x_i)^2 / (n * sum x_i^2). 1.0 = perfectly uniform selection."""
    x = np.array(list(counts_dict.values()), dtype=float)
    if len(x) == 0 or np.sum(x**2) == 0:
        return float('nan')
    return float((np.sum(x) ** 2) / (len(x) * np.sum(x ** 2)))


def run_distribution_study(prepared_client_data=None, prepared_scaler=None,
                            prepared_fl_clients=None, n_features=None, device='cpu',
                            dry_run=False, n_clients=32, **run_kwargs):
    print("=" * 70)
    print("PER-CLIENT SELECTION DISTRIBUTION / FAIRNESS STUDY")
    print("=" * 70)

    results = {}
    for mode in MODES:
        print(f"\nRunning fitness_mode={mode}...")
        if dry_run:
            counts = _synthetic_client_counts(mode, n_clients=n_clients,
                                               n_runs=run_kwargs.get('n_runs', 10))
        else:
            counts = run_configuration(mode, prepared_client_data, prepared_scaler,
                                       prepared_fl_clients, n_features, device=device,
                                       **run_kwargs)
        jfi = jains_fairness_index(counts)
        vals = list(counts.values())
        print(f"  {MODE_LABELS[mode]}: Jain's Fairness Index = {jfi:.4f} "
              f"(min={min(vals) if vals else 0}, max={max(vals) if vals else 0}, "
              f"std={np.std(vals):.2f})")
        results[mode] = counts

    return results


# =============================================================================
# 3. Plotting and table export
# =============================================================================

def save_summary_table(results, output_csv='client_selection_fairness_summary.csv'):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fitness_Mode', 'Jains_Fairness_Index', 'Min_Count', 'Max_Count',
                          'Mean_Count', 'Std_Count', 'N_Clients_Selected_At_Least_Once'])
        for mode in MODES:
            counts = results[mode]
            vals = list(counts.values())
            jfi = jains_fairness_index(counts)
            writer.writerow([
                MODE_LABELS[mode], f"{jfi:.4f}",
                min(vals) if vals else 0, max(vals) if vals else 0,
                f"{np.mean(vals):.2f}" if vals else 0, f"{np.std(vals):.2f}" if vals else 0,
                len(counts),
            ])
    print(f"\n\u2713 Saved summary table to {output_csv}")


def plot_client_distribution(results, n_clients=32, output_path='client_selection_distribution.png'):
    """Bar chart: per-client selection count, one subplot per fitness mode,
    the figure to embed directly in the paper for R1-8."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    for ax, mode in zip(axes, MODES):
        counts = results[mode]
        x = list(range(n_clients))
        y = [counts.get(i, 0) for i in x]
        jfi = jains_fairness_index(counts)
        ax.bar(x, y, color=MODE_COLORS[mode], alpha=0.85)
        ax.axhline(np.mean(y), color='black', linestyle='--', linewidth=1,
                   label=f'Mean = {np.mean(y):.1f}')
        ax.set_ylabel('Selection Count', fontsize=12)
        ax.set_title(f"{MODE_LABELS[mode]} Mode (Jain's Fairness Index = {jfi:.3f})",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    axes[-1].set_xlabel('Client ID', fontsize=12)
    fig.suptitle('Per-Client Selection Distribution by Fitness Mode', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\u2713 Saved {output_path}")
    plt.close(fig)


# =============================================================================
# 4. Entry point
# =============================================================================

def main():
    CSV_DIR = os.environ.get("BIOSECURE_CSV_DIR", os.path.join("dataset", "csv"))
    DEMOGRAPHICS_PATH = os.environ.get("BIOSECURE_DEMOGRAPHICS", os.path.join("dataset", "Demographics.csv"))
    parser = argparse.ArgumentParser(description="Per-client selection distribution / fairness study")
    parser.add_argument('--participation-rate', type=float, default=0.2)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--n-rounds', type=int, default=100)
    parser.add_argument('--selection-method', default='GWO')
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--demographics-path', default=DEMOGRAPHICS_PATH)
    parser.add_argument('--edge-sim-file', default="mobile_sim_32C_100R_V3.csv")
    parser.add_argument('--n-clients', type=int, default=32)
    parser.add_argument('--output-prefix', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        results = run_distribution_study(dry_run=True, n_clients=args.n_clients, n_runs=args.n_runs)
    else:
        import torch
        from FederatedLearningBS import SensSmartTechLoader, FeatureExtractor
        from ResultBS import MultiMethodExperiment

        assert args.csv_dir and args.demographics_path, (
            "--csv-dir and --demographics-path are required unless --dry-run is set."
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        loader = SensSmartTechLoader(args.csv_dir, args.demographics_path)
        volunteer_data = loader.load_all_data()

        extractor = FeatureExtractor()
        sample = list(list(volunteer_data.values())[0]['recordings'].values())[0]
        n_features = len(extractor.extract_all_features(sample))
        print(f"\u2713 {n_features} features")

        experiment = MultiMethodExperiment(n_features, device)
        experiment.prepare_data_once(volunteer_data, extractor)

        results = run_distribution_study(
            prepared_client_data=experiment.prepared_client_data,
            prepared_scaler=experiment.prepared_scaler,
            prepared_fl_clients=experiment.prepared_fl_clients,
            n_features=experiment.n_features,
            device=device,
            dry_run=False,
            n_clients=args.n_clients,
            selection_method=args.selection_method,
            participation_rate=args.participation_rate,
            n_runs=args.n_runs,
            n_rounds=args.n_rounds,
            use_edge_sim=True,
            edge_sim_file=args.edge_sim_file,
        )

    save_summary_table(results, f'{args.output_prefix}client_selection_fairness_summary.csv')
    plot_client_distribution(results, n_clients=args.n_clients,
                              output_path=f'{args.output_prefix}client_selection_distribution.png')

    print("\nDone. Review the CSV and figure before including in the manuscript.")


if __name__ == "__main__":
    main()
