# -*- coding: utf-8 -*-
"""
weight_sensitivity_ablation.py - Fitness-weight sensitivity/ablation study,
addressing Reviewer 1's R1-1 ("sensitivity analysis for fixed fitness
weights; show Pareto front shift across weight profiles") and Reviewer 3's
R3-10 ("justify fitness weight values; explain normalization; ablation/
sensitivity study on weights").

METHODOLOGY
-----------
For each of the 8 fitness-function weights (alpha, beta, gamma, delta,
epsilon, zeta, lambda, mu -- see MetaHeuristicsSelector_multi.py), this
script:
  1. Perturbs that ONE weight by +20% and -20% from its baseline value.
  2. Rescales the OTHER 7 weights proportionally so the total still sums to
     the original total. This is the standard convention for sensitivity
     analysis on a fixed-budget weighted sum: without rescaling, a +20%
     perturbation would silently increase the fitness function's total
     "mass" rather than testing a genuine trade-off between objectives.
     State this rescaling rule explicitly in the paper's methods text.
  3. Runs the FL pipeline (GWO -- the method retained in Part II, per the
     Step 6 decision) at ONE representative participation rate for N_RUNS
     runs, and records final accuracy and total energy consumption.
  4. Reports the accuracy/energy shift relative to the unperturbed baseline
     for every weight, at both +20% and -20%.

Scope note: this is deliberately run at a single participation rate (not
the full 4-rate grid) and for a single fitness mode, to keep the total
experiment count tractable: 8 weights x 2 directions = 16 extra
configurations, each run N_RUNS times, vs. a full-grid sweep which would
multiply this by 4 rates x 3 modes.

IMPORTANT -- READ BEFORE RUNNING:
MetaHeuristicsSelector_multi.py's `fitness_mode` parameter is currently
stored (self.fitness_mode) but is NOT used anywhere to change self.weights
-- every fitness mode uses the same single default weight dict unless
something external calls .set_weights(...) per mode. If your pipeline DOES
set mode-specific weights elsewhere (e.g. in FederatedLearningBS.py or
wherever MetaHeuristicSelector is instantiated), pass that mode's baseline
weights into BASELINE_WEIGHTS below instead of the class default. If it
does NOT, this ablation is being run on the one weight set that is
actually in effect regardless of mode -- which may itself be worth noting
in the paper as the root cause of the near-identical Resource/Learning/
Exploration results reported in Section 4.1.4.

Usage:
    python weight_sensitivity_ablation.py --dry-run     # test plotting/logic with synthetic data
    python weight_sensitivity_ablation.py                # full run (needs FL pipeline wired in, see run_configuration())
"""

import argparse
import os
import copy
import csv
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: the class is named ClientSelector in this file (not
# MetaHeuristicSelector, despite the module's filename) -- imported and
# aliased here for clarity in the rest of this script.
from MetaHeuristicsSelector_multi import ClientSelector as MetaHeuristicSelector
# EnhancedFLSystem (and its torch dependency) is imported lazily inside
# run_configuration() instead of at module level, so --dry-run works even
# in an environment without torch/the full FL stack installed.


# =============================================================================
# 1. Baseline weights and perturbation generation
# =============================================================================

# Copied from MetaHeuristicSelector.__init__ -- update here if your actual
# per-mode weights differ from the class default (see module docstring).
BASELINE_WEIGHTS = {
    'alpha':   0.10,   # Learning utility
    'beta':    0.25,   # Battery level
    'gamma':   0.20,   # Compute capacity
    'delta':   0.15,   # Reliability
    'epsilon': 0.10,   # Latency (negative in fitness)
    'zeta':    0.05,   # Energy cost (negative in fitness)
    'lambda':  0.10,   # Data diversity
    'mu':      0.05,   # Freshness
}

WEIGHT_LABELS = {
    'alpha': 'Learning Utility (\u03b1)',
    'beta': 'Battery Level (\u03b2)',
    'gamma': 'Compute Capacity (\u03b3)',
    'delta': 'Reliability (\u03b4)',
    'epsilon': 'Latency (\u03b5)',
    'zeta': 'Energy Cost (\u03b6)',
    'lambda': 'Data Diversity (\u03bb)',
    'mu': 'Freshness (\u03bc)',
}


def perturb_weight(baseline, weight_name, direction, magnitude=0.20):
    """
    Perturb ONE weight by +/- `magnitude` (default 20%), rescaling the
    other weights proportionally so the total sum is preserved.

    Args:
        baseline: dict of baseline weights (e.g. BASELINE_WEIGHTS).
        weight_name: which key to perturb.
        direction: +1 or -1.
        magnitude: fractional perturbation size (0.20 = 20%).

    Returns:
        dict: new weight configuration, same keys as baseline, summing to
        the same total as baseline.
    """
    assert weight_name in baseline, f"Unknown weight: {weight_name}"
    assert direction in (1, -1), "direction must be +1 or -1"

    total = sum(baseline.values())
    new_weights = dict(baseline)

    old_value = baseline[weight_name]
    new_value = old_value * (1 + direction * magnitude)
    delta = new_value - old_value

    # Rescale every OTHER weight down/up proportionally to absorb `delta`,
    # keeping the total constant. Each other weight absorbs a share
    # proportional to its own current value (so a large weight gives up
    # proportionally more than a small one -- avoids driving small weights
    # negative).
    other_keys = [k for k in baseline if k != weight_name]
    other_total = sum(baseline[k] for k in other_keys)

    if other_total <= 0:
        raise ValueError("Cannot rescale: sum of other weights is zero.")

    for k in other_keys:
        share = baseline[k] / other_total
        new_weights[k] = baseline[k] - delta * share

    new_weights[weight_name] = new_value

    # Sanity checks: no negative weights, total preserved
    if any(v < 0 for v in new_weights.values()):
        raise ValueError(
            f"Perturbing {weight_name} by {direction*magnitude:+.0%} drove another "
            f"weight negative. Reduce `magnitude` or exclude this weight from the sweep."
        )
    assert abs(sum(new_weights.values()) - total) < 1e-9, "Total weight not preserved"

    return new_weights


def generate_perturbed_configs(baseline=BASELINE_WEIGHTS, magnitude=0.20):
    """
    Generate all (weight_name, direction, weights_dict) perturbations for
    the full ablation sweep: 8 weights x 2 directions = 16 configs.
    """
    configs = []
    for weight_name in baseline:
        for direction in (-1, 1):
            weights = perturb_weight(baseline, weight_name, direction, magnitude)
            configs.append((weight_name, direction, weights))
    return configs


# =============================================================================
# 2. Running the FL pipeline for one weight configuration
# =============================================================================

def run_configuration(weights, prepared_client_data, prepared_scaler, prepared_fl_clients,
                       n_features, device='cpu', fitness_mode='resource', selection_method='GWO',
                       participation_rate=0.2, n_runs=10, n_rounds=100,
                       local_epochs=5, lr=0.002, mu=0.03,
                       use_edge_sim=True, edge_sim_file=None, seed_offset=0):
    """
    Run the FL pipeline N_RUNS times under one weight configuration and
    return the mean final accuracy and mean total energy across runs.

    Mirrors the exact per-run pattern used in ResultBS.py's
    MultiMethodExperiment.run_experiment() -- same EnhancedFLSystem setup,
    same train_federated() call and history keys ('global_acc',
    'total_energy'), so results here are directly comparable to your main
    experiment grid. `prepared_client_data`/`prepared_scaler`/
    `prepared_fl_clients` should come from calling
    MultiMethodExperiment.prepare_data_once(volunteer_data, extractor)
    ONCE beforehand (see run_ablation_study_from_prepared_data below) --
    reused across all 17 configurations rather than re-extracted every
    time.

    Returns:
        (mean_accuracy, mean_energy, std_accuracy, std_energy)
    """
    import copy
    import torch
    from FederatedLearningBS import EnhancedFLSystem

    accuracies = []
    energies = []

    with override_selector_weights(weights):
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

            final_accuracy = history['global_acc'][-1] if history['global_acc'] else np.nan
            total_energy = history.get('total_energy', 0)

            accuracies.append(final_accuracy)
            energies.append(total_energy)

    return (float(np.mean(accuracies)), float(np.mean(energies)),
            float(np.std(accuracies)), float(np.std(energies)))


class override_selector_weights:
    """
    Context manager that temporarily replaces MetaHeuristicSelector's
    default weights for the duration of the `with` block, then restores
    them -- so a full training run picks up the perturbed weights without
    any change needed inside the FL training loop (which constructs a
    fresh MetaHeuristicSelector every round).
    """

    def __init__(self, weights):
        self.new_weights = dict(weights)
        self._original_init = None

    def __enter__(self):
        self._original_init = MetaHeuristicSelector.__init__
        new_weights = self.new_weights

        def patched_init(self_selector, *args, **kwargs):
            self._original_init(self_selector, *args, **kwargs)
            self_selector.weights = dict(new_weights)

        MetaHeuristicSelector.__init__ = patched_init
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        MetaHeuristicSelector.__init__ = self._original_init


# =============================================================================
# 3. Orchestration: run the full sweep
# =============================================================================

def run_ablation_study(prepared_client_data=None, prepared_scaler=None, prepared_fl_clients=None,
                        n_features=None, device='cpu', baseline=BASELINE_WEIGHTS,
                        magnitude=0.20, dry_run=False, **run_kwargs):
    """
    Run the full weight-sensitivity ablation: baseline + 16 perturbed
    configs (8 weights x 2 directions).

    `prepared_client_data`/`prepared_scaler`/`prepared_fl_clients`/
    `n_features` are required unless dry_run=True -- obtain them by
    calling prepare_data_once() (see main()) ONCE before this, exactly as
    ResultBS.py's MultiMethodExperiment does, so the same prepared client
    pool is reused across all 17 configurations x N_RUNS runs instead of
    being rebuilt each time.

    Returns:
        list of dicts, one per configuration, with keys:
        weight, direction, accuracy, energy, delta_accuracy, delta_energy
    """
    if not dry_run:
        assert prepared_client_data is not None, (
            "prepared_client_data is required when dry_run=False -- call "
            "prepare_data_once() first (see main())."
        )

    print("=" * 70)
    print("FITNESS-WEIGHT SENSITIVITY / ABLATION STUDY")
    print("=" * 70)
    print(f"Baseline weights: {baseline}")
    print(f"Perturbation magnitude: +/-{magnitude:.0%}")
    print()

    def _run(weights, seed_offset):
        if dry_run:
            return (*_synthetic_result(weights, baseline), 0.0, 0.0)
        return run_configuration(
            weights, prepared_client_data, prepared_scaler, prepared_fl_clients,
            n_features, device=device, seed_offset=seed_offset, **run_kwargs
        )

    results = []

    # --- Baseline run ---
    print("Running BASELINE configuration...")
    base_acc, base_energy, _, _ = _run(baseline, seed_offset=0)
    print(f"  Baseline: accuracy={base_acc:.4f}, energy={base_energy:.2f}")
    results.append({
        'weight': 'baseline', 'direction': 0, 'weights': baseline,
        'accuracy': base_acc, 'energy': base_energy,
        'delta_accuracy': 0.0, 'delta_energy': 0.0,
    })

    # --- Perturbed configs ---
    configs = generate_perturbed_configs(baseline, magnitude)
    for cfg_idx, (weight_name, direction, weights) in enumerate(configs, start=1):
        label = f"{weight_name} {'+' if direction > 0 else '-'}{magnitude:.0%}"
        print(f"Running {label}...")
        acc, energy, _, _ = _run(weights, seed_offset=cfg_idx * 1000)
        print(f"  {label}: accuracy={acc:.4f}, energy={energy:.2f}")
        results.append({
            'weight': weight_name, 'direction': direction, 'weights': weights,
            'accuracy': acc, 'energy': energy,
            'delta_accuracy': acc - base_acc,
            'delta_energy': energy - base_energy,
        })

    return results


def _synthetic_result(weights, baseline):
    """Deterministic synthetic accuracy/energy for --dry-run testing only."""
    diff = sum(abs(weights[k] - baseline[k]) for k in baseline)
    rng = np.random.RandomState(int(diff * 10000) % (2**31))
    acc = 0.804 + rng.normal(0, 0.01) - diff * 0.15
    energy = 66.5 + rng.normal(0, 2.0) + diff * 40
    return acc, energy


# =============================================================================
# 4. Reporting: table + tornado plots
# =============================================================================

def save_results_table(results, output_csv='weight_ablation_results.csv'):
    """Save the full results table as CSV, matching the paper's table format:
    Weight | Direction | Baseline | Perturbed | Accuracy | Energy | ΔAcc | ΔEnergy"""
    baseline_row = next(r for r in results if r['weight'] == 'baseline')

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Weight', 'Direction', 'Weight_Value', 'Accuracy', 'Energy',
                          'Delta_Accuracy_pp', 'Delta_Energy_pct'])
        writer.writerow(['baseline', '0', '-', f"{baseline_row['accuracy']:.4f}",
                          f"{baseline_row['energy']:.2f}", '0.00', '0.00'])
        for r in results:
            if r['weight'] == 'baseline':
                continue
            weight_val = r['weights'][r['weight']]
            delta_acc_pp = r['delta_accuracy'] * 100
            delta_energy_pct = (r['delta_energy'] / baseline_row['energy']) * 100
            writer.writerow([
                r['weight'], '+' if r['direction'] > 0 else '-',
                f"{weight_val:.4f}", f"{r['accuracy']:.4f}", f"{r['energy']:.2f}",
                f"{delta_acc_pp:+.2f}", f"{delta_energy_pct:+.2f}",
            ])
    print(f"\n\u2713 Saved results table to {output_csv}")


def plot_tornado(results, metric='delta_accuracy', title=None,
                  ylabel=None, output_path=None, pct_scale=1.0, unit=''):
    """
    Tornado (diverging horizontal bar) plot -- the standard visualization
    for one-at-a-time sensitivity analysis. Each weight gets one row; the
    bar spans from the -20% outcome to the +20% outcome, sorted by total
    swing (most sensitive weight at top).
    """
    weight_names = [w for w in BASELINE_WEIGHTS]
    rows = []
    for w in weight_names:
        neg = next(r for r in results if r['weight'] == w and r['direction'] == -1)
        pos = next(r for r in results if r['weight'] == w and r['direction'] == 1)
        neg_val = neg[metric] * pct_scale
        pos_val = pos[metric] * pct_scale
        swing = abs(pos_val - neg_val)
        rows.append((w, neg_val, pos_val, swing))

    rows.sort(key=lambda r: r[3])  # ascending swing -> most sensitive ends up at top after plotting

    labels = [WEIGHT_LABELS[r[0]] for r in rows]
    neg_vals = [r[1] for r in rows]
    pos_vals = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(labels))

    for i, (neg_v, pos_v) in enumerate(zip(neg_vals, pos_vals)):
        lo, hi = min(neg_v, pos_v), max(neg_v, pos_v)
        color_neg = '#d62728' if neg_v < 0 else '#2ca02c'
        color_pos = '#d62728' if pos_v < 0 else '#2ca02c'
        ax.barh(i, neg_v, color='#1f77b4', alpha=0.75, height=0.6, label='-20%' if i == 0 else None)
        ax.barh(i, pos_v, color='#ff7f0e', alpha=0.75, height=0.6, label='+20%' if i == 0 else None)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(ylabel or metric, fontsize=13)
    ax.set_title(title or f'Weight Sensitivity: {metric}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

    for i, (neg_v, pos_v) in enumerate(zip(neg_vals, pos_vals)):
        ax.text(neg_v, i, f' {neg_v:+.2f}{unit}', va='center',
                 ha='right' if neg_v < 0 else 'left', fontsize=9)
        ax.text(pos_v, i, f' {pos_v:+.2f}{unit}', va='center',
                 ha='right' if pos_v < 0 else 'left', fontsize=9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\u2713 Saved {output_path}")
    plt.close(fig)


def plot_combined_summary(results, output_path='weight_sensitivity_summary.png'):
    """Two-panel figure: accuracy tornado (left) + energy tornado (right),
    the figure to embed directly in the paper for R1-1/R3-10."""
    weight_names = [w for w in BASELINE_WEIGHTS]
    baseline_row = next(r for r in results if r['weight'] == 'baseline')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, metric, title, unit, scale in [
        (axes[0], 'delta_accuracy', 'Accuracy Sensitivity', ' pp', 100),
        (axes[1], 'delta_energy', 'Energy Sensitivity', '%', None),
    ]:
        rows = []
        for w in weight_names:
            neg = next(r for r in results if r['weight'] == w and r['direction'] == -1)
            pos = next(r for r in results if r['weight'] == w and r['direction'] == 1)
            if metric == 'delta_energy':
                neg_val = (neg[metric] / baseline_row['energy']) * 100
                pos_val = (pos[metric] / baseline_row['energy']) * 100
            else:
                neg_val = neg[metric] * scale
                pos_val = pos[metric] * scale
            rows.append((w, neg_val, pos_val, abs(pos_val - neg_val)))
        rows.sort(key=lambda r: r[3])

        labels = [WEIGHT_LABELS[r[0]] for r in rows]
        neg_vals = [r[1] for r in rows]
        pos_vals = [r[2] for r in rows]
        y_pos = np.arange(len(labels))

        for i, (neg_v, pos_v) in enumerate(zip(neg_vals, pos_vals)):
            ax.barh(i, neg_v, color='#1f77b4', alpha=0.8, height=0.6, label='-20%' if i == 0 else None)
            ax.barh(i, pos_v, color='#ff7f0e', alpha=0.8, height=0.6, label='+20%' if i == 0 else None)

        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(f'\u0394 {title.split()[0]} ({unit.strip()})', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Fitness-Weight Sensitivity Analysis (\u00b120% Perturbation, GWO)',
                  fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\u2713 Saved {output_path}")
    plt.close(fig)


# =============================================================================
# 5. Entry point
# =============================================================================

def main():
    CSV_DIR = os.environ.get("BIOSECURE_CSV_DIR", os.path.join("dataset", "csv"))
    DEMOGRAPHICS_PATH = os.environ.get("BIOSECURE_DEMOGRAPHICS", os.path.join("dataset", "Demographics.csv"))
    parser = argparse.ArgumentParser(description="Fitness-weight sensitivity/ablation study")
    parser.add_argument('--magnitude', type=float, default=0.20,
                         help="Perturbation magnitude (default 0.20 = +/-20%%)")
    parser.add_argument('--participation-rate', type=float, default=0.2)
    parser.add_argument('--n-runs', type=int, default=8)
    parser.add_argument('--n-rounds', type=int, default=100)
    parser.add_argument('--fitness-mode', default='resource')
    parser.add_argument('--selection-method', default='GWO')
    parser.add_argument('--csv-dir', default=CSV_DIR, help="Path to SensSmartTech CSV data (required unless --dry-run)")
    parser.add_argument('--demographics-path', default=DEMOGRAPHICS_PATH)
    parser.add_argument('--edge-sim-file', default="mobile_sim_32C_100R_V3.csv")
    parser.add_argument('--output-prefix', default='')
    parser.add_argument('--dry-run', action='store_true',
                         help="Test the ablation logic and plots with synthetic data, "
                              "without needing the real dataset or a GPU/long run.")
    args = parser.parse_args()

    if args.dry_run:
        results = run_ablation_study(dry_run=True, magnitude=args.magnitude)
    else:
        # Mirrors BioSecureV3.py's data-loading block exactly, so this uses
        # the identical client pool as your main experiment grid.
        import torch
        from FederatedLearningBS import SensSmartTechLoader, FeatureExtractor
        from ResultBS import MultiMethodExperiment

        assert args.csv_dir and args.demographics_path, (
            "--csv-dir and --demographics-path are required unless --dry-run "
            "is set (see BioSecureV3.py's CSV_DIR/DEMOGRAPHICS_PATH for your values)."
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        loader = SensSmartTechLoader(args.csv_dir, args.demographics_path)
        volunteer_data = loader.load_all_data()

        extractor = FeatureExtractor()
        sample = list(list(volunteer_data.values())[0]['recordings'].values())[0]
        n_features = len(extractor.extract_all_features(sample))
        print(f"\u2713 {n_features} features")

        # Prepare the client pool ONCE, exactly as ResultBS.py does, then
        # reuse it across all 17 weight configurations x N_RUNS runs.
        experiment = MultiMethodExperiment(n_features, device)
        experiment.prepare_data_once(volunteer_data, extractor)

        results = run_ablation_study(
            prepared_client_data=experiment.prepared_client_data,
            prepared_scaler=experiment.prepared_scaler,
            prepared_fl_clients=experiment.prepared_fl_clients,
            n_features=experiment.n_features,
            device=device,
            magnitude=args.magnitude,
            dry_run=False,
            fitness_mode=args.fitness_mode,
            selection_method=args.selection_method,
            participation_rate=args.participation_rate,
            n_runs=args.n_runs,
            n_rounds=args.n_rounds,
            use_edge_sim=True,
            edge_sim_file=args.edge_sim_file,
        )

    save_results_table(results, f'{args.output_prefix}weight_ablation_results.csv')
    plot_combined_summary(results, f'{args.output_prefix}weight_sensitivity_summary.png')
    plot_tornado(results, metric='delta_accuracy', title='Weight Sensitivity: Accuracy',
                 ylabel='\u0394 Accuracy (fraction)',
                 output_path=f'{args.output_prefix}tornado_accuracy.png')
    plot_tornado(results, metric='delta_energy', title='Weight Sensitivity: Energy',
                 ylabel='\u0394 Energy (Wh)',
                 output_path=f'{args.output_prefix}tornado_energy.png')

    print("\nDone. Review the CSV and figures before including in the manuscript.")


if __name__ == "__main__":
    main()
