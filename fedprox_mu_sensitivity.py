# -*- coding: utf-8 -*-
"""
fedprox_mu_sensitivity.py - FedProx proximal coefficient (mu) sensitivity
study, addressing Reviewer 1's R1-9 ("justify the FedProx mu=0.2 value;
sensitivity analysis on mu").

METHODOLOGY
-----------
Sweeps FedProx's proximal term coefficient mu across a range spanning
"no regularization" (mu=0, equivalent to vanilla FedAvg local training)
up to strong regularization (mu=1.0), holding client selection method,
participation rate, and all other hyperparameters fixed at the paper's
main-experiment values. For each mu, runs the FL pipeline N_RUNS times
and records final accuracy, final loss, and convergence round (first
round reaching within 1pp of the run's own final accuracy, as a proxy
for convergence speed).

Unlike the fitness-weight ablation (weight_sensitivity_ablation.py),
this requires no monkeypatching: mu is already an explicit keyword
argument to EnhancedFLSystem.train_federated(), so each configuration is
just a direct call with a different mu value.

Scope: run on GWO (the method retained for Part II, per Step 6) at one
representative participation rate, matching the same scope-control
rationale used for the weight-sensitivity study.

Usage:
    python fedprox_mu_sensitivity.py --dry-run     # test plotting/logic with synthetic data
    python fedprox_mu_sensitivity.py --csv-dir ... --demographics-path ...   # full run
"""

import argparse
import os
import csv
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# 1. Sweep configuration
# =============================================================================

# mu=0.2 is the paper's reported baseline (see Section 3.1.3 / hyperparameter
# statement). The sweep brackets it on both sides: 0 (no proximal term) up to
# 1.0 (strong regularization), which is the conventional range explored in
# the FedProx literature (Li et al., 2020) rather than a narrow +/-20% band --
# mu behaves more like a regularization strength than a weight in a fixed-sum
# combination, so a wide sweep is more informative here than a tight one.
MU_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
BASELINE_MU = 0.2


# =============================================================================
# 2. Running the FL pipeline for one mu value
# =============================================================================

def run_configuration(mu, prepared_client_data, prepared_scaler, prepared_fl_clients,
                       n_features, device='cpu', selection_method='GWO',
                       fitness_mode='resource', participation_rate=0.2,
                       n_runs=10, n_rounds=100, local_epochs=5, lr=0.002,
                       use_edge_sim=True, edge_sim_file=None, seed_offset=0):
    """
    Run the FL pipeline N_RUNS times at one mu value and return summary
    statistics. No monkeypatching needed -- mu is passed directly to
    train_federated().

    Returns:
        dict with mean/std of final accuracy, final loss, and convergence
        round, plus the raw per-run values for later inspection.
    """
    import copy
    import torch
    from FederatedLearningBS import EnhancedFLSystem

    accuracies, losses, convergence_rounds = [], [], []

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

        acc_hist = history['global_acc']
        loss_hist = history.get('global_loss', history.get('global_loss_history', []))
        final_acc = acc_hist[-1] if acc_hist else np.nan
        final_loss = loss_hist[-1] if loss_hist else np.nan

        # Convergence round: first round within 1pp of this run's own final accuracy
        conv_round = len(acc_hist)
        if acc_hist:
            target = final_acc - 0.01
            for i, a in enumerate(acc_hist):
                if a >= target:
                    conv_round = i + 1
                    break

        accuracies.append(final_acc)
        losses.append(final_loss)
        convergence_rounds.append(conv_round)

    return {
        'mu': mu,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'mean_loss': float(np.nanmean(losses)) if losses else float('nan'),
        'std_loss': float(np.nanstd(losses)) if losses else float('nan'),
        'mean_convergence_round': float(np.mean(convergence_rounds)),
        'std_convergence_round': float(np.std(convergence_rounds)),
        'raw_accuracies': accuracies,
        'raw_losses': losses,
        'raw_convergence_rounds': convergence_rounds,
    }


def _synthetic_result(mu, n_runs=10):
    """Deterministic synthetic result for --dry-run testing only.
    Models a plausible mu-vs-accuracy curve: mild benefit near mu=0.1-0.3,
    degradation at high mu (over-regularization slows/limits learning),
    with mu=0 showing higher variance (no heterogeneity correction)."""
    rng = np.random.RandomState(int(mu * 1000) + 1)
    peak = 0.2
    acc_center = 0.81 - 0.35 * (mu - peak) ** 2 if mu >= peak else 0.81 - 0.15 * (mu - peak) ** 2
    acc_center = max(acc_center, 0.55)
    noise_scale = 0.02 if mu > 0 else 0.05  # mu=0 is noisier (no drift correction)
    accs = list(np.clip(rng.normal(acc_center, noise_scale, n_runs), 0.3, 0.95))
    losses = list(np.clip(rng.normal(0.65 - mu * 0.05, 0.03, n_runs), 0.3, 1.2))
    conv = list(np.clip(rng.normal(45 - mu * 10, 8, n_runs), 10, 100))
    return {
        'mu': mu,
        'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
        'mean_loss': float(np.mean(losses)), 'std_loss': float(np.std(losses)),
        'mean_convergence_round': float(np.mean(conv)), 'std_convergence_round': float(np.std(conv)),
        'raw_accuracies': accs, 'raw_losses': losses, 'raw_convergence_rounds': conv,
    }


# =============================================================================
# 3. Orchestration
# =============================================================================

def run_mu_sensitivity_study(prepared_client_data=None, prepared_scaler=None,
                              prepared_fl_clients=None, n_features=None, device='cpu',
                              mu_values=MU_VALUES, dry_run=False, **run_kwargs):
    """Sweep all mu values, with an overall progress/ETA tracker matching
    the style used in weight_sensitivity_ablation.py."""
    print("=" * 70)
    print("FEDPROX MU (PROXIMAL COEFFICIENT) SENSITIVITY STUDY")
    print("=" * 70)
    print(f"Sweep values: {mu_values}")
    print(f"Baseline (paper-reported): mu={BASELINE_MU}")
    print()

    n_runs = run_kwargs.get('n_runs', 10)
    total_configs = len(mu_values)
    total_runs = total_configs * n_runs
    start_time = time.time()

    results = []
    for cfg_idx, mu in enumerate(mu_values, start=1):
        label = f"mu={mu}" + (" (baseline)" if mu == BASELINE_MU else "")
        print(f"Running {label}...")
        if dry_run:
            r = _synthetic_result(mu, n_runs)
        else:
            r = run_configuration(mu, prepared_client_data, prepared_scaler,
                                  prepared_fl_clients, n_features, device=device,
                                  seed_offset=cfg_idx * 1000, **run_kwargs)
        print(f"  {label}: accuracy={r['mean_accuracy']:.4f} (\u00b1{r['std_accuracy']:.4f}), "
              f"loss={r['mean_loss']:.4f}, conv_round={r['mean_convergence_round']:.1f}")

        completed = cfg_idx * n_runs
        ratio = completed / total_runs
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (total_runs - completed) if completed > 0 else float('nan')
        print(f"  [Overall progress: {completed}/{total_runs} runs = {ratio:.1%} | "
              f"configs {cfg_idx}/{total_configs} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m]")

        results.append(r)

    print(f"\n\u2713 Mu sweep complete: {total_runs}/{total_runs} runs "
          f"(100.0%) in {(time.time()-start_time)/60:.1f} minutes.")

    return results


# =============================================================================
# 4. Reporting: table + plot
# =============================================================================

def save_results_table(results, output_csv='fedprox_mu_sensitivity_results.csv'):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Mu', 'Mean_Accuracy', 'Std_Accuracy', 'Mean_Loss', 'Std_Loss',
                          'Mean_Convergence_Round', 'Std_Convergence_Round', 'Is_Baseline'])
        for r in results:
            writer.writerow([
                r['mu'], f"{r['mean_accuracy']:.4f}", f"{r['std_accuracy']:.4f}",
                f"{r['mean_loss']:.4f}", f"{r['std_loss']:.4f}",
                f"{r['mean_convergence_round']:.1f}", f"{r['std_convergence_round']:.1f}",
                'Yes' if r['mu'] == BASELINE_MU else 'No',
            ])
    print(f"\u2713 Saved results table to {output_csv}")


def plot_mu_sensitivity(results, output_path='fedprox_mu_sensitivity.png'):
    """Three-panel figure: accuracy vs mu, loss vs mu, convergence round vs
    mu -- each with error bars and the baseline mu marked, the figure to
    embed directly in the paper for R1-9."""
    mus = [r['mu'] for r in results]
    acc_mean = [r['mean_accuracy'] for r in results]
    acc_std = [r['std_accuracy'] for r in results]
    loss_mean = [r['mean_loss'] for r in results]
    loss_std = [r['std_loss'] for r in results]
    conv_mean = [r['mean_convergence_round'] for r in results]
    conv_std = [r['std_convergence_round'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    panels = [
        (axes[0], acc_mean, acc_std, 'Final Accuracy', '#1f77b4'),
        (axes[1], loss_mean, loss_std, 'Final Loss', '#d62728'),
        (axes[2], conv_mean, conv_std, 'Convergence Round\n(first round within 1pp of final)', '#2ca02c'),
    ]

    for ax, mean_vals, std_vals, ylabel, color in panels:
        ax.errorbar(mus, mean_vals, yerr=std_vals, marker='o', color=color,
                     linewidth=2, capsize=4, markersize=7)
        ax.axvline(BASELINE_MU, color='gray', linestyle='--', linewidth=1.2,
                    label=f'Baseline (\u03bc={BASELINE_MU})')
        ax.set_xlabel('\u03bc (FedProx proximal coefficient)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle('FedProx \u03bc Sensitivity (GWO, defended pipeline)', fontsize=15, fontweight='bold')
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
    parser = argparse.ArgumentParser(description="FedProx mu sensitivity study")
    parser.add_argument('--participation-rate', type=float, default=0.2)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--n-rounds', type=int, default=100)
    parser.add_argument('--fitness-mode', default='resource')
    parser.add_argument('--selection-method', default='GWO')
    parser.add_argument('--csv-dir', default=CSV_DIR)
    parser.add_argument('--demographics-path', default=DEMOGRAPHICS_PATH)
    parser.add_argument('--edge-sim-file', default="mobile_sim_32C_100R_V3.csv")
    parser.add_argument('--output-prefix', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        results = run_mu_sensitivity_study(dry_run=True, n_runs=args.n_runs)
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

        results = run_mu_sensitivity_study(
            prepared_client_data=experiment.prepared_client_data,
            prepared_scaler=experiment.prepared_scaler,
            prepared_fl_clients=experiment.prepared_fl_clients,
            n_features=experiment.n_features,
            device=device,
            dry_run=False,
            fitness_mode=args.fitness_mode,
            selection_method=args.selection_method,
            participation_rate=args.participation_rate,
            n_runs=args.n_runs,
            n_rounds=args.n_rounds,
            use_edge_sim=True,
            edge_sim_file=args.edge_sim_file,
        )

    save_results_table(results, f'{args.output_prefix}fedprox_mu_sensitivity_results.csv')
    plot_mu_sensitivity(results, f'{args.output_prefix}fedprox_mu_sensitivity.png')

    print("\nDone. Review the CSV and figure before including in the manuscript.")


if __name__ == "__main__":
    main()
