# -*- coding: utf-8 -*-
"""
client_heterogeneity_analysis.py - Statistical heterogeneity across FL
clients, addressing Reviewer 1's R1-5 ("report the degree of statistical
heterogeneity (e.g. earth mover's distance or weight divergence) across
clients to better contextualize the non-IID challenge").

Computes two complementary heterogeneity measures directly from
prepared_client_data -- no FL training or rerun required:

  1. Label-distribution EMD (Earth Mover's Distance): for this binary
     classification task, EMD between a client's local label distribution
     and the global label distribution reduces to |p_client(1) -
     p_global(1)| (Zhao et al., 2018, "Federated Learning with Non-IID
     Data", the standard formulation for 2-class EMD in the FL
     literature).

  2. Feature-space heterogeneity (weight divergence proxy): the L2
     distance between each client's mean feature vector and the global
     mean feature vector, computed on the same standardized 46-dimensional
     feature space used throughout the paper (Section 3.1.1). This
     captures heterogeneity beyond label imbalance -- e.g. two clients
     could have identical label ratios but systematically different
     signal characteristics (different resting heart rates, activity
     patterns, etc.), which label-only EMD would miss entirely.

Both measures are reported per-client and summarized (mean/std/min/max),
matching the format used throughout the paper's other tables.

Usage:
    python client_heterogeneity_analysis.py --csv-dir ... --demographics-path ...
    python client_heterogeneity_analysis.py --dry-run   # test with synthetic data
"""

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_label_emd(prepared_client_data):
    """
    Binary-class EMD per client: |p_client(label=1) - p_global(label=1)|.
    prepared_client_data: dict of {client_id: {'X': features, 'y': labels}}
    (or equivalent structure with a 'y' array of 0/1 labels per client).
    """
    all_labels = []
    client_label_props = {}

    for cid, data in prepared_client_data.items():
        y = np.asarray(data['y'] if isinstance(data, dict) else data[1])
        all_labels.extend(y.tolist())
        client_label_props[cid] = (y.sum(), len(y))  # (n_positive, n_total)

    all_labels = np.array(all_labels)
    p_global = all_labels.mean()

    emds = {}
    for cid, (n_pos, n_total) in client_label_props.items():
        p_client = n_pos / n_total if n_total > 0 else 0.0
        emds[cid] = abs(p_client - p_global)

    return emds, p_global


def compute_feature_divergence(prepared_client_data):
    """
    L2 distance between each client's mean feature vector and the global
    mean feature vector, on standardized features (weight-divergence proxy).
    """
    all_features = []
    client_features = {}

    for cid, data in prepared_client_data.items():
        X = np.asarray(data['X'] if isinstance(data, dict) else data[0])
        all_features.append(X)
        client_features[cid] = X

    global_mean = np.concatenate(all_features, axis=0).mean(axis=0)

    divergences = {}
    for cid, X in client_features.items():
        client_mean = X.mean(axis=0)
        divergences[cid] = float(np.linalg.norm(client_mean - global_mean))

    return divergences


def _synthetic_prepared_client_data(n_clients=32, n_features=46, seed=0):
    """Synthetic prepared_client_data for --dry-run testing only, roughly
    mimicking the real label-imbalance pattern (client 3 as an outlier)."""
    rng = np.random.RandomState(seed)
    data = {}
    for cid in range(1, n_clients + 1):
        n_samples = rng.randint(5, 21)
        p_pos = 0.716 if cid != 3 else 0.50  # mimic the real outlier client
        y = (rng.rand(n_samples) < p_pos).astype(int)
        X = rng.normal(loc=cid * 0.02, scale=1.0, size=(n_samples, n_features))
        data[cid] = {'X': X, 'y': y}
    return data


def run_heterogeneity_analysis(prepared_client_data):
    print("=" * 70)
    print("CLIENT STATISTICAL HETEROGENEITY ANALYSIS")
    print("=" * 70)

    emds, p_global = compute_label_emd(prepared_client_data)
    divergences = compute_feature_divergence(prepared_client_data)

    emd_vals = np.array(list(emds.values()))
    div_vals = np.array(list(divergences.values()))

    print(f"\nGlobal positive-class proportion: {p_global:.4f}")
    print(f"\n--- Label-distribution EMD (n={len(emd_vals)} clients) ---")
    print(f"Mean: {emd_vals.mean():.4f}  Std: {emd_vals.std():.4f}  "
          f"Min: {emd_vals.min():.4f}  Max: {emd_vals.max():.4f}  "
          f"Median: {np.median(emd_vals):.4f}")

    print(f"\n--- Feature-space divergence (L2, n={len(div_vals)} clients) ---")
    print(f"Mean: {div_vals.mean():.4f}  Std: {div_vals.std():.4f}  "
          f"Min: {div_vals.min():.4f}  Max: {div_vals.max():.4f}  "
          f"Median: {np.median(div_vals):.4f}")

    return {'emds': emds, 'divergences': divergences, 'p_global': p_global}


def save_results_table(results, output_csv='client_heterogeneity_results.csv'):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Client_ID', 'Label_EMD', 'Feature_Divergence_L2'])
        for cid in results['emds']:
            writer.writerow([cid, f"{results['emds'][cid]:.4f}",
                              f"{results['divergences'].get(cid, float('nan')):.4f}"])
    print(f"\n\u2713 Saved {output_csv}")


def plot_heterogeneity(results, output_path='client_heterogeneity.png'):
    emds = results['emds']
    divergences = results['divergences']
    client_ids = sorted(emds.keys())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    vals = [emds[c] for c in client_ids]
    ax.bar(range(len(client_ids)), vals, color='#1f77b4', alpha=0.85)
    ax.axhline(np.mean(vals), color='black', linestyle='--', linewidth=1,
               label=f'Mean = {np.mean(vals):.3f}')
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Label EMD', fontsize=12)
    ax.set_title('Label-Distribution Heterogeneity (EMD)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels(client_ids, fontsize=7, rotation=90)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    vals2 = [divergences[c] for c in client_ids]
    ax.bar(range(len(client_ids)), vals2, color='#d62728', alpha=0.85)
    ax.axhline(np.mean(vals2), color='black', linestyle='--', linewidth=1,
               label=f'Mean = {np.mean(vals2):.3f}')
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Feature Divergence (L2)', fontsize=12)
    ax.set_title('Feature-Space Heterogeneity', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels(client_ids, fontsize=7, rotation=90)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Per-Client Statistical Heterogeneity', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\u2713 Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Client statistical heterogeneity analysis")
    parser.add_argument('--csv-dir', default=None)
    parser.add_argument('--demographics-path', default=None)
    parser.add_argument('--output-prefix', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.dry_run:
        prepared_client_data = _synthetic_prepared_client_data()
    else:
        import torch
        from FederatedLearningBS import SensSmartTechLoader, FeatureExtractor
        from ResultBS import MultiMethodExperiment

        assert args.csv_dir and args.demographics_path, (
            "--csv-dir and --demographics-path are required unless --dry-run is set."
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = SensSmartTechLoader(args.csv_dir, args.demographics_path)
        volunteer_data = loader.load_all_data()

        extractor = FeatureExtractor()
        sample = list(list(volunteer_data.values())[0]['recordings'].values())[0]
        n_features = len(extractor.extract_all_features(sample))
        print(f"\u2713 {n_features} features")

        experiment = MultiMethodExperiment(n_features, device)
        experiment.prepare_data_once(volunteer_data, extractor)

        def _to_numpy(x):
            """Handles numpy arrays, CPU tensors, and CUDA tensors uniformly."""
            if hasattr(x, 'detach'):  # torch.Tensor
                return x.detach().cpu().numpy()
            return np.asarray(x)

        # Reconstruct {client_id: {'X':..., 'y':...}} from the real keys used by
        # EnhancedFLSystem.prepare_client_data(): X_train/y_train/X_test/y_test.
        # Combined here to recover each client's full original label/feature
        # distribution (matches the per-subject counts printed during loading).
        prepared_client_data = {}
        for cid, cdata in experiment.prepared_client_data.items():
            X_train = _to_numpy(cdata['X_train'])
            X_test = _to_numpy(cdata['X_test'])
            y_train = _to_numpy(cdata['y_train'])
            y_test = _to_numpy(cdata['y_test'])
            X_full = np.concatenate([X_train, X_test], axis=0)
            y_full = np.concatenate([y_train, y_test], axis=0)
            prepared_client_data[cid] = {'X': X_full, 'y': y_full}

    results = run_heterogeneity_analysis(prepared_client_data)
    save_results_table(results, f'{args.output_prefix}client_heterogeneity_results.csv')
    plot_heterogeneity(results, f'{args.output_prefix}client_heterogeneity.png')

    print("\nDone. Review the CSV and figure before including in the manuscript.")


if __name__ == "__main__":
    main()
