# -*- coding: utf-8 -*-
"""
weight_ablation_study.py
================================================================================
Zero-weight ABLATION study for the fitness-function weight coefficients
(R3-8, reviewer request #2: "provide an ablation study ... for the weight
values", as distinct from the sensitivity analysis already reported in
Table 5 / weight_ablation_results.csv, which perturbs each weight by ±20%
rather than removing it).

For each of the eight weight coefficients (alpha, beta, gamma, delta,
epsilon, zeta, lambda, mu), this script:
  1. sets that weight to 0.0,
  2. proportionally redistributes its mass across the remaining seven
     weights so they still sum to 1.0 (same convention already used for
     the ±20% perturbation study, for direct comparability),
  3. runs N_RUNS full federated-learning trainings with that weight
     configuration (GWO, Resource fitness mode, 20% participation --
     matching the existing sensitivity study's fixed context),
  4. records final accuracy and total energy consumption.

A 'baseline' configuration (all eight weights at their reported defaults)
is run the same number of times for comparison.

Usage: drop this file alongside BioSecureV3.py / FederatedLearningBS.py /
MetaHeuristicsSelector_multi.py / OptimizersV2.py / DynaEdgeSimV3.py /
ResultBS.py, edit CSV_DIR / DEMOGRAPHICS_PATH / EDGE_SIM_FILE below to
match your paths, then run: python weight_ablation_study.py

Output: weight_ablation_ZEROED_results.csv, with the same column schema
as the existing weight_ablation_results.csv (Weight, Direction,
Weight_Value, Accuracy, Energy, Delta_Accuracy_pp, Delta_Energy_pct),
using Direction='zeroed' and Weight_Value=0.0 for the eight ablated rows
and Direction='baseline' for the reference row -- so it can be
concatenated with the existing CSV for joint reporting.
================================================================================
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import copy
import numpy as np
import pandas as pd
import torch

from FederatedLearningBS import (
    SensSmartTechLoader,
    FeatureExtractor,
    EnhancedFLSystem,
    FLClientWrapper,
)
from MetaHeuristicsSelector_multi import ClientSelector as MetaHeuristicSelector
from ResultBS import MultiMethodExperiment
from DynaEdgeSimV3 import Environment

# ==============================================================================
# CONFIG -- edit these to match your environment (mirrors BioSecureV3.py)
# ==============================================================================
CSV_DIR = os.environ.get("BIOSECURE_CSV_DIR", os.path.join("dataset", "csv"))
DEMOGRAPHICS_PATH = os.environ.get("BIOSECURE_DEMOGRAPHICS", os.path.join("dataset", "Demographics.csv"))
EDGE_SIM_FILE = "mobile_sim_32C_100R_V3.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SELECTION_METHOD = 'GWO'          # matches the fixed context of the existing
FITNESS_MODE = 'resource'         # ±20% sensitivity study (Table 5)
PARTICIPATION_RATE = 0.20

N_RUNS = 5                       # match N_RUNS used elsewhere for comparability;
N_ROUNDS = 100                    # reduce both if a faster smoke-test run is needed
LOCAL_EPOCHS = 3
LR = 0.001
MU = 0.2

OUTPUT_CSV = "weight_ablation_ZEROED_results.csv"

DEFAULT_WEIGHTS = {
    'alpha': 0.10, 'beta': 0.25, 'gamma': 0.20, 'delta': 0.15,
    'epsilon': 0.10, 'zeta': 0.05, 'lambda': 0.10, 'mu': 0.05,
}
WEIGHT_LABELS = {
    'alpha': 'Learning Utility', 'beta': 'Battery Level', 'gamma': 'Compute Capacity',
    'delta': 'Reliability', 'epsilon': 'Latency', 'zeta': 'Energy Cost',
    'lambda': 'Data Diversity', 'mu': 'Freshness',
}


def zeroed_weight_config(weight_to_zero):
    """
    Return a full 8-weight dict with `weight_to_zero` set to 0.0 and its
    mass redistributed proportionally across the other seven, preserving
    sum == 1.0. Matches the redistribution convention already used for
    the ±20% perturbation study.
    """
    cfg = dict(DEFAULT_WEIGHTS)
    removed_mass = cfg[weight_to_zero]
    cfg[weight_to_zero] = 0.0

    remaining_keys = [k for k in cfg if k != weight_to_zero]
    remaining_total = sum(cfg[k] for k in remaining_keys)
    for k in remaining_keys:
        # proportional redistribution: each remaining weight grows in
        # proportion to its own current share of the remaining mass
        cfg[k] = cfg[k] + removed_mass * (cfg[k] / remaining_total)

    assert abs(sum(cfg.values()) - 1.0) < 1e-9, f"weights do not sum to 1: {cfg}"
    return cfg


class AblationFLSystem(EnhancedFLSystem):
    """
    Thin subclass that injects a fixed, externally-specified weight
    configuration into every MetaHeuristicSelector constructed during
    training, instead of leaving it at ClientSelector's hardcoded
    defaults. Overrides only client-selection method construction;
    everything else (training loop, aggregation, evaluation) is
    inherited unchanged from EnhancedFLSystem.
    """

    def __init__(self, *args, weight_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        # None => use ClientSelector's built-in defaults (baseline run)
        self.weight_config = weight_config

    def _select_clients_metaheuristic(self, method_name, participation_rate, show_progress=False):
        client_wrappers = list(self.fl_clients.values())
        k = max(1, int(participation_rate * len(client_wrappers)))

        if self.edge_env is not None:
            self._map_clients_to_devices()

        selector = MetaHeuristicSelector(
            clients_pool=client_wrappers,
            k=k,
            env=self.edge_env,
            fitness_mode=self.fitness_mode,
        )

        if self.weight_config is not None:
            selector.set_weights(**self.weight_config)

        configs = selector.get_all_optimizer_configs()
        if method_name.upper() in configs:
            selector.optimizer_config = configs[method_name.upper()]
        else:
            selector.optimizer_config = configs.get(
                "RSELECT", {"method": "RSELECT", "params": {}, "is_discrete": True}
            )

        metrics, selected_clients, curve, fitness = selector.select_clients(
            client_wrappers, return_curve=True
        )
        n_selected = len(selected_clients)

        self.training_history['selection_metrics'].append(metrics)
        if curve is not None and len(curve) > 0:
            self.training_history['metaheuristic_curves'].append(curve)
        selected_ids = [c.client_id for c in selected_clients]
        self.training_history['selected_client_ids_per_round'].append(selected_ids)
        return selected_ids, fitness


def run_single_run(run_idx, weight_config, prepared_client_data, actual_n_features,
                    shared_edge_env):
    seed = 42 + run_idx * 100
    np.random.seed(seed)
    torch.manual_seed(seed)

    fl = AblationFLSystem(actual_n_features, DEVICE, fitness_mode=FITNESS_MODE,
                           weight_config=weight_config)
    fl.client_data = prepared_client_data

    for subj_id in fl.client_data:
        fl.client_alphas[subj_id] = 0.5
        fl.fl_clients[subj_id] = FLClientWrapper(subj_id, fl.client_data[subj_id])

    fl.initialize_models()

    if shared_edge_env is not None:
        fl.edge_env = copy.deepcopy(shared_edge_env)
        fl.edge_env.SimMoveFirst()
        fl._map_clients_to_devices()

    fl.train_federated(
        selection_method=SELECTION_METHOD,
        participation_rate=PARTICIPATION_RATE,
        n_rounds=N_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        mu=MU,
        verbose=False,
        show_progress=True,
        progress_position=1,
        leave_progress=False,
    )

    fl.personalize_models(verbose=False)
    fl.learn_adaptive_alpha(verbose=False)
    metrics = fl.evaluate_all(verbose=False)

    accuracy = metrics['combined_adaptive']['accuracy']
    energy = fl.training_history['total_energy']
    return accuracy, energy


def main():
    print("=" * 80)
    print("WEIGHT ABLATION STUDY (zero-out, R3-8)")
    print(f"Method={SELECTION_METHOD}  FitnessMode={FITNESS_MODE}  "
          f"Rate={PARTICIPATION_RATE}  Runs/config={N_RUNS}  Rounds/run={N_ROUNDS}")
    print("=" * 80)

    # ---- Load + preprocess data once, exactly like run_participation_rate_comparison ----
    loader = SensSmartTechLoader(CSV_DIR, DEMOGRAPHICS_PATH)
    volunteer_data = loader.load_all_data()

    extractor = FeatureExtractor()
    sample = list(list(volunteer_data.values())[0]['recordings'].values())[0]
    n_features = len(extractor.extract_all_features(sample))

    base_experiment = MultiMethodExperiment(n_features, DEVICE)
    base_experiment.prepare_data_once(volunteer_data, extractor)
    prepared_client_data = base_experiment.prepared_client_data

    actual_n_features = n_features
    if prepared_client_data:
        first_client = next(iter(prepared_client_data.values()))
        actual_n_features = first_client['X_train'].shape[1]

    shared_edge_env = None
    if EDGE_SIM_FILE:
        shared_edge_env = Environment()
        shared_edge_env.load_simulation_data(EDGE_SIM_FILE)
        print(f"Loaded edge simulation from {EDGE_SIM_FILE}")

    # ---- Build the configuration list: baseline + one zero-out per weight ----
    configs = [('baseline', None, None)]  # (label, weight_key_or_None, weight_config)
    for wkey in DEFAULT_WEIGHTS:
        configs.append((wkey, wkey, zeroed_weight_config(wkey)))

    rows = []
    baseline_acc_mean = None
    baseline_energy_mean = None

    for label, wkey, weight_config in configs:
        print(f"\n--- Configuration: {label} ---")
        if weight_config is not None:
            print(f"  Weights: {weight_config}")

        accs, energies = [], []
        for run_idx in range(N_RUNS):
            acc, energy = run_single_run(
                run_idx, weight_config, prepared_client_data, actual_n_features,
                shared_edge_env
            )
            accs.append(acc)
            energies.append(energy)
            print(f"  run {run_idx + 1}/{N_RUNS}: acc={acc:.4f}  energy={energy:.2f}")

        acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
        energy_mean, energy_std = float(np.mean(energies)), float(np.std(energies))

        if label == 'baseline':
            baseline_acc_mean = acc_mean
            baseline_energy_mean = energy_mean
            delta_acc_pp = 0.0
            delta_energy_pct = 0.0
        else:
            delta_acc_pp = (acc_mean - baseline_acc_mean) * 100
            delta_energy_pct = (
                (energy_mean - baseline_energy_mean) / baseline_energy_mean * 100
                if baseline_energy_mean else float('nan')
            )

        rows.append({
            'Weight': label,
            'Weight_Label': WEIGHT_LABELS.get(wkey, 'baseline'),
            'Direction': 'baseline' if label == 'baseline' else 'zeroed',
            'Weight_Value': 0.0 if wkey is not None else '-',
            'Accuracy': round(acc_mean, 4),
            'Accuracy_Std': round(acc_std, 4),
            'Energy': round(energy_mean, 2),
            'Energy_Std': round(energy_std, 2),
            'Delta_Accuracy_pp': round(delta_acc_pp, 2),
            'Delta_Energy_pct': round(delta_energy_pct, 2),
        })

        # write incrementally so partial progress survives a crash/interrupt
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"DONE. Results saved to {OUTPUT_CSV}")
    print("=" * 80)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
