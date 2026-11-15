# -*- coding: utf-8 -*-
"""
================================================================================
fedmccs_experiment.py - FedMCCS Baseline Experiment Orchestration
================================================================================
Runs the FedMCCS baseline (fedmccs_selection.py) across participation rates,
producing a CSV schema-compatible with BioSecure.py's
`participation_rate_comparison.csv`, so the two can be concatenated directly
for combined tables/plots (Option A: FedMCCS run once per rate, independent
of the Resource/Learning/Exploration fitness-mode axis).

--------------------------------------------------------------------------
REQUIRED INTEGRATION HOOK (one change needed in FederatedLearningBS.py)
--------------------------------------------------------------------------
`fl.train_federated(selection_method=..., participation_rate=..., ...)`
presumably dispatches client selection internally to
`MetaHeuristicsSelector_multi.ClientSelector` wherever `selection_method`
is looked up (likely inside `EnhancedFLSystem`, in a method such as
`_select_clients` or inline within `train_federated` itself).

FedMCCS needs to be a SECOND branch at that exact dispatch point, since it
is not a metaheuristic and does not go through `Optimizers.py`. Everything
downstream (fitness logging, energy accounting, aggregation, personalization)
should continue to work unchanged, since client selection is presumably a
pluggable step that just needs to return a list of selected clients.

Add something equivalent to this at the dispatch point (exact location and
variable names depend on FederatedLearningBS.py's internals, not available
at the time this file was written -- confirm/adjust once that file can be
reviewed):

    if selection_method == 'FedMCCS':
        from fedmccs_selection import run_fedmccs_baseline
        selected_clients, fedmccs_diag = run_fedmccs_baseline(
            active_clients=active_clients,          # whatever variable
            participation_rate=participation_rate,  # holds this list here
            client_history=self._fedmccs_history,    # see note below
            env=self.edge_env if hasattr(self, 'edge_env') else None,
        )
        # Persist per-round diagnostics for later CSV export:
        if not hasattr(self, 'fedmccs_diagnostics_per_round'):
            self.fedmccs_diagnostics_per_round = []
        self.fedmccs_diagnostics_per_round.append(fedmccs_diag)
        return selected_clients
    else:
        # existing ClientSelector/Optimizers dispatch
        ...

Per-client history for RUPred-LR (`self._fedmccs_history`, a dict
{client_key: {metric_name: [values]}}) should be updated once per round
with each client's current normalized metrics AFTER selection (whether or
not that client was actually selected this round, so the resource trend is
tracked continuously) -- initialize it as `self._fedmccs_history = {}` in
`EnhancedFLSystem.__init__` or immediately before the first
`train_federated` call. If this history isn't wired up, FedMCCS still runs
correctly via the cold-start fallback (current-value-only prediction) --
it simply won't show the paper's predictive behavior. See
fedmccs_selection.py's module docstring, adaptation #3, for why round
index (not dataset size) is regressed against.
================================================================================
"""

import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from FederatedLearningBS import EnhancedFLSystem, FLClientWrapper
from ResultBS import MultiMethodExperiment


def _format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"


def _to_positive_fitness(val):
    return -val if val < 0 else val


def run_fedmccs_baseline_experiment(volunteer_data, extractor, n_features, device,
                                     participation_rates=(0.1, 0.2, 0.3, 0.5),
                                     n_runs=10, n_rounds=100, local_epochs=3,
                                     lr=0.001, mu=0.2, use_edge_sim=True,
                                     edge_sim_file=None, output_dir=None,
                                     output_base_dir='Results_New'):
    """
    Run the FedMCCS baseline across participation rates only (Option A) --
    no fitness_mode axis, since FedMCCS has no native fitness mode.

    Mirrors BioSecure.py's run_participation_rate_comparison structure
    exactly (same seeding pattern, same client-population setup, same
    metric extraction after training) so results are directly comparable.

    Args:
        volunteer_data, extractor, n_features, device: same as
            run_participation_rate_comparison in BioSecure.py.
        participation_rates: iterable of rates, should match the rates
            used for the main metaheuristic comparison (default matches
            BioSecure.py's PARTICIPATION_RATES).
        n_runs, n_rounds, local_epochs, lr, mu: same semantics as the
            main experiment -- keep these identical to the metaheuristic
            run for a fair comparison.
        output_dir: if provided, reuse an existing output directory
            (e.g. the same one from run_participation_rate_comparison,
            so both CSVs land side by side). If None, creates a fresh
            timestamped directory under output_base_dir.

    Returns:
        (df_fedmccs, output_dir) where df_fedmccs has the SAME COLUMNS
        as BioSecure.py's participation_rate_comparison.csv, plus:
          - Fulfillment_Rate_Avg, Fulfillment_Rate_Std,
            Skipped_Resources_Avg (diagnostic columns)
          - N_Selected_Clients_Actual, Energy_Per_Actual_Selected_Client,
            Efficiency_Score_Actual (CORRECTED energy-per-client metrics
            -- use these, not the base Energy_Per_Selected_Client /
            Efficiency_Score columns, when comparing FedMCCS's energy
            cost against the metaheuristics, e.g. in the Pareto plot.
            The base columns divide by the TARGET client count and
            understate FedMCCS's true per-client cost whenever
            fulfillment_rate < 1 -- see inline comment at the point
            these are computed for the full explanation),
        and Selection_Method is fixed to 'FedMCCS', Fitness_Mode to 'N/A'.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        output_dir = os.path.join(output_base_dir, f"fedmccs_baseline_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("FEDMCCS BASELINE EXPERIMENT (Option A: rate-only, no fitness_mode)")
    print(f"Testing rates: {list(participation_rates)}")
    print(f"Device: {device}")
    print("=" * 80)

    # --- Pre-process data once (mirrors run_participation_rate_comparison) ---
    print("\n" + "-" * 40)
    print("PRE-PROCESSING DATA (one-time for all rates)")
    print("-" * 40)

    base_experiment = MultiMethodExperiment(n_features, device)
    base_experiment.prepare_data_once(volunteer_data, extractor)

    prepared_client_data = base_experiment.prepared_client_data

    actual_n_features = n_features
    if prepared_client_data:
        first_client = next(iter(prepared_client_data.values()))
        actual_n_features = first_client['X_train'].shape[1]

    print(f"Data prepared for {len(prepared_client_data)} clients with {actual_n_features} features")
    n_total_clients = len(prepared_client_data)

    # --- Load edge simulation once ---
    shared_edge_env = None
    if use_edge_sim and edge_sim_file:
        from DynaEdgeSimV3 import Environment
        print("\n" + "-" * 40)
        print("LOADING EDGE SIMULATION (one-time)")
        print("-" * 40)
        shared_edge_env = Environment()
        shared_edge_env.load_simulation_data(edge_sim_file)
        print(f"Loaded simulation from {edge_sim_file}")

    import copy

    def run_single_run(run_idx, rate):
        # Same seeding convention as BioSecure.py's run_single_run
        seed = 42 + run_idx * 100
        np.random.seed(seed)
        torch.manual_seed(seed)

        fl = EnhancedFLSystem(actual_n_features, device, fitness_mode='learning')
        fl.client_data = prepared_client_data

        for subj_id in fl.client_data:
            fl.client_alphas[subj_id] = 0.5
            fl.fl_clients[subj_id] = FLClientWrapper(subj_id, fl.client_data[subj_id])

        fl.initialize_models()

        if shared_edge_env is not None:
            fl.edge_env = copy.deepcopy(shared_edge_env)
            fl.edge_env.SimMoveFirst()
            fl._map_clients_to_devices()

        history = fl.train_federated(
            selection_method='FedMCCS',    # <-- requires the hook, see module docstring
            participation_rate=rate,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            lr=lr,
            mu=mu,
            verbose=False,
            show_progress=True,
            progress_position=2,
            leave_progress=False
        )

        fl.personalize_models(verbose=False)
        fl.learn_adaptive_alpha(verbose=False)
        metrics, predictions = fl.evaluate_all_with_predictions(verbose=False)

        # Per-round FedMCCS diagnostics, populated by the hook if wired up.
        # Falls back gracefully to empty list if the hook isn't in place yet
        # (fulfillment/skip columns will just read as NaN for this run).
        fedmccs_diag_per_round = getattr(fl, 'fedmccs_diagnostics_per_round', [])

        return {
            'run': run_idx,
            'history': history,
            'metrics': metrics,
            'predictions': predictions,
            'fedmccs_diag_per_round': fedmccs_diag_per_round,
        }

    # --- Execute experiments ---
    global_start_time = time.time()
    results_by_rate = defaultdict(lambda: {
        'acc_histories': [], 'loss_histories': [], 'final_metrics': [],
        'energy_histories': [], 'total_energy': [], 'fitness_histories': [],
        'fulfillment_rates': [], 'skipped_resources': [],
    })

    total_experiments = len(list(participation_rates)) * n_runs
    if TQDM_AVAILABLE:
        global_pbar = tqdm(total=total_experiments, desc="FedMCCS Progress",
                            unit="exp", ncols=110)

    for rate in participation_rates:
        rate_start_time = time.time()
        run_results = []

        for run_idx in range(n_runs):
            result = run_single_run(run_idx, rate)
            run_results.append(result)
            if TQDM_AVAILABLE:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    'rate': f"{int(rate*100)}%",
                    'run': f"{run_idx+1}/{n_runs}"
                })

        for result in run_results:
            history = result['history']
            results_by_rate[rate]['acc_histories'].append(history['global_acc'])
            results_by_rate[rate]['loss_histories'].append(history['global_loss'])
            results_by_rate[rate]['final_metrics'].append(result['metrics'])
            results_by_rate[rate]['energy_histories'].append(history.get('energy_consumed', []))
            results_by_rate[rate]['total_energy'].append(history.get('total_energy', 0))
            results_by_rate[rate]['fitness_histories'].append(history.get('selection_fitness', []))

            diag_rounds = result['fedmccs_diag_per_round']
            if diag_rounds:
                results_by_rate[rate]['fulfillment_rates'].append(
                    np.mean([d['fulfillment_rate'] for d in diag_rounds]))
                results_by_rate[rate]['skipped_resources'].append(
                    np.mean([d['skipped_insufficient_resources'] for d in diag_rounds]))

        accs = [m['global']['accuracy'] for m in results_by_rate[rate]['final_metrics']]
        rate_time = time.time() - rate_start_time
        print(f"  Rate {int(rate*100)}%: Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f} "
              f"({_format_time(rate_time)})")

    if TQDM_AVAILABLE:
        global_pbar.close()

    total_time = time.time() - global_start_time
    print(f"\nFEDMCCS BASELINE COMPLETE! Total time: {_format_time(total_time)}")

    # --- Build comparison rows, SAME SCHEMA as participation_rate_comparison.csv ---
    comparison_data = []
    for rate, data in results_by_rate.items():
        final_fitness = [_to_positive_fitness(h[-1]) for h in data['fitness_histories'] if h]
        final_acc = [m['global']['accuracy'] for m in data['final_metrics']]
        total_energy = data['total_energy']
        final_loss = [h[-1] for h in data['loss_histories'] if h]

        # NOTE: unlike the metaheuristics, FedMCCS may select FEWER than
        # ceil(rate * N) clients if not enough clear the resource budget
        # (see fulfillment_rate below). N_Selected_Clients here reports
        # the TARGET ceiling for schema compatibility; the actual average
        # selected count is implied by Fulfillment_Rate_Avg * this value.
        # Matches EnhancedFLSystem._select_clients_metaheuristic /
        # _select_clients_fedmccs's exact k formula (truncating int(),
        # not ceiling) so this target lines up with what the pipeline
        # actually asked FedMCCS to reach each round.
        n_selected_target = max(1, int(rate * n_total_clients))
        energy_per_selected = (np.mean(total_energy) / n_selected_target
                                if total_energy and n_selected_target > 0 else 0)
        accuracy_avg = np.mean(final_acc) if final_acc else 0

        # CORRECTED energy-per-client metric: divides by the ACTUAL number
        # of clients FedMCCS trained (target * fulfillment_rate), not the
        # target itself. FedMCCS's resource-budget gate (budget_compute_min/
        # budget_energy_max/budget_latency_max in fedmccs_selection.py)
        # caps how many clients can ever clear the bar in a round,
        # independent of the requested rate -- so at higher rates,
        # fulfillment_rate drops and 'Energy_Per_Selected_Client' above
        # silently understates FedMCCS's true per-client cost (it looks
        # like it's getting cheaper per client purely because the same
        # roughly-fixed energy total is divided by a growing, mostly
        # unmet target). This column reports the honest figure so
        # plots/tables comparing FedMCCS to the metaheuristics (which DO
        # scale actual selected count with the target) aren't misleading.
        fulfillment_avg = np.mean(data['fulfillment_rates']) if data['fulfillment_rates'] else np.nan
        n_selected_actual = (n_selected_target * fulfillment_avg
                              if not np.isnan(fulfillment_avg) else np.nan)
        energy_per_actual_selected = (
            np.mean(total_energy) / n_selected_actual
            if total_energy and n_selected_actual and n_selected_actual > 0
            else np.nan
        )

        comparison_data.append({
            'Participation_Rate': rate,
            'Selection_Method': 'FedMCCS',
            'Fitness_Mode': 'N/A',   # FedMCCS has no fitness-mode axis (Option A)
            'Fitness_Avg': np.mean(final_fitness) if final_fitness else np.nan,
            'Fitness_Std': np.std(final_fitness) if final_fitness else np.nan,
            'Accuracy_Avg': accuracy_avg,
            'Accuracy_Std': np.std(final_acc) if final_acc else 0,
            'Loss_Avg': np.mean(final_loss) if final_loss else 0,
            'Loss_Std': np.std(final_loss) if final_loss else 0,
            'Energy_Avg': np.mean(total_energy) if total_energy else 0,
            'Energy_Std': np.std(total_energy) if total_energy else 0,
            'N_Total_Clients': n_total_clients,
            'N_Selected_Clients': n_selected_target,
            'Energy_Per_Selected_Client': energy_per_selected,
            # --- Corrected metric (see comment above) -- use THIS one for
            # any accuracy-vs-energy comparison against the metaheuristics,
            # e.g. the Pareto frontier plot ---
            'N_Selected_Clients_Actual': n_selected_actual,
            'Energy_Per_Actual_Selected_Client': energy_per_actual_selected,
            'Energy_Per_Total_Client': (np.mean(total_energy) / n_total_clients
                                         if total_energy and n_total_clients > 0 else 0),
            'Accuracy_Per_Energy': (np.mean(final_acc) / np.mean(total_energy)
                                     if final_acc and total_energy and np.mean(total_energy) > 0 else 0),
            'Efficiency_Score': ((accuracy_avg * 100) / energy_per_selected
                                  if energy_per_selected > 0 else 0),
            # Corrected efficiency score, consistent with Energy_Per_Actual_Selected_Client
            'Efficiency_Score_Actual': ((accuracy_avg * 100) / energy_per_actual_selected
                                         if energy_per_actual_selected and energy_per_actual_selected > 0 else np.nan),
            # --- FedMCCS-specific diagnostic columns (not in the base schema) ---
            'Fulfillment_Rate_Avg': fulfillment_avg,
            'Fulfillment_Rate_Std': (np.std(data['fulfillment_rates'])
                                      if data['fulfillment_rates'] else np.nan),
            'Skipped_Resources_Avg': (np.mean(data['skipped_resources'])
                                       if data['skipped_resources'] else np.nan),
        })

    df_fedmccs = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'fedmccs_baseline_results.csv')
    df_fedmccs.to_csv(csv_path, index=False)
    print(f"\nSaved FedMCCS baseline results to {csv_path}")

    return df_fedmccs, output_dir


def merge_with_metaheuristic_comparison(df_comparison, df_fedmccs, output_dir):
    """
    Concatenate the metaheuristic comparison DataFrame (from
    run_participation_rate_comparison) with the FedMCCS baseline
    DataFrame, aligning on the shared schema, and save the combined CSV.

    The FedMCCS-only diagnostic/corrected-metric columns (Fulfillment_Rate_*,
    Skipped_Resources_Avg, N_Selected_Clients_Actual,
    Energy_Per_Actual_Selected_Client, Efficiency_Score_Actual) will be NaN
    for all metaheuristic rows -- this is expected and useful downstream
    (lets you filter `df['Selection_Method'] == 'FedMCCS'` to isolate the
    diagnostic-relevant rows). For any accuracy-vs-energy plot/table that
    includes FedMCCS (e.g. the Pareto frontier), use
    Energy_Per_Actual_Selected_Client for FedMCCS rows and
    Energy_Per_Selected_Client for metaheuristic rows -- they're already
    equivalent for the metaheuristics (fulfillment is always 1 there), so
    a single combined "energy per selected client" column can be built as:
        df_all['Energy_Per_Selected_Client_Fair'] = df_all['Energy_Per_Actual_Selected_Client'].fillna(df_all['Energy_Per_Selected_Client'])

    Returns:
        df_all: the concatenated DataFrame.
    """
    df_all = pd.concat([df_comparison, df_fedmccs], ignore_index=True, sort=False)
    csv_path = os.path.join(output_dir, 'participation_rate_comparison_with_fedmccs.csv')
    df_all.to_csv(csv_path, index=False)
    print(f"Saved combined comparison (metaheuristics + FedMCCS) to {csv_path}")
    return df_all
