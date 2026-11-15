# -*- coding: utf-8 -*-
"""
================================================================================
BioSecure.py - Main Program for Federated Learning Cardiac Monitoring
================================================================================
SEQUENTIAL VERSION - All parallelization removed for stability and simplicity

Main program that orchestrates:
1. Data loading and feature extraction
2. Federated Learning experiments
3. Participation rate comparison
4. Byzantine resilience experiments
5. Result generation and visualization

Uses:
- FederatedLearningBS: FL system, models, Byzantine defense
- ResultBS: Result management, plotting, statistics
================================================================================
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import copy
import time

import torch

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")

# Import from our libraries
from FederatedLearningBS import (
    SensSmartTechLoader,
    FeatureExtractor,
    EnhancedFLSystem,
    FLClientWrapper,
    FITNESS_MODES,
    ByzantineDefense,
    ByzantineAttackSimulator
)

from ResultBS import (
    MultiMethodExperiment,
    _plot_participation_rate_comparison as plot_participation_rate_comparison,
    plot_active_vs_selected_clients,
    _plot_byzantine_results as plot_byzantine_results,
    _plot_byzantine_method_comparison as plot_byzantine_method_comparison,
    generate_byzantine_summary
)

# Import DynaEdgeSim for edge simulation
from DynaEdgeSimV3 import Environment

# Import FedMCCS baseline experiment + plots (published-method comparison, R3-11)
from fedmccs_experiment import run_fedmccs_baseline_experiment, merge_with_metaheuristic_comparison
from fedmccs_plots import generate_all_fedmccs_plots


# ==============================================================================
# PARTICIPATION RATE COMPARISON EXPERIMENT
# ==============================================================================

def run_participation_rate_comparison(volunteer_data, extractor, n_features, device,
                                      selection_methods, fitness_modes,
                                      participation_rates=[0.3, 0.5, 0.7, 1.0],
                                      n_runs=2, n_rounds=50, local_epochs=5,
                                      lr=0.01, mu=0.01, use_edge_sim=True,
                                      edge_sim_file=None, output_base_dir='Results'):
    """
    Run experiments across different participation rates - SEQUENTIAL execution.
    """
    all_results = {}
    comparison_data = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    output_dir = os.path.join(output_base_dir, f"participation_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PARTICIPATION RATE COMPARISON EXPERIMENT")
    print(f"Testing rates: {participation_rates}")
    print(f"Device: {device}")
    print("→ Sequential processing (optimized)")
    print("=" * 80)
    
    def to_positive_fitness(val):
        return -val if val < 0 else val
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"
    
    # Pre-process data once
    print("\n" + "-" * 40)
    print("PRE-PROCESSING DATA (one-time for all rates)")
    print("-" * 40)
    
    base_experiment = MultiMethodExperiment(n_features, device)
    base_experiment.prepare_data_once(volunteer_data, extractor)
    
    prepared_client_data = base_experiment.prepared_client_data
    prepared_scaler = base_experiment.prepared_scaler
    
    actual_n_features = n_features
    if prepared_client_data:
        first_client = next(iter(prepared_client_data.values()))
        actual_n_features = first_client['X_train'].shape[1]
    
    print(f"✓ Data prepared for {len(prepared_client_data)} clients with {actual_n_features} features")
    n_total_clients = len(prepared_client_data)
    
    # Load edge simulation once
    shared_edge_env = None
    if use_edge_sim and edge_sim_file:
        print("\n" + "-" * 40)
        print("LOADING EDGE SIMULATION (one-time)")
        print("-" * 40)
        shared_edge_env = Environment()
        shared_edge_env.load_simulation_data(edge_sim_file)
        print(f"✓ Loaded simulation from {edge_sim_file}")
    
    # Build combination list
    combinations = []
    for rate in participation_rates:
        for method in selection_methods:
            for fitness_mode in fitness_modes:
                combinations.append({
                    'rate': rate,
                    'method': method,
                    'fitness_mode': fitness_mode
                })
    
    total_combinations = len(combinations)
    total_experiments = total_combinations * n_runs
    
    print(f"\nExperiment structure:")
    print(f"  Combinations: {total_combinations} (rate × method × fitness)")
    print(f"  Runs per combination: {n_runs}")
    print(f"  Total experiments: {total_experiments}")
    
    # Worker function for single run
    def run_single_run(run_idx, rate, method, fitness_mode):
        seed = 42 + run_idx * 100
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        fl = EnhancedFLSystem(actual_n_features, device, fitness_mode=fitness_mode)
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
            selection_method=method,
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
        
        return {
            'run': run_idx,
            'history': history,
            'metrics': metrics,
            'predictions': predictions
        }
    
    # Execute experiments
    global_start_time = time.time()
    completed_experiments = 0
    
    results_by_combo = defaultdict(lambda: {
        'acc_histories': [],
        'loss_histories': [],
        'final_metrics': [],
        'selection_metrics': [],
        'energy_histories': [],
        'total_energy': [],
        'fitness_histories': [],
        'active_clients_histories': [],
        'selected_clients_histories': [],
        'all_predictions': [],
        'avg_battery_histories': [],
        'avg_reliability_histories': [],
        'avg_latency_histories': [],
        'metaheuristic_curves': [],
    })
    
    print(f"\n{'='*80}")
    print(f"🔄 SEQUENTIAL EXECUTION")
    print(f"   Total Combinations: {total_combinations}")
    print(f"   Runs per Combination: {n_runs}")
    print(f"   FL Rounds per Run: {n_rounds}")
    print(f"   Total Experiments: {total_experiments}")
    print(f"{'='*80}\n")
    
    if TQDM_AVAILABLE:
        global_pbar = tqdm(
            total=total_experiments,
            desc="🌐 Global Progress",
            unit="exp",
            ncols=110,
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    
    for combo_idx, combo in enumerate(combinations):
        rate = combo['rate']
        method = combo['method']
        fitness_mode = combo['fitness_mode']
        combo_key = f"{rate}_{method}_{fitness_mode}"
        
        combo_start_time = time.time()
        
        if TQDM_AVAILABLE:
            global_pbar.set_description(f"🌐 [{combo_idx+1}/{total_combinations}] {method}|{fitness_mode}|{int(rate*100)}%")
        
        run_results = []
        
        if TQDM_AVAILABLE:
            run_pbar = tqdm(
                range(n_runs),
                desc=f"  📊 Runs ({method[:5]}|{fitness_mode[:3]}|{int(rate*100)}%)",
                unit="run",
                ncols=100,
                position=1,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
        else:
            run_pbar = range(n_runs)
        
        for run_idx in run_pbar if not TQDM_AVAILABLE else run_pbar:
            result = run_single_run(run_idx, rate, method, fitness_mode)
            run_results.append(result)
            
            if TQDM_AVAILABLE:
                run_pbar.set_postfix({
                    'Acc': f"{result['metrics']['global']['accuracy']:.3f}",
                    'Loss': f"{result['history']['global_loss'][-1]:.3f}" if result['history']['global_loss'] else 'N/A'
                })
            
            completed_experiments += 1
            if TQDM_AVAILABLE:
                avg_acc = np.mean([r['metrics']['global']['accuracy'] for r in run_results])
                global_pbar.update(1)
                global_pbar.set_postfix({
                    'Combo': f"{combo_idx+1}/{total_combinations}",
                    'AvgAcc': f"{avg_acc:.3f}"
                })
        
        if TQDM_AVAILABLE:
            run_pbar.close()
        
        # Store results
        for result in run_results:
            history = result['history']
            metrics = result['metrics']
            predictions = result.get('predictions', {})
            
            results_by_combo[combo_key]['acc_histories'].append(history['global_acc'])
            results_by_combo[combo_key]['loss_histories'].append(history['global_loss'])
            results_by_combo[combo_key]['final_metrics'].append(metrics)
            results_by_combo[combo_key]['selection_metrics'].append(history.get('selection_metrics', []))
            results_by_combo[combo_key]['energy_histories'].append(history.get('energy_consumed', []))
            results_by_combo[combo_key]['total_energy'].append(history.get('total_energy', 0))
            results_by_combo[combo_key]['fitness_histories'].append(history.get('selection_fitness', []))
            results_by_combo[combo_key]['active_clients_histories'].append(history.get('active_clients_per_round', []))
            results_by_combo[combo_key]['selected_clients_histories'].append(history.get('clients_selected_per_round', []))
            results_by_combo[combo_key]['all_predictions'].append(predictions)
            results_by_combo[combo_key]['avg_battery_histories'].append(history.get('avg_battery_level', []))
            results_by_combo[combo_key]['avg_reliability_histories'].append(history.get('avg_reliability', []))
            results_by_combo[combo_key]['avg_latency_histories'].append(history.get('avg_latency', []))
            results_by_combo[combo_key]['metaheuristic_curves'].append(history.get('metaheuristic_curves', []))
        
        results_by_combo[combo_key]['rate'] = rate
        results_by_combo[combo_key]['method'] = method
        results_by_combo[combo_key]['fitness_mode'] = fitness_mode
        results_by_combo[combo_key]['total_clients'] = history.get('total_clients', n_total_clients)
        
        accs = [r['metrics']['global']['accuracy'] for r in run_results]
        combo_time = time.time() - combo_start_time
        if TQDM_AVAILABLE:
            tqdm.write(f"   ✅ {method}|{fitness_mode}|{int(rate*100)}%: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} ({format_time(combo_time)})")
    
    if TQDM_AVAILABLE:
        global_pbar.close()
    
    total_time = time.time() - global_start_time
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per combination: {total_time/total_combinations:.1f}s")
    print(f"Average time per run: {total_time/total_experiments:.1f}s")
    print(f"{'='*60}")
    
    # Build comparison data
    for combo_key, data in results_by_combo.items():
        rate = data.get('rate', 0)
        method = data.get('method', '')
        fitness_mode = data.get('fitness_mode', '')
        
        final_fitness = [to_positive_fitness(h[-1]) for h in data.get('fitness_histories', []) if h]
        final_acc = [m['global']['accuracy'] for m in data.get('final_metrics', [])]
        total_energy = data.get('total_energy', [])
        
        loss_hists = data.get('loss_histories', [])
        final_loss = [h[-1] for h in loss_hists if h]
        
        n_selected_clients = int(rate * n_total_clients)
        energy_per_selected = np.mean(total_energy) / n_selected_clients if total_energy and n_selected_clients > 0 else 0
        accuracy_avg = np.mean(final_acc) if final_acc else 0
        
        comparison_data.append({
            'Participation_Rate': rate,
            'Selection_Method': method,
            'Fitness_Mode': fitness_mode,
            'Fitness_Avg': np.mean(final_fitness) if final_fitness else 0,
            'Fitness_Std': np.std(final_fitness) if final_fitness else 0,
            'Accuracy_Avg': accuracy_avg,
            'Accuracy_Std': np.std(final_acc) if final_acc else 0,
            'Loss_Avg': np.mean(final_loss) if final_loss else 0,
            'Loss_Std': np.std(final_loss) if final_loss else 0,
            'Energy_Avg': np.mean(total_energy) if total_energy else 0,
            'Energy_Std': np.std(total_energy) if total_energy else 0,
            'N_Total_Clients': n_total_clients,
            'N_Selected_Clients': n_selected_clients,
            'Energy_Per_Selected_Client': energy_per_selected,
            'Energy_Per_Total_Client': np.mean(total_energy) / n_total_clients if total_energy and n_total_clients > 0 else 0,
            'Accuracy_Per_Energy': np.mean(final_acc) / np.mean(total_energy) if final_acc and total_energy and np.mean(total_energy) > 0 else 0,
            'Efficiency_Score': (accuracy_avg * 100) / energy_per_selected if energy_per_selected > 0 else 0,
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(f'{output_dir}/participation_rate_comparison.csv', index=False)
    print(f"\n✓ Saved comparison data to {output_dir}/participation_rate_comparison.csv")
    
    # Generate comparison plots
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)
    plot_participation_rate_comparison(df_comparison, selection_methods, fitness_modes, 
                                       participation_rates, output_dir)
    
    # Generate active vs selected clients plot
    plot_active_vs_selected_clients(results_by_combo, output_dir, n_rounds)
    
    # Generate detailed reports per participation rate (rate_XX subfolders)
    print("\n" + "=" * 60)
    print("GENERATING DETAILED REPORTS PER PARTICIPATION RATE")
    print("=" * 60)
    
    for rate in participation_rates:
        rate_dir = os.path.join(output_dir, f"rate_{int(rate*100)}")
        os.makedirs(rate_dir, exist_ok=True)
        print(f"\n--- Rate {int(rate*100)}% ---")
        
        # Create experiment object with results for this rate
        rate_experiment = MultiMethodExperiment(actual_n_features, device)
        rate_experiment.experiment_params = {
            'selection_methods': selection_methods,
            'fitness_modes': fitness_modes,
            'n_runs': n_runs,
            'n_rounds': n_rounds,
            'participation_rate': rate,
            'local_epochs': local_epochs,
            'lr': lr,
            'mu': mu,
            'use_edge_sim': use_edge_sim,
            'edge_sim_file': edge_sim_file,
            'timestamp': timestamp
        }
        
        # Filter results for this rate
        for combo_key, data in results_by_combo.items():
            if data.get('rate') == rate:
                method = data.get('method')
                fitness_mode = data.get('fitness_mode')
                result_key = f"{method}_{fitness_mode}"
                
                rate_experiment.results[result_key] = {
                    'acc_histories': data.get('acc_histories', []),
                    'loss_histories': data.get('loss_histories', []),
                    'final_metrics': data.get('final_metrics', []),
                    'selection_metrics': data.get('selection_metrics', []),
                    'energy_histories': data.get('energy_histories', []),
                    'total_energy': data.get('total_energy', []),
                    'fitness_histories': data.get('fitness_histories', []),
                    'all_predictions': data.get('all_predictions', []),
                    'avg_battery_histories': data.get('avg_battery_histories', []),
                    'avg_reliability_histories': data.get('avg_reliability_histories', []),
                    'avg_latency_histories': data.get('avg_latency_histories', []),
                    'metaheuristic_curves': data.get('metaheuristic_curves', []),
                    'selection_method': method,
                    'fitness_mode': fitness_mode,
                }
        
        # Generate full report for this rate (uses all MultiMethodExperiment methods)
        # This creates a timestamp subfolder like the original code
        if rate_experiment.results:
            stats, summary_df, report_dir = rate_experiment.generate_report(base_output_dir=rate_dir)
            print(f"  ✓ Generated full report for rate {int(rate*100)}% in {report_dir}")
    
    return all_results, df_comparison, output_dir


# ==============================================================================
# BYZANTINE RESILIENCE EXPERIMENT
# ==============================================================================

def run_byzantine_experiment(volunteer_data, extractor, n_features, device,
                            selection_methods=['PSO', 'GWO', 'RSELECT'],
                            attack_types=['none', 'random', 'sign_flip', 'scaling'],
                            byzantine_fractions=[0.0, 0.1, 0.2, 0.3],
                            defense_methods=['none', 'median', 'krum', 'combined'],
                            n_runs=3, n_rounds=30, participation_rate=0.5,
                            local_epochs=5, lr=0.01, mu=0.01,
                            use_edge_sim=True, edge_sim_file=None,
                            output_base_dir='Results'):
    """
    Comprehensive Byzantine resilience experiment - SEQUENTIAL execution.
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    output_dir = os.path.join(output_base_dir, f"byzantine_experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("BYZANTINE RESILIENCE EXPERIMENT")
    print("=" * 80)
    print(f"Attack Types: {attack_types}")
    print(f"Byzantine Fractions: {byzantine_fractions}")
    print(f"Defense Methods: {defense_methods}")
    print(f"Selection Methods: {selection_methods}")
    print(f"Device: {device}")
    print("→ Sequential processing (optimized)")
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"
    
    # Pre-process data once
    print("\n" + "-" * 40)
    print("PRE-PROCESSING DATA (one-time)")
    print("-" * 40)
    
    temp_fl = EnhancedFLSystem(n_features, device, fitness_mode='learning')
    temp_fl.prepare_client_data(volunteer_data,extractor, test_size=0.2, 
                             min_samples_per_client=16, augment_data=False, 
                             max_augment_ratio=2.0)
    
    prepared_client_data = temp_fl.client_data
    prepared_scaler = temp_fl.scaler
    n_clients = len(prepared_client_data)
    
    actual_n_features = temp_fl.n_features
    if prepared_client_data:
        first_client = next(iter(prepared_client_data.values()))
        actual_n_features = first_client['X_train'].shape[1]
    print(f"✓ Prepared data for {n_clients} clients with {actual_n_features} features")
    
    # Load edge simulation once
    shared_edge_env = None
    if use_edge_sim and edge_sim_file:
        print("\n" + "-" * 40)
        print("LOADING EDGE SIMULATION (one-time)")
        print("-" * 40)
        shared_edge_env = Environment()
        shared_edge_env.load_simulation_data(edge_sim_file)
        print(f"✓ Loaded simulation from {edge_sim_file}")
    
    all_results = []
    convergence_data = []
    
    # Build combination list
    combinations = []
    for attack_type in attack_types:
        if attack_type == 'none':
            fractions_to_test = [0.0]
        else:
            fractions_to_test = [f for f in byzantine_fractions if f > 0]
        
        for byz_fraction in fractions_to_test:
            for defense_method in defense_methods:
                for selection_method in selection_methods:
                    combinations.append({
                        'attack_type': attack_type,
                        'byz_fraction': byz_fraction,
                        'defense_method': defense_method,
                        'selection_method': selection_method
                    })
    
    total_combinations = len(combinations)
    total_experiments = total_combinations * n_runs
    
    print(f"\nExperiment structure:")
    print(f"  Combinations: {total_combinations} (attack × fraction × defense × selection)")
    print(f"  Runs per combination: {n_runs}")
    print(f"  Total experiments: {total_experiments}")
    
    def run_single_run(run_idx, attack_type, byz_fraction, defense_method, selection_method):
                
        fl = EnhancedFLSystem(
            actual_n_features,
            device, 
            fitness_mode='learning',
            byzantine_defense=defense_method,
            byzantine_threshold=2.5
        )
        
        fl.client_data = prepared_client_data
        
        for subj_id in fl.client_data:
            fl.client_alphas[subj_id] = 0.5
            fl.fl_clients[subj_id] = FLClientWrapper(subj_id, fl.client_data[subj_id])
        
        fl.initialize_models()
        fl.setup_byzantine_defense()
        
        if attack_type != 'none' and byz_fraction > 0:
            fl.setup_attack_simulation(
                attack_type=attack_type,
                attack_strength=1.0,
                target_fraction=byz_fraction
            )
        
        if shared_edge_env is not None:
            fl.edge_env = copy.deepcopy(shared_edge_env)
            fl.edge_env.SimMoveFirst()
            fl._map_clients_to_devices()
        
        history = fl.train_federated(
            selection_method=selection_method,
            participation_rate=participation_rate,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            lr=lr,
            mu=mu,
            verbose=False,
            show_progress=True,
            progress_position=2,
            leave_progress=False
        )
        
        final_acc = fl.evaluate_global()
        final_loss = fl.compute_global_loss()
        
        byz_stats = {}
        if fl.byzantine_defender:
            byz_stats = fl.byzantine_defender.get_statistics()
        
        return {
            'Run': run_idx,
            'Final_Accuracy': final_acc,
            'Final_Loss': final_loss,
            'Max_Accuracy': max(history['global_acc']) if history['global_acc'] else 0,
            'Min_Loss': min(history['global_loss']) if history['global_loss'] else 0,
            'Convergence_Round': history.get('convergence_round'),
            'Total_Detections': sum(history.get('byzantine_detections', [0])),
            'Avg_Detection_Rate': byz_stats.get('average_detection_rate', 0),
            'Total_Energy': history.get('total_energy', 0),
            'Accuracy_History': history['global_acc'],
            'Loss_History': history['global_loss'],
        }
    
    # Execute experiments
    global_start_time = time.time()
    completed_experiments = 0
    
    print(f"\n{'='*80}")
    print(f"🔄 SEQUENTIAL EXECUTION")
    print(f"   Total Combinations: {total_combinations}")
    print(f"   Runs per Combination: {n_runs}")
    print(f"   FL Rounds per Run: {n_rounds}")
    print(f"   Total Experiments: {total_experiments}")
    print(f"{'='*80}\n")
    
    if TQDM_AVAILABLE:
        global_pbar = tqdm(
            total=total_experiments,
            desc="🌐 Byzantine Exp Progress",
            unit="exp",
            ncols=110,
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    
    for combo_idx, combo in enumerate(combinations):
        attack_type = combo['attack_type']
        byz_fraction = combo['byz_fraction']
        defense_method = combo['defense_method']
        selection_method = combo['selection_method']
        
        combo_start_time = time.time()
        combo_label = f"{attack_type[:4]}|{int(byz_fraction*100)}%|{defense_method[:4]}|{selection_method}"
        
        if TQDM_AVAILABLE:
            global_pbar.set_description(f"🌐 [{combo_idx+1}/{total_combinations}] {combo_label}")
        
        run_results = []
        
        if TQDM_AVAILABLE:
            run_pbar = tqdm(
                range(n_runs),
                desc=f"  📊 Runs ({combo_label})",
                unit="run",
                ncols=100,
                position=1,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
        else:
            run_pbar = range(n_runs)
        
        for run_idx in run_pbar if not TQDM_AVAILABLE else run_pbar:
            result = run_single_run(run_idx, attack_type, byz_fraction, 
                                   defense_method, selection_method)
            run_results.append(result)
            
            if TQDM_AVAILABLE:
                run_pbar.set_postfix({
                    'Acc': f"{result['Final_Accuracy']:.3f}",
                    'Loss': f"{result['Final_Loss']:.3f}"
                })
            
            completed_experiments += 1
            if TQDM_AVAILABLE:
                avg_acc = np.mean([r['Final_Accuracy'] for r in run_results])
                global_pbar.update(1)
                global_pbar.set_postfix({
                    'Combo': f"{combo_idx+1}/{total_combinations}",
                    'AvgAcc': f"{avg_acc:.3f}"
                })
        
        if TQDM_AVAILABLE:
            run_pbar.close()
        
        for result in run_results:
            result['Attack_Type'] = attack_type
            result['Byzantine_Fraction'] = byz_fraction
            result['Defense_Method'] = defense_method
            result['Selection_Method'] = selection_method
            all_results.append(result)
            
            if result['Run'] == 0:
                convergence_data.append({
                    'Attack_Type': attack_type,
                    'Byzantine_Fraction': byz_fraction,
                    'Defense_Method': defense_method,
                    'Selection_Method': selection_method,
                    'Accuracy_History': result['Accuracy_History'],
                    'Loss_History': result['Loss_History']
                })
        
        accs = [r['Final_Accuracy'] for r in run_results]
        combo_time = time.time() - combo_start_time
        if TQDM_AVAILABLE:
            tqdm.write(f"   ✅ {combo_label}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} ({format_time(combo_time)})")
    
    if TQDM_AVAILABLE:
        global_pbar.close()
    
    total_time = time.time() - global_start_time
    print(f"\n{'='*60}")
    print(f"BYZANTINE EXPERIMENT COMPLETE!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per combination: {total_time/total_combinations:.1f}s")
    print(f"Average time per run: {total_time/total_experiments:.1f}s")
    print(f"{'='*60}")
    
    df = pd.DataFrame(all_results)
    df_csv = df.drop(columns=['Accuracy_History', 'Loss_History'], errors='ignore')
    df_csv.to_csv(f'{output_dir}/byzantine_experiment_results.csv', index=False)
    print(f"\n✓ Saved results to {output_dir}/byzantine_experiment_results.csv")

    # PERF/SAFETY: convergence_data (per-round Accuracy_History/Loss_History,
    # tagged with Selection_Method) was previously only held in memory to
    # feed plot_byzantine_results() below -- if plotting crashed or the
    # session closed, this raw trajectory data was gone with no way to
    # rebuild it short of rerunning the whole experiment. Persist it now,
    # unconditionally, before any plotting is attempted. This also means
    # a later single-method run (e.g. DE alone) can be directly compared
    # against this run's saved trajectories without needing to rerun GWO.
    import pickle
    with open(f'{output_dir}/convergence_data.pkl', 'wb') as f:
        pickle.dump(convergence_data, f)
    print(f"✓ Saved convergence_data.pkl ({len(convergence_data)} entries) to {output_dir}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING BYZANTINE EXPERIMENT PLOTS")
    print("=" * 60)
    plot_byzantine_results(df, attack_types, byzantine_fractions, defense_methods,
                          selection_methods, output_dir, convergence_data)

    # NEW: head-to-head selection-method comparison under attack (e.g. GWO vs
    # DE) -- no-ops automatically if only one selection method was run.
    plot_byzantine_method_comparison(df, attack_types, byzantine_fractions,
                                    defense_methods, selection_methods, output_dir)
    
    # Generate summary statistics
    generate_byzantine_summary(df, output_dir)
    
    return df, output_dir


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-METHOD FL WITH THREE FITNESS MODES")
    print("=" * 80)
    
    # Configuration
    CSV_DIR = os.environ.get("BIOSECURE_CSV_DIR", os.path.join("dataset", "csv"))
    DEMOGRAPHICS_PATH = os.environ.get("BIOSECURE_DEMOGRAPHICS", os.path.join("dataset", "Demographics.csv"))
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {DEVICE}")
    
    print(f"\nProcessing Mode: SEQUENTIAL (optimized)")
    
    # Experiment parameters
    SELECTION_METHODS = ['ACO','RSELECT', 'PSO', 'GA', 'GWO', 'DE', 'SA', 'TSA']    

    EXPERIMENT_FITNESS_MODES = ['resource', 'learning', 'exploration']
    #EXPERIMENT_FITNESS_MODES = ['learning']
    
    #N_RUNS = 10
    N_RUNS = 10
    #N_ROUNDS = 100
    N_ROUNDS = 100
    LOCAL_EPOCHS = 3
    LR = 0.001
    MU = 0.2
    
    PARTICIPATION_RATES = [0.1, 0.2, 0.3, 0.5]
    
    USE_EDGE_SIM = True
    EDGE_SIM_FILE = "mobile_sim_32C_100R_V3.csv" 
    
    OUTPUT_BASE_DIR = 'Results_revision1'
 
    run_experiment=False     
    run_byzantine=True
    
    # Load data
    loader = SensSmartTechLoader(CSV_DIR, DEMOGRAPHICS_PATH)
    volunteer_data = loader.load_all_data()
    
    # Feature extraction
    print("\n" + "=" * 40)
    print("FEATURE EXTRACTION")
    print("=" * 40)
    extractor = FeatureExtractor()
    sample = list(list(volunteer_data.values())[0]['recordings'].values())[0]
    n_features = len(extractor.extract_all_features(sample))
    print(f"✓ {n_features} features")
 
    
 
    if (run_experiment):
        # Run Participation Rate Comparison
        all_results, comparison_df, output_dir = run_participation_rate_comparison(
            volunteer_data=volunteer_data,
            extractor=extractor,
            n_features=n_features,
            device=DEVICE,
            selection_methods=SELECTION_METHODS,
            fitness_modes=EXPERIMENT_FITNESS_MODES,
            participation_rates=PARTICIPATION_RATES,
            n_runs=N_RUNS,
            n_rounds=N_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            lr=LR,
            mu=MU,
            use_edge_sim=USE_EDGE_SIM,
            edge_sim_file=EDGE_SIM_FILE,
            output_base_dir=OUTPUT_BASE_DIR
        )
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        
        # Print summary
        print("\n" + "=" * 40)
        print("PARTICIPATION RATE COMPARISON SUMMARY")
        print("=" * 40)
        for rate in PARTICIPATION_RATES:
            rate_df = comparison_df[comparison_df['Participation_Rate'] == rate]
            print(f"\nRate {int(rate*100)}%:")
            print(f"  Avg Fitness:  {rate_df['Fitness_Avg'].mean():.4f}")
            print(f"  Avg Accuracy: {rate_df['Accuracy_Avg'].mean():.4f}")
            print(f"  Avg Energy:   {rate_df['Energy_Avg'].mean():.2f}")

        # ----------------------------------------------------------------
        # FedMCCS baseline (published-method comparison, R3-11).
        # Option A: run once per rate, independent of the fitness_mode
        # axis. Reuses the SAME output_dir so both CSVs and all plots
        # land together, and the SAME core hyperparameters (n_runs,
        # n_rounds, local_epochs, lr, mu, edge sim) as the main
        # metaheuristic comparison above, for a fair comparison.
        # ----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("RUNNING FEDMCCS BASELINE (published-method comparison)")
        print("=" * 80)

        fedmccs_df, _ = run_fedmccs_baseline_experiment(
            volunteer_data=volunteer_data,
            extractor=extractor,
            n_features=n_features,
            device=DEVICE,
            participation_rates=PARTICIPATION_RATES,
            n_runs=N_RUNS,
            n_rounds=N_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            lr=LR,
            mu=MU,
            use_edge_sim=USE_EDGE_SIM,
            edge_sim_file=EDGE_SIM_FILE,
            output_dir=output_dir,   # same folder as the main comparison
        )

        print("\n" + "=" * 40)
        print("FEDMCCS BASELINE SUMMARY")
        print("=" * 40)
        for rate in PARTICIPATION_RATES:
            row = fedmccs_df[fedmccs_df['Participation_Rate'] == rate]
            if len(row):
                print(f"\nRate {int(rate*100)}%:")
                print(f"  Accuracy:          {row['Accuracy_Avg'].values[0]:.4f}")
                print(f"  Fulfillment Rate:  {row['Fulfillment_Rate_Avg'].values[0]:.2f}")
                print(f"  Skipped (budget):  {row['Skipped_Resources_Avg'].values[0]:.1f} clients/round")

        # Merge with the metaheuristic comparison and generate combined plots
        combined_df = merge_with_metaheuristic_comparison(comparison_df, fedmccs_df, output_dir)
        generate_all_fedmccs_plots(comparison_df, fedmccs_df, output_dir)

        print(f"\nFedMCCS baseline results + combined CSV + plots saved to: {output_dir}")
    
    # Optionally run Byzantine experiment    
    if (run_byzantine): 
         print("\n" + "=" * 80)
         print("RUNNING BYZANTINE RESILIENCE EXPERIMENT")
         print("=" * 80)
         
         byz_df, byz_output_dir = run_byzantine_experiment(
             volunteer_data=volunteer_data,
             extractor=extractor,
             n_features=n_features,
             device=DEVICE,
             # UPDATED: GWO + DE, to empirically test whether GWO's lower
             # fitness-optimization variance (Part I, clean training) holds
             # up as a stability advantage specifically under attack -- see
             # plot_byzantine_method_comparison() for the new comparison
             # plots this produces.
             # RUN 2 OF 2: DE alone. GWO's convergence_data.pkl from the
             # original single-method run should already exist from your
             # earlier run's output_dir -- once this DE-only run finishes,
             # its convergence_data.pkl (same output_dir structure) can be
             # loaded alongside GWO's saved pickle to build a true overlay
             # comparison (both methods as separate lines in the same
             # subplot), rather than two separate side-by-side grids.
             selection_methods=['DE'],
             # NOTE: 'gaussian_noise' was listed here previously but is NOT
             # implemented in ByzantineAttackSimulator (see FederatedLearningBS.py)
             # -- only random/sign_flip/scaling are wired up. Passing an
             # unimplemented attack type will silently no-op or raise inside
             # the simulator depending on its default-case handling; removed
             # until it's actually implemented (needed to legitimately claim
             # more than three attack types per R3-12).
             attack_types=['none', 'random', 'sign_flip', 'scaling'],
             byzantine_fractions=[0.0, 0.1, 0.2, 0.3],
             defense_methods=['none', 'median', 'krum', 'combined'],
             n_runs=8,
             n_rounds=100,
             participation_rate=0.5,
             local_epochs=LOCAL_EPOCHS,
             lr=LR,
             mu=MU,
             use_edge_sim=USE_EDGE_SIM,
             edge_sim_file=EDGE_SIM_FILE,
             output_base_dir=OUTPUT_BASE_DIR
         )
         
         print(f"\nByzantine experiment results saved to: {byz_output_dir}")
