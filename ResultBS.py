# -*- coding: utf-8 -*-
"""
================================================================================
ResultBS.py - Results Management Library for BioSecure
================================================================================
Contains: MultiMethodExperiment class, plotting functions, CSV exports, statistics
================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import copy
import json
import time
import torch

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from FederatedLearningBS import EnhancedFLSystem, FLClientWrapper, FITNESS_MODES


class MultiMethodExperiment:
    """
    Run experiments with multiple selection methods AND fitness modes
    Computes statistics across N runs
    """
    
    def __init__(self, n_features, device='cpu'):
        self.n_features = n_features
        self.device = device
        self.results = {}
        self.fl_system = None
        self.prepared_client_data = None
        self.prepared_scaler = None
        self.prepared_fl_clients = None
        self.experiment_params = {}
        
    def prepare_data_once(self, volunteer_data, extractor):
        """Prepare client data once - to be reused across all runs"""
        print("\n" + "=" * 60)
        print("PREPARING DATA (ONE-TIME)")
        print("=" * 60)
        
        self.fl_system = EnhancedFLSystem(self.n_features, self.device)
   
        self.fl_system.prepare_client_data( volunteer_data,extractor, test_size=0.2, 
                                 min_samples_per_client=16, augment_data=False, 
                                 max_augment_ratio=2.0)
        
        self.prepared_client_data = copy.deepcopy(self.fl_system.client_data)
        self.prepared_scaler = copy.deepcopy(self.fl_system.scaler)
        self.prepared_fl_clients = copy.deepcopy(self.fl_system.fl_clients)
        self.n_features = self.fl_system.n_features
        
        print("✓ Data prepared and cached for all experiments")
        
    def run_experiment(self, volunteer_data, extractor, 
                      selection_methods=['RANDOM', 'PSO', 'GA', 'GWO', 'DE'],
                      fitness_modes=['learning'],  # Can now test multiple fitness modes!
                      n_runs=5, n_rounds=30, participation_rate=0.5,
                      local_epochs=5, lr=0.002, mu=0.03,
                      use_edge_sim=True, edge_sim_file=None):
        """
        Run FL experiments for each selection method AND fitness mode.
        
        Parameters:
        -----------
        selection_methods : list
            List of optimizer names: RANDOM, PSO, GA, GWO, DE, SA, TSA, RSELECT
        fitness_modes : list
            List of fitness modes: 'resource', 'learning', 'exploration'
        n_runs : int
            Number of runs for each combination
        """
        
        print("\n" + "=" * 80)
        print("MULTI-METHOD FEDERATED LEARNING EXPERIMENT")
        print("=" * 80)
        print(f"Selection Methods: {selection_methods}")
        print(f"Fitness Modes: {fitness_modes}")
        print(f"Number of Runs per Combination: {n_runs}")
        print(f"FL Rounds per Run: {n_rounds}")
        print("=" * 80)
        
        self.prepare_data_once(volunteer_data, extractor)
        
        # Store experiment parameters
        self.experiment_params = {
            'selection_methods': selection_methods,
            'fitness_modes': fitness_modes,
            'n_runs': n_runs,
            'n_rounds': n_rounds,
            'participation_rate': participation_rate,
            'local_epochs': local_epochs,
            'lr': lr,
            'mu': mu,
            'use_edge_sim': use_edge_sim,
            'edge_sim_file': edge_sim_file,
            'timestamp': datetime.now().strftime("%Y%m%d_%H-%M-%S")
        }
        
        # Create result keys as "METHOD_FITNESS" combinations
        all_combinations = []
        for method in selection_methods:
            for fitness_mode in fitness_modes:
                combo_key = f"{method}_{fitness_mode}"
                all_combinations.append((method, fitness_mode, combo_key))
                self.results[combo_key] = {
                    'acc_histories': [],
                    'loss_histories': [],
                    'final_metrics': [],
                    'selection_metrics': [],
                    'energy_histories': [],
                    'total_energy': [],
                    'avg_battery_histories': [],
                    'avg_reliability_histories': [],
                    'avg_latency_histories': [],
                    'convergence_rounds': [],
                    'client_utilization': [],
                    'fitness_histories': [],
                    'metaheuristic_curves': [],  # Store all iteration curves from all rounds
                    'all_predictions': [],
                    # Learning metrics
                    'avg_loss_improvement_histories': [],
                    'avg_update_norm_histories': [],
                    'avg_disagreement_histories': [],
                    # Metadata
                    'selection_method': method,
                    'fitness_mode': fitness_mode,
                }
        
        total_runs = len(all_combinations) * n_runs
        current_run = 0
        start_time = time.time()
        
        # Create progress bar if tqdm available
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_runs, desc="Experiment Progress", 
                       unit="run", ncols=100, leave=True,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for method, fitness_mode, combo_key in all_combinations:
            print(f"\n{'='*60}")
            print(f"SELECTION: {method} | FITNESS: {fitness_mode.upper()}")
            print(f"{'='*60}")
            
            for run in range(n_runs):
                current_run += 1
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if current_run > 1:
                    avg_time_per_run = elapsed / (current_run - 1)
                    remaining_runs = total_runs - current_run + 1
                    eta_seconds = avg_time_per_run * remaining_runs
                    eta_str = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                else:
                    eta_str = "ETA: calculating..."
                
                print(f"\n--- Run {run + 1}/{n_runs} (Overall: {current_run}/{total_runs}) | {eta_str} ---")
                
                seed = 42 + run * 100
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Create FL system with specific fitness mode
                fl = EnhancedFLSystem(self.n_features, self.device, fitness_mode=fitness_mode)
                fl.client_data = copy.deepcopy(self.prepared_client_data)
                fl.scaler = copy.deepcopy(self.prepared_scaler)
                fl.fl_clients = copy.deepcopy(self.prepared_fl_clients)
                fl.client_alphas = {cid: 0.5 for cid in fl.client_data}
                
                fl.initialize_models()
                
                if use_edge_sim:
                    fl.setup_edge_simulation(sim_file=edge_sim_file)
                
                history = fl.train_federated(
                    selection_method=method,
                    participation_rate=participation_rate,
                    n_rounds=n_rounds,
                    local_epochs=local_epochs,
                    lr=lr,
                    mu=mu,
                    verbose=False,
                    show_progress=True
                )
                
                fl.personalize_models(verbose=False)
                fl.learn_adaptive_alpha(verbose=False)
                metrics, predictions = fl.evaluate_all_with_predictions(verbose=False)
                
                # Store results
                self.results[combo_key]['acc_histories'].append(history['global_acc'])
                self.results[combo_key]['loss_histories'].append(history['global_loss'])
                self.results[combo_key]['final_metrics'].append(metrics)
                self.results[combo_key]['selection_metrics'].append(history.get('selection_metrics', []))
                self.results[combo_key]['energy_histories'].append(history.get('energy_consumed', []))
                self.results[combo_key]['total_energy'].append(history.get('total_energy', 0))
                self.results[combo_key]['avg_battery_histories'].append(history.get('avg_battery_level', []))
                self.results[combo_key]['avg_reliability_histories'].append(history.get('avg_reliability', []))
                self.results[combo_key]['avg_latency_histories'].append(history.get('avg_latency', []))
                self.results[combo_key]['convergence_rounds'].append(history.get('convergence_round', None))
                self.results[combo_key]['client_utilization'].append(history.get('client_utilization', 0))
                self.results[combo_key]['fitness_histories'].append(history.get('selection_fitness', []))
                self.results[combo_key]['metaheuristic_curves'].append(history.get('metaheuristic_curves', []))
                self.results[combo_key]['all_predictions'].append(predictions)
                
                # Learning metrics
                self.results[combo_key]['avg_loss_improvement_histories'].append(history.get('avg_loss_improvement', []))
                self.results[combo_key]['avg_update_norm_histories'].append(history.get('avg_update_norm', []))
                self.results[combo_key]['avg_disagreement_histories'].append(history.get('avg_disagreement', []))
                
                print(f"  Global Acc: {metrics['global']['accuracy']:.4f}, "
                      f"Energy: {history.get('total_energy', 0):.2f}")
                
                # Update progress bar
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix({'Method': method, 'Mode': fitness_mode, 'Acc': f"{metrics['global']['accuracy']:.3f}"})
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        # Print total time
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE!")
        print(f"Total time: {int(total_time//60)}m {int(total_time%60)}s")
        print(f"Average time per run: {total_time/total_runs:.1f}s")
        print(f"{'='*60}")
        
        return self.results
    
    def run_experiment_with_prepared_data(self, 
                      selection_methods=['RANDOM', 'PSO', 'GA', 'GWO', 'DE'],
                      fitness_modes=['learning'],
                      n_runs=5, n_rounds=30, participation_rate=0.5,
                      local_epochs=5, lr=0.002, mu=0.03,
                      use_edge_sim=True, edge_sim_file=None, shared_edge_env=None):
        """
        Run FL experiments using PRE-PREPARED client data (avoids re-processing).
        
        This method assumes self.prepared_client_data and self.prepared_scaler 
        are already set before calling.
        """
        
        print("\n" + "=" * 80)
        print("MULTI-METHOD FEDERATED LEARNING EXPERIMENT (WITH PRE-PREPARED DATA)")
        print("=" * 80)
        print(f"Selection Methods: {selection_methods}")
        print(f"Fitness Modes: {fitness_modes}")
        print(f"Number of Runs per Combination: {n_runs}")
        print(f"FL Rounds per Run: {n_rounds}")
        print(f"Participation Rate: {participation_rate}")
        print("=" * 80)
        
        # Store experiment parameters
        self.experiment_params = {
            'selection_methods': selection_methods,
            'fitness_modes': fitness_modes,
            'n_runs': n_runs,
            'n_rounds': n_rounds,
            'participation_rate': participation_rate,
            'local_epochs': local_epochs,
            'lr': lr,
            'mu': mu,
            'use_edge_sim': use_edge_sim,
            'edge_sim_file': edge_sim_file,
            'n_clients': len(self.prepared_client_data),
            'timestamp': datetime.now().strftime("%Y%m%d_%H-%M-%S")
        }
        
        # Create result keys as "METHOD_FITNESS" combinations
        all_combinations = []
        for method in selection_methods:
            for fitness_mode in fitness_modes:
                combo_key = f"{method}_{fitness_mode}"
                all_combinations.append((method, fitness_mode, combo_key))
                self.results[combo_key] = {
                    'acc_histories': [],
                    'loss_histories': [],
                    'final_metrics': [],
                    'selection_metrics': [],
                    'energy_histories': [],
                    'total_energy': [],
                    'avg_battery_histories': [],
                    'avg_reliability_histories': [],
                    'avg_latency_histories': [],
                    'convergence_rounds': [],
                    'client_utilization': [],
                    'fitness_histories': [],
                    'metaheuristic_curves': [],
                    'all_predictions': [],
                    'avg_loss_improvement_histories': [],
                    'avg_update_norm_histories': [],
                    'avg_disagreement_histories': [],
                    'selection_method': method,
                    'fitness_mode': fitness_mode,
                }
        
        total_runs = len(all_combinations) * n_runs
        current_run = 0
        start_time = time.time()
        
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_runs, desc="Experiment Progress", 
                       unit="run", ncols=100, leave=True,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for method, fitness_mode, combo_key in all_combinations:
            print(f"\n{'='*60}")
            print(f"SELECTION: {method} | FITNESS: {fitness_mode.upper()}")
            print(f"{'='*60}")
            
            for run in range(n_runs):
                current_run += 1
                
                seed = 42 + run * 100
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Create new FL system with specified fitness mode
                fl = EnhancedFLSystem(self.n_features, self.device, fitness_mode=fitness_mode)
                
                # REUSE PRE-PREPARED DATA (clone to avoid modifying original)
                fl.client_data = {k: {
                    'X_train': v['X_train'].clone(),
                    'y_train': v['y_train'].clone(),
                    'X_test': v['X_test'].clone(),
                    'y_test': v['y_test'].clone(),
                    'n_train': v['n_train']
                } for k, v in self.prepared_client_data.items()}
                fl.scaler = self.prepared_scaler
                
                # Initialize FL client wrappers
                for subj_id in fl.client_data:
                    fl.client_alphas[subj_id] = 0.5
                    fl.fl_clients[subj_id] = FLClientWrapper(subj_id, fl.client_data[subj_id])
                
                # Initialize models
                fl.initialize_models()
                
                # Setup edge simulation - use shared environment if available
                if shared_edge_env is not None:
                    fl.edge_env = shared_edge_env
                    fl.edge_env.SimMoveFirst()
                    fl._map_clients_to_devices()
                elif use_edge_sim and edge_sim_file:
                    fl.setup_edge_simulation(sim_file=edge_sim_file)
                
                # Train
                history = fl.train_federated(
                    selection_method=method,
                    participation_rate=participation_rate,
                    n_rounds=n_rounds,
                    local_epochs=local_epochs,
                    lr=lr,
                    mu=mu,
                    verbose=False,
                    show_progress=True
                )
                
                # Personalize and evaluate
                fl.personalize_models(verbose=False)
                fl.learn_adaptive_alpha(verbose=False)
                metrics, predictions = fl.evaluate_all_with_predictions(verbose=False)
                
                # Store results
                self.results[combo_key]['acc_histories'].append(history['global_acc'])
                self.results[combo_key]['loss_histories'].append(history['global_loss'])
                self.results[combo_key]['final_metrics'].append(metrics)
                self.results[combo_key]['selection_metrics'].append(history['selection_metrics'])
                self.results[combo_key]['energy_histories'].append(history['energy_consumed'])
                self.results[combo_key]['total_energy'].append(history.get('total_energy', 0))
                self.results[combo_key]['avg_battery_histories'].append(history['avg_battery_level'])
                self.results[combo_key]['avg_reliability_histories'].append(history['avg_reliability'])
                self.results[combo_key]['avg_latency_histories'].append(history['avg_latency'])
                self.results[combo_key]['convergence_rounds'].append(history['convergence_round'])
                self.results[combo_key]['client_utilization'].append(history.get('client_utilization', 0))
                self.results[combo_key]['fitness_histories'].append(history['selection_fitness'])
                self.results[combo_key]['metaheuristic_curves'].extend(history.get('metaheuristic_curves', []))
                self.results[combo_key]['all_predictions'].append(predictions)
                self.results[combo_key]['avg_loss_improvement_histories'].append(history['avg_loss_improvement'])
                self.results[combo_key]['avg_update_norm_histories'].append(history['avg_update_norm'])
                self.results[combo_key]['avg_disagreement_histories'].append(history['avg_disagreement'])
                
                print(f"  Run {run+1}/{n_runs}: Acc={metrics['global']['accuracy']:.4f}, "
                      f"Energy: {history.get('total_energy', 0):.2f}")
                
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix({'Method': method, 'Mode': fitness_mode, 'Acc': f"{metrics['global']['accuracy']:.3f}"})
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE!")
        print(f"Total time: {int(total_time//60)}m {int(total_time%60)}s")
        print(f"Average time per run: {total_time/total_runs:.1f}s")
        print(f"{'='*60}")
        
        return self.results
    
    def compute_statistics(self):
        """Compute comprehensive statistics for each method-fitness combination"""
        stats = {}
        
        def safe_stats(values):
            valid = [v for v in values if v is not None]
            if not valid:
                return {'min': 0, 'max': 0, 'avg': 0, 'std': 0}
            return {
                'min': np.min(valid),
                'max': np.max(valid),
                'avg': np.mean(valid),
                'std': np.std(valid)
            }
        
        for combo_key, data in self.results.items():
            global_accs = [m['global']['accuracy'] for m in data['final_metrics']]
            combined_accs = [m['combined_adaptive']['accuracy'] for m in data['final_metrics']]
            personalized_accs = [m['personalized']['accuracy'] for m in data['final_metrics']]
            global_f1s = [m['global']['f1'] for m in data['final_metrics']]
            
            total_energies = data.get('total_energy', [])
            
            avg_batteries = []
            avg_reliabilities = []
            avg_latencies = []
            for battery_hist in data.get('avg_battery_histories', []):
                if battery_hist:
                    avg_batteries.append(np.mean(battery_hist))
            for rel_hist in data.get('avg_reliability_histories', []):
                if rel_hist:
                    avg_reliabilities.append(np.mean(rel_hist))
            for lat_hist in data.get('avg_latency_histories', []):
                if lat_hist:
                    avg_latencies.append(np.mean(lat_hist))
            
            convergence_rounds = [r for r in data.get('convergence_rounds', []) if r is not None]
            client_utils = data.get('client_utilization', [])
            
            stats[combo_key] = {
                'global_accuracy': safe_stats(global_accs),
                'combined_accuracy': safe_stats(combined_accs),
                'personalized_accuracy': safe_stats(personalized_accs),
                'global_f1': safe_stats(global_f1s),
                'total_energy': safe_stats(total_energies),
                'avg_battery': safe_stats(avg_batteries),
                'avg_reliability': safe_stats(avg_reliabilities),
                'avg_latency': safe_stats(avg_latencies),
                'convergence_round': safe_stats(convergence_rounds) if convergence_rounds else {'min': None, 'max': None, 'avg': None, 'std': None},
                'convergence_rate': len(convergence_rounds) / len(global_accs) if global_accs else 0,
                'client_utilization': safe_stats(client_utils),
                'accuracy_per_energy': safe_stats([
                    global_accs[i] / total_energies[i] if total_energies[i] > 0 else 0 
                    for i in range(min(len(global_accs), len(total_energies)))
                ]) if total_energies else {'min': 0, 'max': 0, 'avg': 0, 'std': 0},
                'n_runs': len(global_accs),
                'selection_method': data.get('selection_method', ''),
                'fitness_mode': data.get('fitness_mode', ''),
            }
        
        return stats
    
    def print_summary(self, stats=None):
        """Print comprehensive summary tables"""
        if stats is None:
            stats = self.compute_statistics()
        
        print("\n" + "=" * 120)
        print("PERFORMANCE SUMMARY BY SELECTION METHOD AND FITNESS MODE")
        print("=" * 120)
        
        acc_data = []
        for combo_key, s in stats.items():
            acc_data.append({
                'Method': combo_key,
                'Fitness': s.get('fitness_mode', ''),
                'Global Acc': f"{s['global_accuracy']['avg']:.4f}±{s['global_accuracy']['std']:.4f}",
                'Combined Acc': f"{s['combined_accuracy']['avg']:.4f}",
                'Personal Acc': f"{s['personalized_accuracy']['avg']:.4f}",
                'F1': f"{s['global_f1']['avg']:.4f}",
                'Energy': f"{s['total_energy']['avg']:.2f}",
            })
        
        df_acc = pd.DataFrame(acc_data)
        print(df_acc.to_string(index=False))
        
        print("\n" + "=" * 120)
        print("BEST METHODS SUMMARY")
        print("=" * 120)
        
        best_acc = max(stats.keys(), key=lambda m: stats[m]['global_accuracy']['avg'])
        print(f"✓ Best by Global Accuracy: {best_acc} ({stats[best_acc]['global_accuracy']['avg']:.4f})")
        
        best_energy = min(stats.keys(), key=lambda m: stats[m]['total_energy']['avg'])
        print(f"✓ Most Energy Efficient: {best_energy} ({stats[best_energy]['total_energy']['avg']:.2f})")
        
        best_efficiency = max(stats.keys(), key=lambda m: stats[m]['accuracy_per_energy']['avg'])
        print(f"✓ Best Accuracy/Energy: {best_efficiency} ({stats[best_efficiency]['accuracy_per_energy']['avg']:.4f})")
        
        return df_acc
    
    def plot_global_statistics_by_fitness(self, fitness_mode, output_dir, save=True):
        """
        Plot global statistics for a SINGLE fitness mode using SELECTION FITNESS values
        - Converts negative fitness to positive (higher = better)
        - Includes accuracy/loss convergence curves
        - Adjusts y-scale to emphasize differences (not starting from 0)
        - No min/max annotations on bars
        """
        # Filter results for this fitness mode
        filtered_results = {
            k: v for k, v in self.results.items() 
            if v.get('fitness_mode') == fitness_mode
        }
        
        if not filtered_results:
            print(f"  No results found for fitness mode: {fitness_mode}")
            return None
        
        methods = [v['selection_method'] for v in filtered_results.values()]
        methods = list(dict.fromkeys(methods))  # Unique, preserve order
        n_methods = len(methods)
        
        # Get number of rounds from first result
        first_key = list(filtered_results.keys())[0]
        n_rounds = len(filtered_results[first_key]['fitness_histories'][0]) if filtered_results[first_key]['fitness_histories'] else 0
        n_runs = len(filtered_results[first_key]['fitness_histories'])
        
        if n_rounds == 0:
            print(f"  No fitness data for fitness mode: {fitness_mode}")
            return None
        
        # Get fitness mode info
        mode_info = FITNESS_MODES.get(fitness_mode, {})
        mode_desc = mode_info.get('description', fitness_mode)
        mode_color = mode_info.get('color', '#333333')
        
        # Create figure with more subplots
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle(f'Selection Performance - {mode_desc}\nFitness Mode: {fitness_mode.upper()}', 
                    fontsize=14, fontweight='bold', color=mode_color)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        
        # Helper function to convert negative fitness to positive (higher = better)
        def to_positive_fitness(fitness_val):
            return -fitness_val if fitness_val < 0 else fitness_val
        
        # 1. Rankings heatmap across rounds (top) - using POSITIVE FITNESS values
        ax1 = fig.add_subplot(4, 2, (1, 2))
        
        rankings_matrix = np.zeros((n_methods, n_rounds))
        
        for rnd in range(n_rounds):
            round_fitness = []
            for method in methods:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in filtered_results:
                    fitness_at_round = []
                    for run_hist in filtered_results[combo_key]['fitness_histories']:
                        if rnd < len(run_hist):
                            fitness_at_round.append(to_positive_fitness(run_hist[rnd]))
                    avg_fitness = np.mean(fitness_at_round) if fitness_at_round else 0
                    round_fitness.append(avg_fitness)
                else:
                    round_fitness.append(0)
            
            # Rank (1 = best = highest positive fitness)
            ranks = rankdata(-np.array(round_fitness))
            rankings_matrix[:, rnd] = ranks
        
        im = ax1.imshow(rankings_matrix, cmap='YlGnBu', aspect='auto')
        ax1.set_yticks(range(n_methods))
        ax1.set_yticklabels(methods)
        ax1.set_xlabel('Time Step (Round)')
        ax1.set_ylabel('Optimizer')
        ax1.set_title(f'Optimizer Rankings Across Time Steps\n(Based on Selection Fitness - Rank 1 = Best)', fontsize=11)
        
        for i in range(n_methods):
            for j in range(0, n_rounds, max(1, n_rounds//15)):
                ax1.text(j, i, f'{int(rankings_matrix[i, j])}', ha='center', va='center', fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Rank (1=best)')
        
        # 2. Average POSITIVE Fitness Convergence Curves
        ax2 = fig.add_subplot(4, 2, 3)
        
        all_fitness_values = []
        for idx, method in enumerate(methods):
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                fitness_histories = filtered_results[combo_key]['fitness_histories']
                if fitness_histories:
                    min_len = min(len(h) for h in fitness_histories if h)
                    if min_len > 0:
                        aligned = [[to_positive_fitness(v) for v in h[:min_len]] for h in fitness_histories if h]
                        avg_fitness = np.mean(aligned, axis=0)
                        all_fitness_values.extend(avg_fitness)
                        rounds = np.arange(1, len(avg_fitness) + 1)
                        ax2.plot(rounds, avg_fitness, label=method, color=colors[idx], linewidth=1,  markersize=2)
        
        ax2.set_xlabel('FL Round')
        ax2.set_ylabel('Fitness Value (Higher = Better)')
        ax2.set_title('Average Selection Fitness Convergence\n(Across FL Rounds)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        if all_fitness_values:
            y_min, y_max = min(all_fitness_values), max(all_fitness_values)
            y_margin = (y_max - y_min) * 0.1
            ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # 3. Global Accuracy Convergence Curves
        ax3 = fig.add_subplot(4, 2, 4)
        
        all_acc_values = []
        for idx, method in enumerate(methods):
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                acc_histories = filtered_results[combo_key]['acc_histories']
                if acc_histories:
                    min_len = min(len(h) for h in acc_histories if h)
                    if min_len > 0:
                        aligned = [h[:min_len] for h in acc_histories if h]
                        avg_acc = np.mean(aligned, axis=0)
                        all_acc_values.extend(avg_acc)
                        rounds = np.arange(1, len(avg_acc) + 1)
                        ax3.plot(rounds, avg_acc, label=method, color=colors[idx], linewidth=1, markersize=2)
        
        ax3.set_xlabel('FL Round')
        ax3.set_ylabel('Global Accuracy')
        ax3.set_title('Global Model Accuracy Convergence\n(Across FL Rounds)', fontsize=11)
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        if all_acc_values:
            y_min, y_max = min(all_acc_values), max(all_acc_values)
            y_margin = (y_max - y_min) * 0.1
            ax3.set_ylim(max(0, y_min - y_margin), min(1, y_max + y_margin))
        
        # 4. Global Loss Convergence Curves
        ax4 = fig.add_subplot(4, 2, 5)
        
        all_loss_values = []
        for idx, method in enumerate(methods):
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                loss_histories = filtered_results[combo_key]['loss_histories']
                if loss_histories:
                    min_len = min(len(h) for h in loss_histories if h)
                    if min_len > 0:
                        aligned = [h[:min_len] for h in loss_histories if h]
                        avg_loss = np.mean(aligned, axis=0)
                        all_loss_values.extend(avg_loss)
                        rounds = np.arange(1, len(avg_loss) + 1)
                        ax4.plot(rounds, avg_loss, label=method, color=colors[idx], linewidth=1, alpha=0.9, markersize=2)
        
        ax4.set_xlabel('FL Round')
        ax4.set_ylabel('Global Loss')
        ax4.set_title('Global Model Loss Convergence\n(Across FL Rounds)', fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        if all_loss_values:
            y_min, y_max = min(all_loss_values), max(all_loss_values)
            y_margin = (y_max - y_min) * 0.1
            ax4.set_ylim(max(0, y_min - y_margin), y_max + y_margin)
        
        # 5. Performance Distribution Box Plot - using POSITIVE final fitness
        ax5 = fig.add_subplot(4, 2, 6)
        
        fitness_data = []
        method_labels = []
        for method in methods:
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                final_fitness = [to_positive_fitness(h[-1]) for h in filtered_results[combo_key]['fitness_histories'] if h]
                if final_fitness:
                    fitness_data.append(final_fitness)
                    method_labels.append(method)
        
        if fitness_data:
            bp = ax5.boxplot(fitness_data, tick_labels=method_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(method_labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            all_box_values = [v for sublist in fitness_data for v in sublist]
            if all_box_values:
                y_min, y_max = min(all_box_values), max(all_box_values)
                y_margin = (y_max - y_min) * 0.15
                ax5.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax5.set_xlabel('Optimizer')
        ax5.set_ylabel('Fitness Value (Higher = Better)')
        ax5.set_title('Fitness Distribution\n(Final Selection Fitness)', fontsize=11)
        ax5.tick_params(axis='x', rotation=25)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Summary Table
        ax6 = fig.add_subplot(4, 2, (7, 8))
        ax6.axis('off')
        
        table_data = []
        for method in methods:
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                final_fitness = [to_positive_fitness(h[-1]) for h in filtered_results[combo_key]['fitness_histories'] if h]
                final_accuracy = [m['global']['accuracy'] for m in filtered_results[combo_key]['final_metrics']]
                
                if final_fitness:
                    table_data.append([
                        method,
                        f'{np.mean(final_fitness):.4f}',
                        f'{np.max(final_fitness):.4f}',
                        f'{np.std(final_fitness):.4f}',
                        f'{np.mean(final_accuracy):.4f}' if final_accuracy else 'N/A'
                    ])
        
        if table_data:
            table = ax6.table(cellText=table_data,
                             colLabels=['Optimizer', 'Avg Fitness', 'Best Fitness', 'Std Dev', 'Avg Accuracy'],
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        ax6.set_title(f'Overall Performance Summary - {fitness_mode.upper()} Fitness\n(Higher Fitness Value = Better Selection)', 
                     fontsize=11, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            save_path = f'{output_dir}/global_statistics_{fitness_mode}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved global statistics plot for {fitness_mode} to {save_path}")
        
        plt.close()
        return fig
    
    def plot_metaheuristic_convergence_curves(self, fitness_mode, output_dir, save=True):
        """
        Plot average fitness convergence curves of metaheuristic search iterations.
        X-axis = metaheuristic iterations (e.g., 50 iterations)
        Y-axis = fitness value (averaged across all runs and rounds)
        One plot per fitness mode showing all selection methods.
        """
        filtered_results = {
            k: v for k, v in self.results.items() 
            if v.get('fitness_mode') == fitness_mode
        }
        
        if not filtered_results:
            print(f"  No results found for fitness mode: {fitness_mode}")
            return None
        
        methods = list(dict.fromkeys([v['selection_method'] for v in filtered_results.values()]))
        n_methods = len(methods)
        
        mode_info = FITNESS_MODES.get(fitness_mode, {})
        mode_desc = mode_info.get('description', fitness_mode)
        mode_color = mode_info.get('color', '#333333')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Metaheuristic Optimization Convergence - {mode_desc}\nFitness Mode: {fitness_mode.upper()}', 
                    fontsize=14, fontweight='bold', color=mode_color)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        
        def to_positive_fitness(val):
            return -val if val < 0 else val
        
        all_fitness_values = []
        
        for idx, method in enumerate(methods):
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                # Get all metaheuristic curves from all runs
                all_curves_for_method = []
                
                meta_curves_per_run = filtered_results[combo_key].get('metaheuristic_curves', [])
                
                for run_curves in meta_curves_per_run:
                    # run_curves is a list of curves (one per FL round)
                    if isinstance(run_curves, list):
                        for curve in run_curves:
                            if curve is not None and len(curve) > 0:
                                positive_curve = [to_positive_fitness(v) for v in curve]
                                all_curves_for_method.append(positive_curve)
                    elif run_curves is not None and len(run_curves) > 0:
                        # Single curve
                        positive_curve = [to_positive_fitness(v) for v in run_curves]
                        all_curves_for_method.append(positive_curve)
                
                # If no metaheuristic curves, fall back to fitness_histories
                if not all_curves_for_method:
                    fitness_hists = filtered_results[combo_key].get('fitness_histories', [])
                    for hist in fitness_hists:
                        if hist and len(hist) > 0:
                            positive_curve = [to_positive_fitness(v) for v in hist]
                            all_curves_for_method.append(positive_curve)
                
                if all_curves_for_method:
                    # Pad curves to same length and average
                    target_len = max(len(c) for c in all_curves_for_method)
                    padded_curves = []
                    
                    for curve in all_curves_for_method:
                        if len(curve) < target_len:
                            padded = curve + [curve[-1]] * (target_len - len(curve))
                        else:
                            padded = curve[:target_len]
                        padded_curves.append(padded)
                    
                    avg_curve = np.mean(padded_curves, axis=0)
                    std_curve = np.std(padded_curves, axis=0)
                    all_fitness_values.extend(avg_curve)
                    
                    iterations = np.arange(1, len(avg_curve) + 1)
                    ax.plot(iterations, avg_curve, label=method, color=colors[idx], 
                           linewidth=2, marker='o', markersize=3, markevery=max(1, len(iterations)//10))
        
        ax.set_xlabel('Metaheuristic Iteration', fontsize=12)
        ax.set_ylabel('Fitness Value (Higher = Better)', fontsize=12)
        ax.set_title('Average Fitness Convergence\n(Averaged over all runs and FL rounds)', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if all_fitness_values:
            y_min, y_max = min(all_fitness_values), max(all_fitness_values)
            y_margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        plt.tight_layout()
        
        if save:
            save_path = f'{output_dir}/metaheuristic_convergence_{fitness_mode}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved metaheuristic convergence plot for {fitness_mode} to {save_path}")
        
        plt.close()
        return fig

    def plot_fitness_comparison_across_modes(self, output_dir, save=True):
        """
        Create final comparison plot across all three fitness modes
        Uses POSITIVE fitness values (higher = better)
        No min/max annotations, adjusted y-scale
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        if len(fitness_modes) < 2:
            print("  Need at least 2 fitness modes for comparison")
            return None
        
        def to_positive_fitness(val):
            return -val if val < 0 else val
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparison Across Fitness Modes\n(Higher Fitness = Better)', fontsize=14, fontweight='bold')
        
        # Prepare data with POSITIVE fitness
        comparison_data = {mode: {} for mode in fitness_modes}
        
        for combo_key, data in self.results.items():
            method = data.get('selection_method')
            mode = data.get('fitness_mode')
            if method and mode:
                final_fitness = [to_positive_fitness(h[-1]) for h in data.get('fitness_histories', []) if h]
                if final_fitness:
                    comparison_data[mode][method] = {
                        'avg': np.mean(final_fitness),
                        'std': np.std(final_fitness),
                        'min': np.min(final_fitness),
                        'max': np.max(final_fitness)
                    }
        
        # Plot 1: Grouped bar chart - GROUPED BY FITNESS MODE
        ax1 = axes[0, 0]
        x = np.arange(len(fitness_modes))  # X-axis is fitness modes
        width = 0.8 / len(selection_methods)  # Bar width based on number of methods
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selection_methods)))
        
        all_avgs = []
        for i, method in enumerate(selection_methods):
            avgs = [comparison_data[mode].get(method, {}).get('avg', 0) for mode in fitness_modes]
            stds = [comparison_data[mode].get(method, {}).get('std', 0) for mode in fitness_modes]
            all_avgs.extend(avgs)
            offset = (i - len(selection_methods)/2 + 0.5) * width
            ax1.bar(x + offset, avgs, width, yerr=stds, label=method, 
                   color=colors[i], alpha=0.8, capsize=2)
        
        ax1.set_xlabel('Fitness Mode')
        ax1.set_ylabel('Average Fitness (Higher = Better)')
        ax1.set_title('Average Selection Fitness\n(Grouped by Fitness Mode)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in fitness_modes])
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        if all_avgs:
            y_min, y_max = min(all_avgs), max(all_avgs)
            y_margin = (y_max - y_min) * 0.15
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Plot 2: Heatmap of rankings
        ax2 = axes[0, 1]
        
        rank_matrix = np.zeros((len(selection_methods), len(fitness_modes)))
        for j, mode in enumerate(fitness_modes):
            avgs = [comparison_data[mode].get(m, {}).get('avg', 0) for m in selection_methods]
            ranks = rankdata(-np.array(avgs))  # Higher is better
            rank_matrix[:, j] = ranks
        
        im = ax2.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto')
        ax2.set_xticks(range(len(fitness_modes)))
        ax2.set_xticklabels([m.upper() for m in fitness_modes])
        ax2.set_yticks(range(len(selection_methods)))
        ax2.set_yticklabels(selection_methods)
        ax2.set_xlabel('Fitness Mode')
        ax2.set_ylabel('Selection Method')
        ax2.set_title('Method Rankings by Fitness Mode\n(1=Best)')
        
        for i in range(len(selection_methods)):
            for j in range(len(fitness_modes)):
                ax2.text(j, i, f'{int(rank_matrix[i, j])}', ha='center', va='center', fontsize=10)
        
        plt.colorbar(im, ax=ax2, label='Rank')
        
        # Plot 3: Best method per fitness mode
        ax3 = axes[1, 0]
        
        best_methods, best_fitness, mode_colors = [], [], []
        for mode in fitness_modes:
            mode_data = comparison_data[mode]
            if mode_data:
                best_method = max(mode_data.keys(), key=lambda m: mode_data[m]['avg'])
                best_methods.append(f"{best_method}\n({mode.upper()})")
                best_fitness.append(mode_data[best_method]['avg'])
                mode_colors.append(FITNESS_MODES.get(mode, {}).get('color', '#333333'))
        
        bars = ax3.bar(range(len(best_methods)), best_fitness, color=mode_colors, alpha=0.8)
        ax3.set_xticks(range(len(best_methods)))
        ax3.set_xticklabels(best_methods)
        ax3.set_ylabel('Best Fitness Value (Higher = Better)')
        ax3.set_title('Best Performing Method per Fitness Mode')
        ax3.grid(True, alpha=0.3, axis='y')
        
        if best_fitness:
            y_min, y_max = min(best_fitness), max(best_fitness)
            y_margin = (y_max - y_min) * 0.15
            ax3.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        for method in selection_methods:
            row = [method]
            for mode in fitness_modes:
                data = comparison_data[mode].get(method, {})
                if data:
                    row.append(f"{data['avg']:.4f}±{data['std']:.4f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        best_row = ['BEST']
        for mode in fitness_modes:
            mode_data = comparison_data[mode]
            if mode_data:
                best_method = max(mode_data.keys(), key=lambda m: mode_data[m]['avg'])
                best_row.append(f"{best_method}")
            else:
                best_row.append("N/A")
        table_data.append(best_row)
        
        col_labels = ['Method'] + [f'{m.upper()}\nFitness' for m in fitness_modes]
        table = ax4.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for j in range(len(col_labels)):
            table[(len(table_data), j)].set_facecolor('#90EE90')
        
        ax4.set_title('Fitness Comparison Summary\n(Avg±Std, Higher = Better)', fontsize=11, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            save_path = f'{output_dir}/fitness_modes_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved fitness modes comparison to {save_path}")
        
        plt.close()
        return fig

    def plot_roc_curves_by_fitness(self, fitness_mode, output_dir, save=True):
        """
        Plot ROC curves for all selection methods under a specific fitness mode.
        Shows Global, Personalized, and Combined model ROC curves.
        
        Args:
            fitness_mode: 'resource', 'learning', or 'exploration'
            output_dir: Directory to save plots
            save: Whether to save the plot
        """
        from sklearn.metrics import roc_curve, auc
        
        filtered_results = {
            k: v for k, v in self.results.items() 
            if v.get('fitness_mode') == fitness_mode
        }
        
        if not filtered_results:
            print(f"  No results found for fitness mode: {fitness_mode}")
            return None
        
        methods = list(dict.fromkeys([v['selection_method'] for v in filtered_results.values()]))
        n_methods = len(methods)
        
        mode_info = FITNESS_MODES.get(fitness_mode, {})
        mode_desc = mode_info.get('description', fitness_mode)
        mode_color = mode_info.get('color', '#333333')
        
        # Create figure with 3 subplots: Global, Personalized, Combined
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'ROC Curves - {mode_desc}\nFitness Mode: {fitness_mode.upper()}', 
                    fontsize=14, fontweight='bold', color=mode_color)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        model_types = ['global', 'personalized', 'combined_adaptive']
        model_titles = ['Global Model', 'Personalized Model', 'Combined Adaptive Model']
        
        for ax_idx, (model_type, model_title) in enumerate(zip(model_types, model_titles)):
            ax = axes[ax_idx]
            
            for idx, method in enumerate(methods):
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in filtered_results:
                    all_preds = filtered_results[combo_key].get('all_predictions', [])
                    
                    # Aggregate predictions across all runs
                    y_true_all, y_prob_all = [], []
                    for pred in all_preds:
                        if model_type in pred and 'y_true' in pred[model_type] and 'y_prob' in pred[model_type]:
                            y_true_all.extend(pred[model_type]['y_true'])
                            y_prob_all.extend(pred[model_type]['y_prob'])
                    
                    if len(y_true_all) > 0 and len(y_prob_all) > 0:
                        y_true_arr = np.array(y_true_all)
                        y_prob_arr = np.array(y_prob_all)
                        
                        # Compute ROC curve
                        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
                        roc_auc = auc(fpr, tpr)
                        
                        ax.plot(fpr, tpr, color=colors[idx], linewidth=1,
                               label=f'{method} (AUC = {roc_auc:.4f})')
            
            # Plot diagonal (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title(model_title, fontsize=12)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            save_path = f'{output_dir}/roc_curves_{fitness_mode}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved ROC curves for {fitness_mode} to {save_path}")
        
        plt.close()
        return fig
    
    def plot_roc_comparison_all_modes(self, output_dir, save=True):
        """
        Plot ROC curves comparing all fitness modes for the Global model.
        One subplot per selection method.
        """
        from sklearn.metrics import roc_curve, auc
        
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        n_methods = len(selection_methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle('ROC Curves Comparison Across Fitness Modes\n(Global Model)', 
                    fontsize=14, fontweight='bold')
        
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_methods > 1 else axes
        
        for idx, method in enumerate(selection_methods):
            ax = axes[idx] if n_methods > 1 else axes[0]
            
            for mode in fitness_modes:
                combo_key = f"{method}_{mode}"
                if combo_key in self.results:
                    all_preds = self.results[combo_key].get('all_predictions', [])
                    
                    y_true_all, y_prob_all = [], []
                    for pred in all_preds:
                        if 'global' in pred and 'y_true' in pred['global'] and 'y_prob' in pred['global']:
                            y_true_all.extend(pred['global']['y_true'])
                            y_prob_all.extend(pred['global']['y_prob'])
                    
                    if len(y_true_all) > 0 and len(y_prob_all) > 0:
                        y_true_arr = np.array(y_true_all)
                        y_prob_arr = np.array(y_prob_all)
                        
                        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
                        roc_auc = auc(fpr, tpr)
                        
                        mode_color = FITNESS_MODES.get(mode, {}).get('color', '#333333')
                        ax.plot(fpr, tpr, color=mode_color, linewidth=1,
                               label=f'{mode.upper()} (AUC = {roc_auc:.4f})')
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'{method}', fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes) if isinstance(axes, np.ndarray) else 1):
            if isinstance(axes, np.ndarray):
                axes[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            save_path = f'{output_dir}/roc_curves_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved ROC curves comparison to {save_path}")
        
        plt.close()
        return fig
    
    def save_auc_summary(self, output_dir):
        """Save AUC values for all methods, fitness modes, and model types to CSV"""
        from sklearn.metrics import roc_auc_score
        
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        model_types = ['global', 'personalized', 'combined_adaptive']
        
        auc_data = []
        
        for method in selection_methods:
            for fitness_mode in fitness_modes:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in self.results:
                    all_preds = self.results[combo_key].get('all_predictions', [])
                    
                    for model_type in model_types:
                        y_true_all, y_prob_all = [], []
                        for pred in all_preds:
                            if model_type in pred and 'y_true' in pred[model_type] and 'y_prob' in pred[model_type]:
                                y_true_all.extend(pred[model_type]['y_true'])
                                y_prob_all.extend(pred[model_type]['y_prob'])
                        
                        if len(y_true_all) > 0 and len(set(y_true_all)) > 1:
                            try:
                                auc_val = roc_auc_score(y_true_all, y_prob_all)
                            except:
                                auc_val = 0.0
                        else:
                            auc_val = 0.0
                        
                        auc_data.append({
                            'Method': method,
                            'Fitness_Mode': fitness_mode,
                            'Model_Type': model_type,
                            'AUC': auc_val
                        })
        
        if auc_data:
            df = pd.DataFrame(auc_data)
            df.to_csv(f'{output_dir}/auc_summary.csv', index=False)
            print(f"✓ Saved AUC summary to {output_dir}/auc_summary.csv")
            
            # Also create a pivot table
            pivot_df = df.pivot_table(index=['Method', 'Model_Type'], 
                                      columns='Fitness_Mode', 
                                      values='AUC')
            pivot_df.to_csv(f'{output_dir}/auc_summary_pivot.csv')
            print(f"✓ Saved AUC pivot table to {output_dir}/auc_summary_pivot.csv")
        
        return auc_data

    def save_fitness_comparison_tables(self, output_dir):
        """
        Save CSV tables comparing performance across fitness modes
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        def to_positive_fitness(val):
            return -val if val < 0 else val
        
        # Table 1: Detailed comparison
        detailed_rows = []
        for combo_key, data in self.results.items():
            method = data.get('selection_method')
            mode = data.get('fitness_mode')
            
            final_fitness = [to_positive_fitness(h[-1]) for h in data.get('fitness_histories', []) if h]
            final_accuracy = [m['global']['accuracy'] for m in data.get('final_metrics', [])]
            
            if final_fitness:
                detailed_rows.append({
                    'Selection_Method': method,
                    'Fitness_Mode': mode,
                    'Fitness_Avg': np.mean(final_fitness),
                    'Fitness_Std': np.std(final_fitness),
                    'Fitness_Min': np.min(final_fitness),
                    'Fitness_Max': np.max(final_fitness),
                    'Accuracy_Avg': np.mean(final_accuracy) if final_accuracy else 0,
                    'Accuracy_Std': np.std(final_accuracy) if final_accuracy else 0,
                })
        
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(f'{output_dir}/fitness_comparison_detailed.csv', index=False)
        print(f"✓ Saved detailed fitness comparison to {output_dir}/fitness_comparison_detailed.csv")
        
        # Table 2: Pivot table - Methods vs Fitness Modes
        pivot_data = []
        for method in selection_methods:
            row = {'Method': method}
            for mode in fitness_modes:
                combo_key = f"{method}_{mode}"
                if combo_key in self.results:
                    final_fitness = [to_positive_fitness(h[-1]) for h in self.results[combo_key].get('fitness_histories', []) if h]
                    if final_fitness:
                        row[f'{mode}_fitness_avg'] = np.mean(final_fitness)
                        row[f'{mode}_fitness_std'] = np.std(final_fitness)
            pivot_data.append(row)
        
        df_pivot = pd.DataFrame(pivot_data)
        df_pivot.to_csv(f'{output_dir}/fitness_comparison_pivot.csv', index=False)
        print(f"✓ Saved pivot fitness comparison to {output_dir}/fitness_comparison_pivot.csv")
        
        # Table 3: Rankings per fitness mode
        rankings_data = []
        for mode in fitness_modes:
            mode_results = {}
            for method in selection_methods:
                combo_key = f"{method}_{mode}"
                if combo_key in self.results:
                    final_fitness = [to_positive_fitness(h[-1]) for h in self.results[combo_key].get('fitness_histories', []) if h]
                    if final_fitness:
                        mode_results[method] = np.mean(final_fitness)
            
            if mode_results:
                # Higher fitness is better now
                sorted_methods = sorted(mode_results.keys(), key=lambda m: mode_results[m], reverse=True)
                for rank, method in enumerate(sorted_methods, 1):
                    rankings_data.append({
                        'Fitness_Mode': mode,
                        'Rank': rank,
                        'Method': method,
                        'Avg_Fitness': mode_results[method]
                    })
        
        df_rankings = pd.DataFrame(rankings_data)
        df_rankings.to_csv(f'{output_dir}/fitness_rankings_by_mode.csv', index=False)
        print(f"✓ Saved rankings by fitness mode to {output_dir}/fitness_rankings_by_mode.csv")
        
        return df_detailed, df_pivot, df_rankings
        """
        Save CSV tables comparing performance across fitness modes
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        # Table 1: Detailed comparison
        detailed_rows = []
        for combo_key, data in self.results.items():
            method = data.get('selection_method')
            mode = data.get('fitness_mode')
            
            final_fitness = []
            for hist in data.get('fitness_histories', []):
                if hist:
                    final_fitness.append(hist[-1])
            
            final_accuracy = [m['global']['accuracy'] for m in data.get('final_metrics', [])]
            
            if final_fitness:
                detailed_rows.append({
                    'Selection_Method': method,
                    'Fitness_Mode': mode,
                    'Fitness_Avg': np.mean(final_fitness),
                    'Fitness_Std': np.std(final_fitness),
                    'Fitness_Min': np.min(final_fitness),
                    'Fitness_Max': np.max(final_fitness),
                    'Accuracy_Avg': np.mean(final_accuracy) if final_accuracy else 0,
                    'Accuracy_Std': np.std(final_accuracy) if final_accuracy else 0,
                })
        
        df_detailed = pd.DataFrame(detailed_rows)
        df_detailed.to_csv(f'{output_dir}/fitness_comparison_detailed.csv', index=False)
        print(f"✓ Saved detailed fitness comparison to {output_dir}/fitness_comparison_detailed.csv")
        
        # Table 2: Pivot table - Methods vs Fitness Modes
        pivot_data = []
        for method in selection_methods:
            row = {'Method': method}
            for mode in fitness_modes:
                combo_key = f"{method}_{mode}"
                if combo_key in self.results:
                    final_fitness = []
                    for hist in self.results[combo_key].get('fitness_histories', []):
                        if hist:
                            final_fitness.append(hist[-1])
                    if final_fitness:
                        row[f'{mode}_fitness_avg'] = np.mean(final_fitness)
                        row[f'{mode}_fitness_std'] = np.std(final_fitness)
            pivot_data.append(row)
        
        df_pivot = pd.DataFrame(pivot_data)
        df_pivot.to_csv(f'{output_dir}/fitness_comparison_pivot.csv', index=False)
        print(f"✓ Saved pivot fitness comparison to {output_dir}/fitness_comparison_pivot.csv")
        
        # Table 3: Rankings per fitness mode
        rankings_data = []
        for mode in fitness_modes:
            mode_results = {}
            for method in selection_methods:
                combo_key = f"{method}_{mode}"
                if combo_key in self.results:
                    final_fitness = []
                    for hist in self.results[combo_key].get('fitness_histories', []):
                        if hist:
                            final_fitness.append(hist[-1])
                    if final_fitness:
                        mode_results[method] = np.mean(final_fitness)
            
            if mode_results:
                sorted_methods = sorted(mode_results.keys(), key=lambda m: mode_results[m])
                for rank, method in enumerate(sorted_methods, 1):
                    rankings_data.append({
                        'Fitness_Mode': mode,
                        'Rank': rank,
                        'Method': method,
                        'Avg_Fitness': mode_results[method]
                    })
        
        df_rankings = pd.DataFrame(rankings_data)
        df_rankings.to_csv(f'{output_dir}/fitness_rankings_by_mode.csv', index=False)
        print(f"✓ Saved rankings by fitness mode to {output_dir}/fitness_rankings_by_mode.csv")
        
        return df_detailed, df_pivot, df_rankings
    
    # ==========================================================================
    # COMPREHENSIVE STATISTICS FOR PAPER
    # ==========================================================================
    
    def plot_accuracy_statistics_by_fitness(self, fitness_mode, output_dir, save=True):
        """
        Plot accuracy statistics similar to fitness plots - for Global Model Accuracy
        Includes: Rankings heatmap, convergence curves, box plots, summary table
        """
        filtered_results = {
            k: v for k, v in self.results.items() 
            if v.get('fitness_mode') == fitness_mode
        }
        
        if not filtered_results:
            print(f"  No results found for fitness mode: {fitness_mode}")
            return None
        
        methods = list(dict.fromkeys([v['selection_method'] for v in filtered_results.values()]))
        n_methods = len(methods)
        
        first_key = list(filtered_results.keys())[0]
        n_rounds = len(filtered_results[first_key]['acc_histories'][0]) if filtered_results[first_key]['acc_histories'] else 0
        
        if n_rounds == 0:
            print(f"  No accuracy data for fitness mode: {fitness_mode}")
            return None
        
        mode_info = FITNESS_MODES.get(fitness_mode, {})
        mode_desc = mode_info.get('description', fitness_mode)
        mode_color = mode_info.get('color', '#333333')
        
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(f'Global Model Accuracy Statistics - {mode_desc}\nFitness Mode: {fitness_mode.upper()}', 
                    fontsize=14, fontweight='bold', color=mode_color)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        
        # 1. Rankings heatmap across rounds
        ax1 = fig.add_subplot(3, 2, (1, 2))
        
        rankings_matrix = np.zeros((n_methods, n_rounds))
        
        for rnd in range(n_rounds):
            round_acc = []
            for method in methods:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in filtered_results:
                    acc_at_round = []
                    for run_hist in filtered_results[combo_key]['acc_histories']:
                        if rnd < len(run_hist):
                            acc_at_round.append(run_hist[rnd])
                    avg_acc = np.mean(acc_at_round) if acc_at_round else 0
                    round_acc.append(avg_acc)
                else:
                    round_acc.append(0)
            
            # Higher accuracy is better
            ranks = rankdata(-np.array(round_acc))
            rankings_matrix[:, rnd] = ranks
        
        im = ax1.imshow(rankings_matrix, cmap='YlGnBu', aspect='auto')
        ax1.set_yticks(range(n_methods))
        ax1.set_yticklabels(methods)
        ax1.set_xlabel('FL Round')
        ax1.set_ylabel('Optimizer')
        ax1.set_title('Optimizer Rankings Across FL Rounds\n(Based on Global Accuracy - Rank 1 = Best)', fontsize=11)
        
        for i in range(n_methods):
            for j in range(0, n_rounds, max(1, n_rounds//15)):
                ax1.text(j, i, f'{int(rankings_matrix[i, j])}', ha='center', va='center', fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Rank (1=best)')
        
        # 2. Accuracy Convergence Curves
        ax2 = fig.add_subplot(3, 2, 3)
        
        all_acc_values = []
        for idx, method in enumerate(methods):
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                acc_histories = filtered_results[combo_key]['acc_histories']
                if acc_histories:
                    min_len = min(len(h) for h in acc_histories if h)
                    if min_len > 0:
                        aligned = [h[:min_len] for h in acc_histories if h]
                        avg_acc = np.mean(aligned, axis=0)
                        std_acc = np.std(aligned, axis=0)
                        all_acc_values.extend(avg_acc)
                        rounds = np.arange(1, len(avg_acc) + 1)
                        ax2.plot(rounds, avg_acc, label=method, color=colors[idx], linewidth=2)
        
        ax2.set_xlabel('FL Round')
        ax2.set_ylabel('Global Accuracy')
        ax2.set_title('Accuracy Convergence Curves\n(with ±1 Std Dev)', fontsize=11)
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        if all_acc_values:
            y_min, y_max = min(all_acc_values), max(all_acc_values)
            y_margin = (y_max - y_min) * 0.1
            ax2.set_ylim(max(0, y_min - y_margin), min(1, y_max + y_margin))
        
        # 3. Box Plot of Final Accuracy
        ax3 = fig.add_subplot(3, 2, 4)
        
        acc_data = []
        method_labels = []
        for method in methods:
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                final_acc = [m['global']['accuracy'] for m in filtered_results[combo_key]['final_metrics']]
                if final_acc:
                    acc_data.append(final_acc)
                    method_labels.append(method)
        
        if acc_data:
            bp = ax3.boxplot(acc_data, tick_labels=method_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(method_labels)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            all_box_values = [v for sublist in acc_data for v in sublist]
            if all_box_values:
                y_min, y_max = min(all_box_values), max(all_box_values)
                y_margin = (y_max - y_min) * 0.15
                ax3.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax3.set_xlabel('Optimizer')
        ax3.set_ylabel('Global Accuracy')
        ax3.set_title('Final Accuracy Distribution', fontsize=11)
        ax3.tick_params(axis='x', rotation=25)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Bar chart comparison
        ax4 = fig.add_subplot(3, 2, 5)
        
        avg_accs = []
        std_accs = []
        for method in methods:
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                final_acc = [m['global']['accuracy'] for m in filtered_results[combo_key]['final_metrics']]
                avg_accs.append(np.mean(final_acc) if final_acc else 0)
                std_accs.append(np.std(final_acc) if final_acc else 0)
            else:
                avg_accs.append(0)
                std_accs.append(0)
        
        x = np.arange(len(methods))
        bars = ax4.bar(x, avg_accs, yerr=std_accs, color=colors[:len(methods)], alpha=0.8, capsize=3)
        ax4.set_xlabel('Optimizer')
        ax4.set_ylabel('Average Accuracy')
        ax4.set_title('Average Final Accuracy by Optimizer', fontsize=11)
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=25, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        if avg_accs:
            y_min, y_max = min(avg_accs), max(avg_accs)
            y_margin = (y_max - y_min) * 0.15
            ax4.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # 5. Summary Table
        ax5 = fig.add_subplot(3, 2, 6)
        ax5.axis('off')
        
        table_data = []
        for method in methods:
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                final_acc = [m['global']['accuracy'] for m in filtered_results[combo_key]['final_metrics']]
                final_f1 = [m['global']['f1'] for m in filtered_results[combo_key]['final_metrics']]
                if final_acc:
                    table_data.append([
                        method,
                        f'{np.mean(final_acc):.4f}',
                        f'{np.max(final_acc):.4f}',
                        f'{np.std(final_acc):.4f}',
                        f'{np.mean(final_f1):.4f}' if final_f1 else 'N/A'
                    ])
        
        if table_data:
            table = ax5.table(cellText=table_data,
                             colLabels=['Optimizer', 'Avg Acc', 'Best Acc', 'Std Dev', 'Avg F1'],
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        ax5.set_title(f'Accuracy Summary - {fitness_mode.upper()}', fontsize=11, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            save_path = f'{output_dir}/accuracy_statistics_{fitness_mode}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved accuracy statistics plot for {fitness_mode} to {save_path}")
        
        plt.close()
        return fig
    
    def plot_energy_statistics_by_fitness(self, fitness_mode, output_dir, save=True):
        """
        Plot energy/battery consumption statistics as percentages for comparison
        100% baseline = average of all methods, showing +/- difference
        """
        filtered_results = {
            k: v for k, v in self.results.items() 
            if v.get('fitness_mode') == fitness_mode
        }
        
        if not filtered_results:
            print(f"  No results found for fitness mode: {fitness_mode}")
            return None
        
        methods = list(dict.fromkeys([v['selection_method'] for v in filtered_results.values()]))
        n_methods = len(methods)
        
        mode_info = FITNESS_MODES.get(fitness_mode, {})
        mode_desc = mode_info.get('description', fitness_mode)
        mode_color = mode_info.get('color', '#333333')
        
        # Get participation rate for title
        participation_rate = self.experiment_params.get('participation_rate', 'N/A')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Energy & Resource Statistics - {mode_desc}\nFitness: {fitness_mode.upper()} | Participation Rate: {participation_rate}', 
                    fontsize=14, fontweight='bold', color=mode_color)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        
        # Collect data
        energy_data = {method: [] for method in methods}
        battery_data = {method: [] for method in methods}
        
        for method in methods:
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                energy_data[method] = filtered_results[combo_key].get('total_energy', [])
                battery_hists = filtered_results[combo_key].get('avg_battery_histories', [])
                battery_data[method] = [np.mean(h) for h in battery_hists if h]
        
        # 1. Energy consumption bar chart (relative to average)
        ax1 = axes[0, 0]
        
        avg_energy = [np.mean(energy_data[m]) if energy_data[m] else 0 for m in methods]
        overall_avg_energy = np.mean([e for e in avg_energy if e > 0]) if any(e > 0 for e in avg_energy) else 1
        
        # Calculate difference from average (lower energy is better, so negative diff is good)
        energy_diff = [((overall_avg_energy - e) / overall_avg_energy * 100) if e > 0 else 0 for e in avg_energy]
        
        x = np.arange(n_methods)
        bar_colors = ['green' if d >= 0 else 'red' for d in energy_diff]
        bars = ax1.bar(x, energy_diff, color=bar_colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Average (0%)')
        ax1.set_xlabel('Optimizer')
        ax1.set_ylabel('Difference from Average (%)')
        ax1.set_title('Energy Efficiency\n(Positive = More Efficient than Average)', fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=25, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to emphasize differences
        if energy_diff:
            y_min, y_max = min(energy_diff), max(energy_diff)
            y_margin = max(abs(y_min), abs(y_max)) * 0.2
            ax1.set_ylim(min(y_min - y_margin, -5), max(y_max + y_margin, 5))
        
        # Add value labels
        for bar, val, method in zip(bars, energy_diff, methods):
            label = f'{method}\n{val:+.1f}%'
            y_pos = bar.get_height() + (0.5 if val >= 0 else -1.5)
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
        
        # 2. Battery level bar chart (relative to average)
        ax2 = axes[0, 1]
        
        avg_battery = [np.mean(battery_data[m]) if battery_data[m] else 0 for m in methods]
        overall_avg_battery = np.mean([b for b in avg_battery if b > 0]) if any(b > 0 for b in avg_battery) else 1
        
        # Calculate difference from average (higher battery is better, so positive diff is good)
        battery_diff = [((b - overall_avg_battery) / overall_avg_battery * 100) if b > 0 else 0 for b in avg_battery]
        
        bar_colors = ['green' if d >= 0 else 'red' for d in battery_diff]
        bars = ax2.bar(x, battery_diff, color=bar_colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Average (0%)')
        ax2.set_xlabel('Optimizer')
        ax2.set_ylabel('Difference from Average (%)')
        ax2.set_title('Battery Level of Selected Clients\n(Positive = Higher than Average)', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=25, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        if battery_diff:
            y_min, y_max = min(battery_diff), max(battery_diff)
            y_margin = max(abs(y_min), abs(y_max)) * 0.2
            ax2.set_ylim(min(y_min - y_margin, -5), max(y_max + y_margin, 5))
        
        for bar, val in zip(bars, battery_diff):
            y_pos = bar.get_height() + (0.5 if val >= 0 else -1.5)
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%', 
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=9, fontweight='bold')
        
        # 3. Energy consumption over rounds
        ax3 = axes[1, 0]
        
        for idx, method in enumerate(methods):
            combo_key = f"{method}_{fitness_mode}"
            if combo_key in filtered_results:
                energy_hists = filtered_results[combo_key].get('energy_histories', [])
                if energy_hists:
                    min_len = min(len(h) for h in energy_hists if h)
                    if min_len > 0:
                        aligned = [h[:min_len] for h in energy_hists if h]
                        cum_energy = np.cumsum(np.mean(aligned, axis=0))
                        rounds = np.arange(1, len(cum_energy) + 1)
                        ax3.plot(rounds, cum_energy, label=method, color=colors[idx], linewidth=1.5)
        
        ax3.set_xlabel('FL Round')
        ax3.set_ylabel('Cumulative Energy')
        ax3.set_title('Cumulative Energy Consumption Over Rounds', fontsize=11)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary table with relative percentages
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        for i, method in enumerate(methods):
            table_data.append([
                method,
                f'{avg_energy[i]:.2f}',
                f'{energy_diff[i]:+.1f}%',
                f'{avg_battery[i]*100:.1f}%' if avg_battery[i] else 'N/A',
                f'{battery_diff[i]:+.1f}%'
            ])
        
        if table_data:
            table = ax4.table(cellText=table_data,
                             colLabels=['Optimizer', 'Avg Energy', 'Energy Diff', 'Avg Battery', 'Battery Diff'],
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Color code the difference columns
            for i in range(len(table_data)):
                # Energy diff column (index 2)
                cell = table[(i+1, 2)]
                if energy_diff[i] >= 0:
                    cell.set_facecolor('#90EE90')  # Light green
                else:
                    cell.set_facecolor('#FFB6C1')  # Light red
                # Battery diff column (index 4)
                cell = table[(i+1, 4)]
                if battery_diff[i] >= 0:
                    cell.set_facecolor('#90EE90')
                else:
                    cell.set_facecolor('#FFB6C1')
        
        ax4.set_title(f'Energy & Battery Summary\n(Participation Rate: {participation_rate})', fontsize=11, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            save_path = f'{output_dir}/energy_statistics_{fitness_mode}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved energy statistics plot for {fitness_mode} to {save_path}")
        
        plt.close()
        return fig
    
    def plot_average_rankings(self, output_dir, save=True):
        """
        Plot average rankings of optimizers across all rounds for both fitness and accuracy
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        def to_positive_fitness(val):
            return -val if val < 0 else val
        
        fig, axes = plt.subplots(len(fitness_modes), 2, figsize=(14, 5*len(fitness_modes)))
        if len(fitness_modes) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Average Rankings Across FL Rounds\n(Lower Rank = Better)', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selection_methods)))
        
        for mode_idx, fitness_mode in enumerate(fitness_modes):
            filtered_results = {
                k: v for k, v in self.results.items() 
                if v.get('fitness_mode') == fitness_mode
            }
            
            if not filtered_results:
                continue
            
            first_key = list(filtered_results.keys())[0]
            n_rounds = len(filtered_results[first_key]['acc_histories'][0]) if filtered_results[first_key]['acc_histories'] else 0
            
            # Calculate average rank for fitness
            fitness_ranks = {m: [] for m in selection_methods}
            accuracy_ranks = {m: [] for m in selection_methods}
            
            for rnd in range(n_rounds):
                # Fitness ranking
                round_fitness = []
                for method in selection_methods:
                    combo_key = f"{method}_{fitness_mode}"
                    if combo_key in filtered_results:
                        fitness_at_round = []
                        for run_hist in filtered_results[combo_key].get('fitness_histories', []):
                            if rnd < len(run_hist):
                                fitness_at_round.append(to_positive_fitness(run_hist[rnd]))
                        round_fitness.append(np.mean(fitness_at_round) if fitness_at_round else 0)
                    else:
                        round_fitness.append(0)
                
                ranks = rankdata(-np.array(round_fitness))  # Higher fitness = better = rank 1
                for i, method in enumerate(selection_methods):
                    fitness_ranks[method].append(ranks[i])
                
                # Accuracy ranking
                round_acc = []
                for method in selection_methods:
                    combo_key = f"{method}_{fitness_mode}"
                    if combo_key in filtered_results:
                        acc_at_round = []
                        for run_hist in filtered_results[combo_key].get('acc_histories', []):
                            if rnd < len(run_hist):
                                acc_at_round.append(run_hist[rnd])
                        round_acc.append(np.mean(acc_at_round) if acc_at_round else 0)
                    else:
                        round_acc.append(0)
                
                ranks = rankdata(-np.array(round_acc))  # Higher accuracy = better = rank 1
                for i, method in enumerate(selection_methods):
                    accuracy_ranks[method].append(ranks[i])
            
            # Plot Fitness Rankings
            ax1 = axes[mode_idx, 0]
            avg_fitness_ranks = [np.mean(fitness_ranks[m]) for m in selection_methods]
            std_fitness_ranks = [np.std(fitness_ranks[m]) for m in selection_methods]
            
            x = np.arange(len(selection_methods))
            bars = ax1.bar(x, avg_fitness_ranks, yerr=std_fitness_ranks, color=colors, alpha=0.8, capsize=3)
            ax1.set_xlabel('Optimizer')
            ax1.set_ylabel('Average Rank')
            ax1.set_title(f'Fitness Rankings - {fitness_mode.upper()}\n(Lower = Better)', fontsize=11)
            ax1.set_xticks(x)
            ax1.set_xticklabels(selection_methods, rotation=25, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_ylim(0, len(selection_methods) + 0.5)
            
            for bar, val in zip(bars, avg_fitness_ranks):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Plot Accuracy Rankings
            ax2 = axes[mode_idx, 1]
            avg_acc_ranks = [np.mean(accuracy_ranks[m]) for m in selection_methods]
            std_acc_ranks = [np.std(accuracy_ranks[m]) for m in selection_methods]
            
            bars = ax2.bar(x, avg_acc_ranks, yerr=std_acc_ranks, color=colors, alpha=0.8, capsize=3)
            ax2.set_xlabel('Optimizer')
            ax2.set_ylabel('Average Rank')
            ax2.set_title(f'Accuracy Rankings - {fitness_mode.upper()}\n(Lower = Better)', fontsize=11)
            ax2.set_xticks(x)
            ax2.set_xticklabels(selection_methods, rotation=25, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, len(selection_methods) + 0.5)
            
            for bar, val in zip(bars, avg_acc_ranks):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            save_path = f'{output_dir}/average_rankings.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved average rankings plot to {save_path}")
        
        plt.close()
        return fig
    
    def perform_friedman_test(self, output_dir):
        """
        Perform Friedman test for statistical significance comparison of optimizers
        Tests both fitness values and accuracy across runs
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        def to_positive_fitness(val):
            return -val if val < 0 else val
        
        friedman_results = []
        
        for fitness_mode in fitness_modes:
            filtered_results = {
                k: v for k, v in self.results.items() 
                if v.get('fitness_mode') == fitness_mode
            }
            
            if not filtered_results:
                continue
            
            # Collect final fitness values per method
            fitness_per_method = []
            accuracy_per_method = []
            
            for method in selection_methods:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in filtered_results:
                    # Final fitness values (positive)
                    final_fitness = [to_positive_fitness(h[-1]) for h in filtered_results[combo_key].get('fitness_histories', []) if h]
                    fitness_per_method.append(final_fitness if final_fitness else [0])
                    
                    # Final accuracy values
                    final_acc = [m['global']['accuracy'] for m in filtered_results[combo_key].get('final_metrics', [])]
                    accuracy_per_method.append(final_acc if final_acc else [0])
                else:
                    fitness_per_method.append([0])
                    accuracy_per_method.append([0])
            
            # Align lengths for Friedman test (need same number of observations)
            min_len_fitness = min(len(f) for f in fitness_per_method)
            min_len_acc = min(len(a) for a in accuracy_per_method)
            
            if min_len_fitness >= 3 and len(selection_methods) >= 3:
                aligned_fitness = [f[:min_len_fitness] for f in fitness_per_method]
                try:
                    stat_f, p_value_f = friedmanchisquare(*aligned_fitness)
                    friedman_results.append({
                        'Fitness_Mode': fitness_mode,
                        'Metric': 'Fitness',
                        'Friedman_Statistic': stat_f,
                        'P_Value': p_value_f,
                        'Significant': 'Yes' if p_value_f < 0.05 else 'No',
                        'N_Samples': min_len_fitness
                    })
                except Exception as e:
                    print(f"  Warning: Friedman test for fitness failed: {e}")
            
            if min_len_acc >= 3 and len(selection_methods) >= 3:
                aligned_acc = [a[:min_len_acc] for a in accuracy_per_method]
                try:
                    stat_a, p_value_a = friedmanchisquare(*aligned_acc)
                    friedman_results.append({
                        'Fitness_Mode': fitness_mode,
                        'Metric': 'Accuracy',
                        'Friedman_Statistic': stat_a,
                        'P_Value': p_value_a,
                        'Significant': 'Yes' if p_value_a < 0.05 else 'No',
                        'N_Samples': min_len_acc
                    })
                except Exception as e:
                    print(f"  Warning: Friedman test for accuracy failed: {e}")
        
        # Save results
        if friedman_results:
            df = pd.DataFrame(friedman_results)
            df.to_csv(f'{output_dir}/friedman_test_results.csv', index=False)
            print(f"✓ Saved Friedman test results to {output_dir}/friedman_test_results.csv")
            
            # Print summary
            print("\n" + "=" * 60)
            print("FRIEDMAN TEST RESULTS")
            print("=" * 60)
            for _, row in df.iterrows():
                print(f"  {row['Fitness_Mode'].upper()} - {row['Metric']}: "
                      f"χ²={row['Friedman_Statistic']:.4f}, p={row['P_Value']:.4f} "
                      f"({'Significant' if row['P_Value'] < 0.05 else 'Not Significant'})")
        
        return friedman_results
    
    def save_comprehensive_statistics_csv(self, output_dir):
        """
        Save comprehensive statistics tables for paper
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        def to_positive_fitness(val):
            return -val if val < 0 else val
        
        # Table 1: Complete performance summary
        complete_summary = []
        for method in selection_methods:
            for fitness_mode in fitness_modes:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in self.results:
                    data = self.results[combo_key]
                    
                    # Fitness metrics
                    final_fitness = [to_positive_fitness(h[-1]) for h in data.get('fitness_histories', []) if h]
                    
                    # Accuracy metrics
                    final_acc = [m['global']['accuracy'] for m in data.get('final_metrics', [])]
                    final_f1 = [m['global']['f1'] for m in data.get('final_metrics', [])]
                    final_precision = [m['global'].get('precision', 0) for m in data.get('final_metrics', [])]
                    final_recall = [m['global'].get('recall', 0) for m in data.get('final_metrics', [])]
                    
                    # Energy metrics
                    total_energy = data.get('total_energy', [])
                    
                    complete_summary.append({
                        'Method': method,
                        'Fitness_Mode': fitness_mode,
                        'Fitness_Avg': np.mean(final_fitness) if final_fitness else 0,
                        'Fitness_Std': np.std(final_fitness) if final_fitness else 0,
                        'Accuracy_Avg': np.mean(final_acc) if final_acc else 0,
                        'Accuracy_Std': np.std(final_acc) if final_acc else 0,
                        'F1_Avg': np.mean(final_f1) if final_f1 else 0,
                        'Precision_Avg': np.mean(final_precision) if final_precision else 0,
                        'Recall_Avg': np.mean(final_recall) if final_recall else 0,
                        'Energy_Avg': np.mean(total_energy) if total_energy else 0,
                        'Energy_Std': np.std(total_energy) if total_energy else 0,
                    })
        
        df_summary = pd.DataFrame(complete_summary)
        df_summary.to_csv(f'{output_dir}/complete_performance_summary.csv', index=False)
        print(f"✓ Saved complete performance summary to {output_dir}/complete_performance_summary.csv")
        
        # Table 2: Rankings summary
        rankings_summary = []
        for fitness_mode in fitness_modes:
            filtered = {k: v for k, v in self.results.items() if v.get('fitness_mode') == fitness_mode}
            
            # Rank by fitness
            fitness_avgs = {}
            acc_avgs = {}
            for method in selection_methods:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in filtered:
                    final_fitness = [to_positive_fitness(h[-1]) for h in filtered[combo_key].get('fitness_histories', []) if h]
                    final_acc = [m['global']['accuracy'] for m in filtered[combo_key].get('final_metrics', [])]
                    fitness_avgs[method] = np.mean(final_fitness) if final_fitness else 0
                    acc_avgs[method] = np.mean(final_acc) if final_acc else 0
            
            # Sort by fitness (higher = better)
            sorted_by_fitness = sorted(fitness_avgs.keys(), key=lambda m: fitness_avgs[m], reverse=True)
            sorted_by_acc = sorted(acc_avgs.keys(), key=lambda m: acc_avgs[m], reverse=True)
            
            for rank, method in enumerate(sorted_by_fitness, 1):
                acc_rank = sorted_by_acc.index(method) + 1 if method in sorted_by_acc else 0
                rankings_summary.append({
                    'Fitness_Mode': fitness_mode,
                    'Method': method,
                    'Fitness_Rank': rank,
                    'Fitness_Value': fitness_avgs.get(method, 0),
                    'Accuracy_Rank': acc_rank,
                    'Accuracy_Value': acc_avgs.get(method, 0),
                })
        
        df_rankings = pd.DataFrame(rankings_summary)
        df_rankings.to_csv(f'{output_dir}/rankings_summary.csv', index=False)
        print(f"✓ Saved rankings summary to {output_dir}/rankings_summary.csv")
        
        # Table 3: Energy efficiency comparison (relative %)
        energy_comparison = []
        for fitness_mode in fitness_modes:
            energy_by_method = {}
            for method in selection_methods:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in self.results:
                    total_energy = self.results[combo_key].get('total_energy', [])
                    energy_by_method[method] = np.mean(total_energy) if total_energy else 0
            
            if energy_by_method:
                min_energy = min([e for e in energy_by_method.values() if e > 0]) if any(e > 0 for e in energy_by_method.values()) else 1
                
                for method, energy in energy_by_method.items():
                    energy_comparison.append({
                        'Fitness_Mode': fitness_mode,
                        'Method': method,
                        'Energy_Absolute': energy,
                        'Energy_Relative_%': (min_energy / energy * 100) if energy > 0 else 0
                    })
        
        df_energy = pd.DataFrame(energy_comparison)
        df_energy.to_csv(f'{output_dir}/energy_comparison.csv', index=False)
        print(f"✓ Saved energy comparison to {output_dir}/energy_comparison.csv")
        
        # Table 4: Convergence statistics
        convergence_stats = []
        for method in selection_methods:
            for fitness_mode in fitness_modes:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in self.results:
                    data = self.results[combo_key]
                    conv_rounds = [r for r in data.get('convergence_rounds', []) if r is not None]
                    
                    convergence_stats.append({
                        'Method': method,
                        'Fitness_Mode': fitness_mode,
                        'Avg_Convergence_Round': np.mean(conv_rounds) if conv_rounds else 'N/A',
                        'Std_Convergence_Round': np.std(conv_rounds) if conv_rounds else 'N/A',
                        'Converged_Runs': len(conv_rounds),
                        'Total_Runs': len(data.get('acc_histories', []))
                    })
        
        df_convergence = pd.DataFrame(convergence_stats)
        df_convergence.to_csv(f'{output_dir}/convergence_statistics.csv', index=False)
        print(f"✓ Saved convergence statistics to {output_dir}/convergence_statistics.csv")
        
        return df_summary, df_rankings, df_energy, df_convergence
    
    def plot_accuracy_comparison_across_modes(self, output_dir, save=True):
        """
        Plot accuracy comparison across fitness modes (similar to fitness comparison)
        Bars grouped by fitness mode
        """
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        if len(fitness_modes) < 2:
            print("  Need at least 2 fitness modes for comparison")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Accuracy Comparison Across Fitness Modes', fontsize=14, fontweight='bold')
        
        # Prepare data
        comparison_data = {mode: {} for mode in fitness_modes}
        
        for combo_key, data in self.results.items():
            method = data.get('selection_method')
            mode = data.get('fitness_mode')
            if method and mode:
                final_acc = [m['global']['accuracy'] for m in data.get('final_metrics', [])]
                if final_acc:
                    comparison_data[mode][method] = {
                        'avg': np.mean(final_acc),
                        'std': np.std(final_acc),
                        'min': np.min(final_acc),
                        'max': np.max(final_acc)
                    }
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selection_methods)))
        
        # Plot 1: Grouped bar chart by fitness mode
        ax1 = axes[0, 0]
        x = np.arange(len(fitness_modes))
        width = 0.8 / len(selection_methods)
        
        all_avgs = []
        for i, method in enumerate(selection_methods):
            avgs = [comparison_data[mode].get(method, {}).get('avg', 0) for mode in fitness_modes]
            stds = [comparison_data[mode].get(method, {}).get('std', 0) for mode in fitness_modes]
            all_avgs.extend(avgs)
            offset = (i - len(selection_methods)/2 + 0.5) * width
            ax1.bar(x + offset, avgs, width, yerr=stds, label=method, 
                   color=colors[i], alpha=0.8, capsize=2)
        
        ax1.set_xlabel('Fitness Mode')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Accuracy\n(Grouped by Fitness Mode)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in fitness_modes])
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        if all_avgs:
            y_min, y_max = min(all_avgs), max(all_avgs)
            y_margin = (y_max - y_min) * 0.15
            ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Plot 2: Heatmap
        ax2 = axes[0, 1]
        
        acc_matrix = np.zeros((len(selection_methods), len(fitness_modes)))
        for i, method in enumerate(selection_methods):
            for j, mode in enumerate(fitness_modes):
                acc_matrix[i, j] = comparison_data[mode].get(method, {}).get('avg', 0)
        
        im = ax2.imshow(acc_matrix, cmap='YlGn', aspect='auto')
        ax2.set_xticks(range(len(fitness_modes)))
        ax2.set_xticklabels([m.upper() for m in fitness_modes])
        ax2.set_yticks(range(len(selection_methods)))
        ax2.set_yticklabels(selection_methods)
        ax2.set_xlabel('Fitness Mode')
        ax2.set_ylabel('Selection Method')
        ax2.set_title('Accuracy Heatmap')
        
        for i in range(len(selection_methods)):
            for j in range(len(fitness_modes)):
                ax2.text(j, i, f'{acc_matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)
        
        plt.colorbar(im, ax=ax2, label='Accuracy')
        
        # Plot 3: Best method per mode
        ax3 = axes[1, 0]
        
        best_methods, best_accs, mode_colors = [], [], []
        for mode in fitness_modes:
            mode_data = comparison_data[mode]
            if mode_data:
                best_method = max(mode_data.keys(), key=lambda m: mode_data[m]['avg'])
                best_methods.append(f"{best_method}\n({mode.upper()})")
                best_accs.append(mode_data[best_method]['avg'])
                mode_colors.append(FITNESS_MODES.get(mode, {}).get('color', '#333333'))
        
        bars = ax3.bar(range(len(best_methods)), best_accs, color=mode_colors, alpha=0.8)
        ax3.set_xticks(range(len(best_methods)))
        ax3.set_xticklabels(best_methods)
        ax3.set_ylabel('Best Accuracy')
        ax3.set_title('Best Method per Fitness Mode')
        ax3.grid(True, alpha=0.3, axis='y')
        
        if best_accs:
            y_min, y_max = min(best_accs), max(best_accs)
            y_margin = (y_max - y_min) * 0.15
            ax3.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        for method in selection_methods:
            row = [method]
            for mode in fitness_modes:
                data = comparison_data[mode].get(method, {})
                if data:
                    row.append(f"{data['avg']:.4f}±{data['std']:.4f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        best_row = ['BEST']
        for mode in fitness_modes:
            mode_data = comparison_data[mode]
            if mode_data:
                best_method = max(mode_data.keys(), key=lambda m: mode_data[m]['avg'])
                best_row.append(f"{best_method}")
            else:
                best_row.append("N/A")
        table_data.append(best_row)
        
        col_labels = ['Method'] + [f'{m.upper()}' for m in fitness_modes]
        table = ax4.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for j in range(len(col_labels)):
            table[(len(table_data), j)].set_facecolor('#90EE90')
        
        ax4.set_title('Accuracy Comparison Summary\n(Avg±Std)', fontsize=11, pad=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            save_path = f'{output_dir}/accuracy_comparison_across_modes.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved accuracy comparison to {save_path}")
        
        plt.close()
        return fig
    
    def generate_report(self, base_output_dir='Results'):
        """Generate comprehensive report with separate plots per fitness mode"""
        timestamp = self.experiment_params.get('timestamp', datetime.now().strftime("%Y%m%d_%H-%M-%S"))
        output_dir = os.path.join(base_output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 80)
        print(f"GENERATING COMPREHENSIVE REPORT")
        print(f"Output Directory: {output_dir}")
        print("=" * 80)
        
        stats = self.compute_statistics()
        summary_df = self.print_summary(stats)
        
        # Save experiment parameters
        self._save_experiment_parameters(output_dir)
        
        # Save summary to CSV
        summary_df.to_csv(f'{output_dir}/summary_by_method_fitness.csv', index=False)
        
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        
        # ============ GENERATE PLOTS PER FITNESS MODE ============
        print("\n" + "=" * 60)
        print("GENERATING STATISTICS PLOTS (PER FITNESS MODE)")
        print("=" * 60)
        
        for fitness_mode in fitness_modes:
            print(f"\n--- Fitness Mode: {fitness_mode.upper()} ---")
            self.plot_global_statistics_by_fitness(fitness_mode, output_dir, save=True)
            self.plot_accuracy_statistics_by_fitness(fitness_mode, output_dir, save=True)
            self.plot_energy_statistics_by_fitness(fitness_mode, output_dir, save=True)
            self.plot_metaheuristic_convergence_curves(fitness_mode, output_dir, save=True)
            self.plot_roc_curves_by_fitness(fitness_mode, output_dir, save=True)
        
        # ============ GENERATE COMPARISON ACROSS FITNESS MODES ============
        if len(fitness_modes) > 1:
            print("\n" + "=" * 60)
            print("GENERATING CROSS-MODE COMPARISONS")
            print("=" * 60)
            
            self.plot_fitness_comparison_across_modes(output_dir, save=True)
            self.plot_accuracy_comparison_across_modes(output_dir, save=True)
            self.save_fitness_comparison_tables(output_dir)
            self.plot_roc_comparison_all_modes(output_dir, save=True)
        
        # ============ GENERATE RANKINGS & STATISTICAL TESTS ============
        print("\n" + "=" * 60)
        print("GENERATING RANKINGS & STATISTICAL TESTS")
        print("=" * 60)
        
        self.plot_average_rankings(output_dir, save=True)
        self.perform_friedman_test(output_dir)
        
        # ============ SAVE AUC SUMMARY ============
        print("\n" + "=" * 60)
        print("SAVING AUC SUMMARY")
        print("=" * 60)
        self.save_auc_summary(output_dir)
        
        # ============ SAVE COMPREHENSIVE STATISTICS ============
        print("\n" + "=" * 60)
        print("SAVING COMPREHENSIVE STATISTICS TABLES")
        print("=" * 60)
        self.save_comprehensive_statistics_csv(output_dir)
        
        # ============ SAVE DETAILED STATISTICS ============
        print("\n" + "=" * 60)
        print("SAVING DETAILED STATISTICS")
        print("=" * 60)
        
        detailed_stats = []
        for combo_key, s in stats.items():
            for metric_name, metric_key in [
                ('Global Accuracy', 'global_accuracy'),
                ('Combined Accuracy', 'combined_accuracy'),
                ('Personalized Accuracy', 'personalized_accuracy'),
                ('Global F1', 'global_f1'),
                ('Total Energy', 'total_energy'),
                ('Accuracy/Energy', 'accuracy_per_energy'),
            ]:
                if metric_key in s:
                    detailed_stats.append({
                        'Method_Fitness': combo_key,
                        'Selection_Method': s.get('selection_method', ''),
                        'Fitness_Mode': s.get('fitness_mode', ''),
                        'Metric': metric_name,
                        'Min': s[metric_key]['min'],
                        'Max': s[metric_key]['max'],
                        'Avg': s[metric_key]['avg'],
                        'Std': s[metric_key]['std']
                    })
        
        pd.DataFrame(detailed_stats).to_csv(f'{output_dir}/detailed_statistics.csv', index=False)
        print(f"✓ Saved detailed statistics to {output_dir}/detailed_statistics.csv")
        
        # Save per-round fitness data for each combination
        self._save_per_round_fitness_data(output_dir)
        
        # Save device metrics as relative percentages
        self._save_device_metrics_relative(output_dir)
        
        print("\n" + "=" * 80)
        print(f"REPORT GENERATION COMPLETE")
        print(f"All files saved to: {output_dir}")
        print("=" * 80)
        
        return stats, summary_df, output_dir
    
    def _save_device_metrics_relative(self, output_dir):
        """Save device metrics (battery, energy, etc.) as relative percentages"""
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        selection_methods = self.experiment_params.get('selection_methods', [])
        
        relative_data = []
        
        for method in selection_methods:
            metrics_per_mode = {}
            for fitness_mode in fitness_modes:
                combo_key = f"{method}_{fitness_mode}"
                if combo_key in self.results:
                    data = self.results[combo_key]
                    battery_avgs = [np.mean(h) for h in data.get('avg_battery_histories', []) if h]
                    energy_avgs = [np.mean(h) for h in data.get('energy_histories', []) if h]
                    reliability_avgs = [np.mean(h) for h in data.get('avg_reliability_histories', []) if h]
                    
                    metrics_per_mode[fitness_mode] = {
                        'battery': np.mean(battery_avgs) if battery_avgs else 0,
                        'energy': np.mean(energy_avgs) if energy_avgs else 0,
                        'reliability': np.mean(reliability_avgs) if reliability_avgs else 0,
                    }
            
            if not metrics_per_mode:
                continue
            
            best_battery = max(m['battery'] for m in metrics_per_mode.values()) if metrics_per_mode else 1
            min_energy = min(m['energy'] for m in metrics_per_mode.values()) if metrics_per_mode else 1
            best_reliability = max(m['reliability'] for m in metrics_per_mode.values()) if metrics_per_mode else 1
            
            for fitness_mode, metrics in metrics_per_mode.items():
                relative_data.append({
                    'Method': method,
                    'Fitness_Mode': fitness_mode,
                    'Battery_Abs': metrics['battery'],
                    'Battery_Rel%': (metrics['battery'] / best_battery * 100) if best_battery > 0 else 0,
                    'Energy_Abs': metrics['energy'],
                    'Energy_Rel%': (min_energy / metrics['energy'] * 100) if metrics['energy'] > 0 else 0,
                    'Reliability_Abs': metrics['reliability'],
                    'Reliability_Rel%': (metrics['reliability'] / best_reliability * 100) if best_reliability > 0 else 0,
                })
        
        if relative_data:
            df = pd.DataFrame(relative_data)
            df.to_csv(f'{output_dir}/device_metrics_relative.csv', index=False)
            print(f"✓ Saved relative device metrics to {output_dir}/device_metrics_relative.csv")
    
    def _save_per_round_fitness_data(self, output_dir):
        """Save per-round fitness data for all combinations"""
        round_data = []
        for combo_key, data in self.results.items():
            method = data.get('selection_method', '')
            mode = data.get('fitness_mode', '')
            
            for run_idx, fitness_hist in enumerate(data.get('fitness_histories', [])):
                if fitness_hist:
                    for rnd, fitness in enumerate(fitness_hist):
                        round_data.append({
                            'Selection_Method': method,
                            'Fitness_Mode': mode,
                            'Combo_Key': combo_key,
                            'Run': run_idx + 1,
                            'Round': rnd + 1,
                            'Selection_Fitness': fitness
                        })
        
        if round_data:
            df = pd.DataFrame(round_data)
            df.to_csv(f'{output_dir}/per_round_fitness_data.csv', index=False)
            print(f"✓ Saved per-round fitness data to {output_dir}/per_round_fitness_data.csv")
    
    def _save_experiment_parameters(self, output_dir):
        """Save experiment parameters to a text file"""
        params_file = os.path.join(output_dir, 'experiment_parameters.txt')
        
        with open(params_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXPERIMENT PARAMETERS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Timestamp: {self.experiment_params.get('timestamp', 'N/A')}\n")
            f.write(f"Selection Methods: {self.experiment_params.get('selection_methods', [])}\n")
            f.write(f"Fitness Modes: {self.experiment_params.get('fitness_modes', [])}\n")
            f.write(f"Number of Runs: {self.experiment_params.get('n_runs', 'N/A')}\n")
            f.write(f"Number of Rounds: {self.experiment_params.get('n_rounds', 'N/A')}\n")
            f.write(f"Participation Rate: {self.experiment_params.get('participation_rate', 'N/A')}\n")
            f.write(f"Local Epochs: {self.experiment_params.get('local_epochs', 'N/A')}\n")
            f.write(f"Learning Rate: {self.experiment_params.get('lr', 'N/A')}\n")
            f.write(f"FedProx Mu: {self.experiment_params.get('mu', 'N/A')}\n")
            f.write(f"Edge Simulation: {self.experiment_params.get('use_edge_sim', 'N/A')}\n")
            f.write(f"Edge Sim File: {self.experiment_params.get('edge_sim_file', 'None')}\n")
            f.write("\n")
            
            f.write("FITNESS MODE DEFINITIONS:\n")
            for mode, config in FITNESS_MODES.items():
                f.write(f"\n  {mode.upper()}: {config['description']}\n")
                f.write(f"    Weights: {config['weights']}\n")
        
        print(f"✓ Saved experiment parameters to {params_file}")
        
        json_file = os.path.join(output_dir, 'experiment_parameters.json')
        with open(json_file, 'w') as f:
            json.dump(self.experiment_params, f, indent=2, default=str)
        print(f"✓ Saved experiment parameters to {json_file}")
    
    def generate_report_to_dir(self, output_dir):
        """Generate comprehensive report to a specific directory (for rate subfolders)"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n  Generating report to: {output_dir}")
        
        stats = self.compute_statistics()
        summary_df = self.print_summary(stats)
        
        # Save experiment parameters
        self._save_experiment_parameters(output_dir)
        
        # Save summary
        summary_df.to_csv(f'{output_dir}/summary_by_method_fitness.csv', index=False)
        
        fitness_modes = self.experiment_params.get('fitness_modes', ['learning'])
        
        # Generate plots per fitness mode
        for fitness_mode in fitness_modes:
            self.plot_global_statistics_by_fitness(fitness_mode, output_dir, save=True)
            self.plot_accuracy_statistics_by_fitness(fitness_mode, output_dir, save=True)
            self.plot_energy_statistics_by_fitness(fitness_mode, output_dir, save=True)
            self.plot_metaheuristic_convergence_curves(fitness_mode, output_dir, save=True)
            self.plot_roc_curves_by_fitness(fitness_mode, output_dir, save=True)
        
        # Cross-mode comparisons
        if len(fitness_modes) > 1:
            self.plot_fitness_comparison_across_modes(output_dir, save=True)
            self.plot_accuracy_comparison_across_modes(output_dir, save=True)
            self.save_fitness_comparison_tables(output_dir)
            self.plot_roc_comparison_all_modes(output_dir, save=True)
        
        # Rankings and statistical tests
        self.plot_average_rankings(output_dir, save=True)
        self.perform_friedman_test(output_dir)
        
        # Save summaries
        self.save_auc_summary(output_dir)
        self.save_comprehensive_statistics_csv(output_dir)
        
        # Detailed statistics
        detailed_stats = []
        for combo_key, s in stats.items():
            for metric_name, metric_key in [
                ('Global Accuracy', 'global_accuracy'),
                ('Combined Accuracy', 'combined_accuracy'),
                ('Personalized Accuracy', 'personalized_accuracy'),
                ('Global F1', 'global_f1'),
                ('Total Energy', 'total_energy'),
                ('Accuracy/Energy', 'accuracy_per_energy'),
            ]:
                if metric_key in s:
                    detailed_stats.append({
                        'Method_Fitness': combo_key,
                        'Selection_Method': s.get('selection_method', ''),
                        'Fitness_Mode': s.get('fitness_mode', ''),
                        'Metric': metric_name,
                        'Min': s[metric_key]['min'],
                        'Max': s[metric_key]['max'],
                        'Avg': s[metric_key]['avg'],
                        'Std': s[metric_key]['std']
                    })
        
        pd.DataFrame(detailed_stats).to_csv(f'{output_dir}/detailed_statistics.csv', index=False)
        
        # Save additional data
        self._save_per_round_fitness_data(output_dir)
        self._save_device_metrics_relative(output_dir)
        
        return stats, summary_df



# ==============================================================================
# STANDALONE PLOTTING FUNCTIONS (COMPLETE VERSION FROM ORIGINAL)
# ==============================================================================

def _plot_participation_rate_comparison(df, selection_methods, fitness_modes, 
                                        participation_rates, output_dir):
    """Generate comprehensive comparison plots across participation rates"""
    if len(participation_rates)==1:
        return()
    colors = plt.cm.tab10(np.linspace(0, 1, len(selection_methods)))
    rate_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(participation_rates)))
    
    def get_y_limits_with_margin(values, stds=None, margin_pct=0.15):
        """Calculate y-axis limits with margin to show error bars clearly"""
        if stds is not None:
            min_val = min(v - s for v, s in zip(values, stds))
            max_val = max(v + s for v, s in zip(values, stds))
        else:
            min_val, max_val = min(values), max(values)
        
        data_range = max_val - min_val
        if data_range == 0:
            data_range = abs(max_val) * 0.1 if max_val != 0 else 0.1
        
        margin = data_range * margin_pct
        return min_val - margin, max_val + margin
    
    # 1. Fitness comparison across rates (grouped by fitness mode)
    fig, axes = plt.subplots(1, len(fitness_modes), figsize=(6*len(fitness_modes), 6))
    if len(fitness_modes) == 1:
        axes = [axes]
    
    fig.suptitle('Fitness Performance Across Participation Rates\n(Higher = Better)', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, fitness_mode in enumerate(fitness_modes):
        ax = axes[ax_idx]
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        
        x = np.arange(len(participation_rates))
        width = 0.8 / len(selection_methods)
        
        all_vals, all_stds = [], []
        for i, method in enumerate(selection_methods):
            method_df = mode_df[mode_df['Selection_Method'] == method]
            avgs = [method_df[method_df['Participation_Rate'] == r]['Fitness_Avg'].values[0] 
                   if len(method_df[method_df['Participation_Rate'] == r]) > 0 else 0 
                   for r in participation_rates]
            stds = [method_df[method_df['Participation_Rate'] == r]['Fitness_Std'].values[0] 
                   if len(method_df[method_df['Participation_Rate'] == r]) > 0 else 0 
                   for r in participation_rates]
            all_vals.extend(avgs)
            all_stds.extend(stds)
            
            offset = (i - len(selection_methods)/2 + 0.5) * width
            ax.bar(x + offset, avgs, width, yerr=stds, label=method, color=colors[i], alpha=0.8, capsize=2)
        
        ax.set_xlabel('Participation Rate')
        ax.set_ylabel('Fitness (Higher = Better)')
        ax.set_title(f'{fitness_mode.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis with proper margins for error bars
        if all_vals:
            y_min, y_max = get_y_limits_with_margin(all_vals, all_stds)
            ax.set_ylim(y_min, y_max)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/fitness_by_participation_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fitness comparison to {output_dir}/fitness_by_participation_rate.png")
    
    # 2. Accuracy comparison across rates (with proper y-axis scaling)
    fig, axes = plt.subplots(1, len(fitness_modes), figsize=(6*len(fitness_modes), 6))
    if len(fitness_modes) == 1:
        axes = [axes]
    
    fig.suptitle('Accuracy Performance Across Participation Rates', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, fitness_mode in enumerate(fitness_modes):
        ax = axes[ax_idx]
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        
        x = np.arange(len(participation_rates))
        width = 0.8 / len(selection_methods)
        
        all_vals, all_stds = [], []
        for i, method in enumerate(selection_methods):
            method_df = mode_df[mode_df['Selection_Method'] == method]
            avgs = [method_df[method_df['Participation_Rate'] == r]['Accuracy_Avg'].values[0] 
                   if len(method_df[method_df['Participation_Rate'] == r]) > 0 else 0 
                   for r in participation_rates]
            stds = [method_df[method_df['Participation_Rate'] == r]['Accuracy_Std'].values[0] 
                   if len(method_df[method_df['Participation_Rate'] == r]) > 0 else 0 
                   for r in participation_rates]
            all_vals.extend(avgs)
            all_stds.extend(stds)
            
            offset = (i - len(selection_methods)/2 + 0.5) * width
            ax.bar(x + offset, avgs, width, yerr=stds, label=method, color=colors[i], alpha=0.8, capsize=2)
        
        ax.set_xlabel('Participation Rate')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{fitness_mode.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis with proper margins for error bars
        if all_vals:
            y_min, y_max = get_y_limits_with_margin(all_vals, all_stds)
            ax.set_ylim(max(0, y_min), min(1, y_max))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/accuracy_by_participation_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy comparison to {output_dir}/accuracy_by_participation_rate.png")
    
    # 3. Performance trends with Loss included
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Trends Across Participation Rates', fontsize=14, fontweight='bold')
    
    # Fitness trend
    ax1 = axes[0, 0]
    for idx, fitness_mode in enumerate(fitness_modes):
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        avg_by_rate = mode_df.groupby('Participation_Rate')['Fitness_Avg'].mean()
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[idx])
        ax1.plot(participation_rates, [avg_by_rate.get(r, 0) for r in participation_rates], 
                 linewidth=2, label=fitness_mode.upper(), color=mode_color)
    ax1.set_xlabel('Participation Rate')
    ax1.set_ylabel('Average Fitness')
    ax1.set_title('Average Fitness vs Participation Rate')
    ax1.set_xticks(participation_rates)
    ax1.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy trend
    ax2 = axes[0, 1]
    for idx, fitness_mode in enumerate(fitness_modes):
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        avg_by_rate = mode_df.groupby('Participation_Rate')['Accuracy_Avg'].mean()
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[idx])
        ax2.plot(participation_rates, [avg_by_rate.get(r, 0) for r in participation_rates], 
                marker='s', linewidth=2, label=fitness_mode.upper(), color=mode_color)
    ax2.set_xlabel('Participation Rate')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Average Accuracy vs Participation Rate')
    ax2.set_xticks(participation_rates)
    ax2.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss trend (NEW)
    ax3 = axes[0, 2]
    for idx, fitness_mode in enumerate(fitness_modes):
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        avg_by_rate = mode_df.groupby('Participation_Rate')['Loss_Avg'].mean()
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[idx])
        ax3.plot(participation_rates, [avg_by_rate.get(r, 0) for r in participation_rates], 
                marker='^', linewidth=2, label=fitness_mode.upper(), color=mode_color)
    ax3.set_xlabel('Participation Rate')
    ax3.set_ylabel('Average Loss')
    ax3.set_title('Average Loss vs Participation Rate\n(Lower = Better)')
    ax3.set_xticks(participation_rates)
    ax3.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Energy per client (selected)
    ax4 = axes[1, 0]
    for idx, fitness_mode in enumerate(fitness_modes):
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        avg_by_rate = mode_df.groupby('Participation_Rate')['Energy_Per_Selected_Client'].mean()
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[idx])
        ax4.plot(participation_rates, [avg_by_rate.get(r, 0) for r in participation_rates], 
                marker='D', linewidth=2, label=fitness_mode.upper(), color=mode_color)
    ax4.set_xlabel('Participation Rate')
    ax4.set_ylabel('Energy per Selected Client')
    ax4.set_title('Energy Consumption per Selected Client')
    ax4.set_xticks(participation_rates)
    ax4.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Efficiency Score (Accuracy / Energy per client)
    ax5 = axes[1, 1]
    for idx, fitness_mode in enumerate(fitness_modes):
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        avg_by_rate = mode_df.groupby('Participation_Rate')['Efficiency_Score'].mean()
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[idx])
        ax5.plot(participation_rates, [avg_by_rate.get(r, 0) for r in participation_rates], 
                marker='*', linewidth=2, markersize=10, label=fitness_mode.upper(), color=mode_color)
    ax5.set_xlabel('Participation Rate')
    ax5.set_ylabel('Efficiency Score')
    ax5.set_title('Efficiency Score vs Participation Rate\n(Accuracy×100 / Energy per Client)')
    ax5.set_xticks(participation_rates)
    ax5.set_xticklabels([f'{int(r*100)}%' for r in participation_rates])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Summary table with efficiency metrics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_data = []
    for rate in participation_rates:
        rate_df = df[df['Participation_Rate'] == rate]
        n_selected = rate_df['N_Selected_Clients'].iloc[0] if len(rate_df) > 0 else 0
        n_total = rate_df['N_Total_Clients'].iloc[0] if len(rate_df) > 0 else 0
        summary_data.append([
            f'{int(rate*100)}%',
            f'{n_selected}/{n_total}',
            f'{rate_df["Accuracy_Avg"].mean():.4f}',
            f'{rate_df["Loss_Avg"].mean():.4f}',
            f'{rate_df["Energy_Avg"].mean():.2f}',
            f'{rate_df["Energy_Per_Selected_Client"].mean():.2f}',
            f'{rate_df["Efficiency_Score"].mean():.2f}'
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Rate', 'Clients', 'Acc', 'Loss', 'Total E', 'E/Client', 'Efficiency'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    ax6.set_title('Summary by Participation Rate', fontsize=12, pad=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/participation_rate_trends.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved trend plots to {output_dir}/participation_rate_trends.png")
    
    # 4. Efficiency Analysis Plot (Accuracy vs Energy per Client)
    fig, axes = plt.subplots(1, len(fitness_modes), figsize=(6*len(fitness_modes), 6))
    if len(fitness_modes) == 1:
        axes = [axes]
    
    fig.suptitle('Accuracy vs Energy Efficiency\n(Top-Left = Best: High Accuracy, Low Energy)', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, fitness_mode in enumerate(fitness_modes):
        ax = axes[ax_idx]
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        
        for i, method in enumerate(selection_methods):
            method_df = mode_df[mode_df['Selection_Method'] == method]
            if len(method_df) > 0:
                for _, row in method_df.iterrows():
                    rate = row['Participation_Rate']
                    ax.scatter(row['Energy_Per_Selected_Client'], row['Accuracy_Avg'], 
                              s=100, c=[colors[i]], marker='o', alpha=0.7)
                    ax.annotate(f"{method}\n{int(rate*100)}%", 
                               (row['Energy_Per_Selected_Client'], row['Accuracy_Avg']),
                               textcoords="offset points", xytext=(5, 5), fontsize=7)
        
        # Add trend lines for each rate
        for rate in participation_rates:
            rate_df = mode_df[mode_df['Participation_Rate'] == rate]
            if len(rate_df) > 1:
                ax.plot(rate_df['Energy_Per_Selected_Client'], rate_df['Accuracy_Avg'], 
                       '--', alpha=0.3, color='gray')
        
        ax.set_xlabel('Energy per Selected Client')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{fitness_mode.upper()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/accuracy_vs_energy_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved efficiency plot to {output_dir}/accuracy_vs_energy_efficiency.png")
    
    # 5. 3D Visualization: Accuracy (X) vs Participation Rate (Y) vs Relative Energy (Z)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 6*len(fitness_modes)))
        
        for mode_idx, fitness_mode in enumerate(fitness_modes):
            ax = fig.add_subplot(1, len(fitness_modes), mode_idx + 1, projection='3d')
            mode_df = df[df['Fitness_Mode'] == fitness_mode]
            
            # Normalize energy for this mode
            if len(mode_df) > 0:
                max_energy = mode_df['Energy_Avg'].max()
                mode_df = mode_df.copy()
                mode_df['Relative_Energy'] = mode_df['Energy_Avg'] / max_energy * 100
            
            for i, method in enumerate(selection_methods):
                method_df = mode_df[mode_df['Selection_Method'] == method]
                if len(method_df) > 0:
                    x = method_df['Accuracy_Avg'].values
                    y = method_df['Participation_Rate'].values
                    z = method_df['Relative_Energy'].values
                    ax.scatter(x, y, z, c=[colors[i]], s=100, label=method, alpha=0.8)
                    
                    # Connect points for same method
                    sorted_idx = np.argsort(y)
                    ax.plot(x[sorted_idx], y[sorted_idx], z[sorted_idx], 
                           c=colors[i], alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Accuracy', fontsize=10)
            ax.set_ylabel('Participation Rate', fontsize=10)
            ax.set_zlabel('Relative Energy (%)', fontsize=10)
            ax.set_title(f'{fitness_mode.upper()}\n3D: Accuracy vs Rate vs Energy', fontsize=12)
            ax.legend(loc='upper left', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3d_accuracy_rate_energy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved 3D visualization to {output_dir}/3d_accuracy_rate_energy.png")
    except Exception as e:
        print(f"  Warning: Could not create 3D plot: {e}")
    
    # 6. Comprehensive Efficiency Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Efficiency Metrics Heatmaps by Participation Rate', fontsize=14, fontweight='bold')
    
    # Prepare data for heatmaps
    metrics_to_plot = [
        ('Accuracy_Avg', 'Accuracy', axes[0, 0], 'RdYlGn'),
        ('Energy_Per_Selected_Client', 'Energy per Client', axes[0, 1], 'RdYlGn_r'),
        ('Efficiency_Score', 'Efficiency Score', axes[1, 0], 'RdYlGn'),
        ('Loss_Avg', 'Loss', axes[1, 1], 'RdYlGn_r')
    ]
    
    for metric, title, ax, cmap in metrics_to_plot:
        # Create pivot table (average across fitness modes)
        pivot_data = df.groupby(['Selection_Method', 'Participation_Rate'])[metric].mean().unstack()
        
        # Reorder columns
        pivot_data = pivot_data[[r for r in participation_rates if r in pivot_data.columns]]
        pivot_data.columns = [f'{int(r*100)}%' for r in pivot_data.columns]
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                   cbar_kws={'label': title})
        ax.set_title(f'{title} by Method and Participation Rate')
        ax.set_xlabel('Participation Rate')
        ax.set_ylabel('Selection Method')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/efficiency_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved efficiency heatmaps to {output_dir}/efficiency_heatmaps.png")
    
    # 7. Academic comparison: Pareto frontier plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Pareto Frontier Analysis: Accuracy vs Energy Consumption\n(Per Selected Client)', 
                fontsize=14, fontweight='bold')
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for mode_idx, fitness_mode in enumerate(fitness_modes):
        mode_df = df[df['Fitness_Mode'] == fitness_mode]
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[mode_idx])
        
        for i, method in enumerate(selection_methods):
            method_df = mode_df[mode_df['Selection_Method'] == method]
            if len(method_df) > 0:
                for _, row in method_df.iterrows():
                    marker = markers[i % len(markers)]
                    ax.scatter(row['Energy_Per_Selected_Client'], row['Accuracy_Avg'],
                              s=100, c=[mode_color], marker=marker, alpha=0.7,
                              edgecolors='black', linewidths=0.5)
    
    # Add legend for methods
    for i, method in enumerate(selection_methods):
        ax.scatter([], [], marker=markers[i % len(markers)], c='gray', 
                  s=100, label=method, edgecolors='black')
    
    # Add legend for fitness modes
    for mode_idx, fitness_mode in enumerate(fitness_modes):
        mode_color = FITNESS_MODES.get(fitness_mode, {}).get('color', colors[mode_idx])
        ax.scatter([], [], marker='o', c=[mode_color], s=100, 
                  label=f'● {fitness_mode.upper()}')
    
    ax.set_xlabel('Energy per Selected Client (Lower = Better)', fontsize=12)
    ax.set_ylabel('Accuracy (Higher = Better)', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add rate annotations
    for _, row in df.iterrows():
        ax.annotate(f'{int(row["Participation_Rate"]*100)}%', 
                   (row['Energy_Per_Selected_Client'], row['Accuracy_Avg']),
                   textcoords="offset points", xytext=(3, 3), fontsize=6, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Pareto frontier to {output_dir}/pareto_frontier.png")
    
    # 8. Save comprehensive efficiency CSV
    efficiency_summary = []
    for rate in participation_rates:
        rate_df = df[df['Participation_Rate'] == rate]
        for fitness_mode in fitness_modes:
            mode_df = rate_df[rate_df['Fitness_Mode'] == fitness_mode]
            for method in selection_methods:
                method_df = mode_df[mode_df['Selection_Method'] == method]
                if len(method_df) > 0:
                    row = method_df.iloc[0]
                    efficiency_summary.append({
                        'Participation_Rate': f'{int(rate*100)}%',
                        'Fitness_Mode': fitness_mode.upper(),
                        'Method': method,
                        'Accuracy': f'{row["Accuracy_Avg"]:.4f}',
                        'Loss': f'{row["Loss_Avg"]:.4f}',
                        'Total_Energy': f'{row["Energy_Avg"]:.2f}',
                        'N_Selected_Clients': int(row['N_Selected_Clients']),
                        'N_Total_Clients': int(row['N_Total_Clients']),
                        'Energy_Per_Selected': f'{row["Energy_Per_Selected_Client"]:.3f}',
                        'Energy_Per_Total': f'{row["Energy_Per_Total_Client"]:.3f}',
                        'Accuracy_Per_Energy': f'{row["Accuracy_Per_Energy"]:.6f}',
                        'Efficiency_Score': f'{row["Efficiency_Score"]:.2f}'
                    })
    
    efficiency_df = pd.DataFrame(efficiency_summary)
    efficiency_df.to_csv(f'{output_dir}/comprehensive_efficiency_analysis.csv', index=False)
    print(f"✓ Saved efficiency analysis to {output_dir}/comprehensive_efficiency_analysis.csv")

def plot_active_vs_selected_clients(results_by_combo, output_dir, n_rounds):
    """Plot active vs selected clients across all experiments"""
    print("\n" + "=" * 40)
    print("PLOTTING ACTIVE vs SELECTED CLIENTS")
    print("=" * 40)
    
    all_active, all_selected = [], []
    
    for combo_key, data in results_by_combo.items():
        for active_hist in data.get('active_clients_histories', []):
            if active_hist and len(active_hist) > 0:
                all_active.append(np.array(active_hist))
        for selected_hist in data.get('selected_clients_histories', []):
            if selected_hist and len(selected_hist) > 0:
                all_selected.append(np.array(selected_hist))
    
    if not all_active or not all_selected:
        print("⚠ No active/selected client data")
        return
    
    max_rounds = max(len(a) for a in all_active + all_selected)
    
    def pad_array(arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len]
        return np.pad(arr, (0, target_len - len(arr)), mode='edge')
    
    all_active = [pad_array(a, max_rounds) for a in all_active]
    all_selected = [pad_array(a, max_rounds) for a in all_selected]
    
    active_matrix = np.array(all_active)
    selected_matrix = np.array(all_selected)
    
    rounds = np.arange(1, max_rounds + 1)
    
    active_avg = np.mean(active_matrix, axis=0)
    active_min = np.min(active_matrix, axis=0)
    active_max = np.max(active_matrix, axis=0)
    
    selected_avg = np.mean(selected_matrix, axis=0)
    selected_min = np.min(selected_matrix, axis=0)
    selected_max = np.max(selected_matrix, axis=0)
    
    total_clients = 0
    for data in results_by_combo.values():
        if data.get('total_clients', 0) > 0:
            total_clients = data['total_clients']
            break
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Active vs Selected Clients Analysis', fontsize=14, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(rounds, active_avg, 'b-', linewidth=2, label='Active (Avg)', marker='o', markersize=3)
    ax1.fill_between(rounds, active_min, active_max, color='blue', alpha=0.2)
    ax1.plot(rounds, selected_avg, 'g-', linewidth=2, label='Selected (Avg)', marker='s', markersize=3)
    ax1.fill_between(rounds, selected_min, selected_max, color='green', alpha=0.2)
    if total_clients > 0:
        ax1.axhline(y=total_clients, color='red', linestyle='--', linewidth=1.5, label=f'Total ({total_clients})')
    ax1.set_xlabel('FL Round')
    ax1.set_ylabel('Number of Clients')
    ax1.set_title('Active vs Selected Clients')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(rounds, active_avg, 'b-', linewidth=2, label='Active (Avg)')
    ax2.fill_between(rounds, active_avg - np.std(active_matrix, axis=0), active_avg + np.std(active_matrix, axis=0), color='blue', alpha=0.3)
    ax2.set_xlabel('FL Round')
    ax2.set_ylabel('Active Clients')
    ax2.set_title('Active Clients Detail')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(rounds, selected_avg, 'g-', linewidth=2, label='Selected (Avg)')
    ax3.fill_between(rounds, selected_avg - np.std(selected_matrix, axis=0), selected_avg + np.std(selected_matrix, axis=0), color='green', alpha=0.3)
    ax3.set_xlabel('FL Round')
    ax3.set_ylabel('Selected Clients')
    ax3.set_title('Selected Clients Detail')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_avg = np.where(active_avg > 0, selected_avg / active_avg, 0)
    ax4.plot(rounds, ratio_avg, 'purple', linewidth=2, label='Selection Ratio')
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='100%')
    ax4.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='50%')
    ax4.set_xlabel('FL Round')
    ax4.set_ylabel('Ratio (Selected/Active)')
    ax4.set_title('Client Selection Ratio')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}/active_vs_selected_clients.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved active vs selected clients plot")
    
    stats_df = pd.DataFrame({
        'Round': rounds,
        'Active_Avg': active_avg, 'Active_Min': active_min, 'Active_Max': active_max,
        'Selected_Avg': selected_avg, 'Selected_Min': selected_min, 'Selected_Max': selected_max,
        'Selection_Ratio_Avg': ratio_avg,
    })
    stats_df.to_csv(f'{output_dir}/active_selected_clients_stats.csv', index=False)
    print(f"✓ Saved client statistics CSV")

def _plot_byzantine_results(df, attack_types, byzantine_fractions, defense_methods,
                           selection_methods, output_dir, convergence_data=None):
    """Generate comprehensive Byzantine experiment plots"""
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(defense_methods)))
    attack_types_no_none = [a for a in attack_types if a != 'none']
    
    # Handle edge case of empty attack types
    if len(attack_types_no_none) == 0:
        print("Warning: No attack types to plot")
        return
    
    # 1. Accuracy vs Byzantine Fraction (for each attack type)
    n_attack_plots = max(1, len(attack_types_no_none))
    fig, axes = plt.subplots(1, n_attack_plots, figsize=(6*n_attack_plots, 5))
    if n_attack_plots == 1:
        axes = [axes]
    
    fig.suptitle('Model Accuracy Under Byzantine Attacks\n(by Defense Method)', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, attack_type in enumerate(attack_types_no_none):
        ax = axes[ax_idx]
        attack_df = df[df['Attack_Type'] == attack_type]
        
        for i, defense in enumerate(defense_methods):
            defense_df = attack_df[attack_df['Defense_Method'] == defense]
            
            means = []
            stds = []
            for frac in byzantine_fractions:
                frac_df = defense_df[defense_df['Byzantine_Fraction'] == frac]
                if len(frac_df) > 0:
                    means.append(frac_df['Final_Accuracy'].mean())
                    stds.append(frac_df['Final_Accuracy'].std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            ax.errorbar([f*100 for f in byzantine_fractions], means, yerr=stds,
                       marker='o', label=defense.upper(), color=colors[i],
                       linewidth=2, capsize=4)
        
        ax.set_xlabel('Byzantine Fraction (%)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{attack_type.upper()} Attack')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_byzantine_fraction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy vs Byzantine fraction plot")
    
    # 2. Defense Method Comparison Heatmap
    fig, axes = plt.subplots(1, len(attack_types_no_none), figsize=(5*len(attack_types_no_none), 4))
    if len(attack_types_no_none) == 1:
        axes = [axes]
    
    fig.suptitle('Defense Method Effectiveness\n(Accuracy at Different Byzantine Fractions)', 
                fontsize=14, fontweight='bold')
    
    for ax_idx, attack_type in enumerate(attack_types_no_none):
        ax = axes[ax_idx]
        attack_df = df[df['Attack_Type'] == attack_type]
        
        # Create pivot table
        pivot_data = attack_df.groupby(['Defense_Method', 'Byzantine_Fraction'])['Final_Accuracy'].mean().unstack()
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                       vmin=0.5, vmax=1.0, cbar_kws={'label': 'Accuracy'})
            ax.set_title(f'{attack_type.upper()} Attack')
            ax.set_xlabel('Byzantine Fraction')
            ax.set_ylabel('Defense Method')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/defense_effectiveness_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved defense effectiveness heatmap")
    
    # 3. Accuracy Degradation Plot (relative to no attack)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get baseline accuracy (no attack)
    baseline_df = df[df['Attack_Type'] == 'none']
    if len(baseline_df) > 0:
        baseline_acc = baseline_df['Final_Accuracy'].mean()
    else:
        # Use 0% byzantine as baseline
        baseline_acc = df[df['Byzantine_Fraction'] == 0]['Final_Accuracy'].mean()
    
    bar_width = 0.2
    x = np.arange(len(byzantine_fractions[1:]))  # Skip 0%
    
    for i, defense in enumerate(defense_methods):
        defense_df = df[(df['Defense_Method'] == defense) & (df['Attack_Type'] != 'none')]
        
        degradations = []
        for frac in byzantine_fractions[1:]:  # Skip 0%
            frac_df = defense_df[defense_df['Byzantine_Fraction'] == frac]
            if len(frac_df) > 0:
                acc = frac_df['Final_Accuracy'].mean()
                degradation = (baseline_acc - acc) / baseline_acc * 100
                degradations.append(degradation)
            else:
                degradations.append(0)
        
        offset = (i - len(defense_methods)/2 + 0.5) * bar_width
        ax.bar(x + offset, degradations, bar_width, label=defense.upper(), color=colors[i])
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Byzantine Fraction (%)')
    ax.set_ylabel('Accuracy Degradation (%)')
    ax.set_title('Accuracy Degradation Relative to No Attack\n(Lower = Better Defense)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(f*100)}%' for f in byzantine_fractions[1:]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy degradation plot")
    
    # 4. Detection Rate Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    detection_data = df[df['Byzantine_Fraction'] > 0].groupby('Defense_Method')['Avg_Detection_Rate'].mean()
    
    if len(detection_data) > 0:
        bars = ax.bar(detection_data.index, detection_data.values * 100, color=colors[:len(detection_data)])
        ax.set_xlabel('Defense Method')
        ax.set_ylabel('Average Detection Rate (%)')
        ax.set_title('Byzantine Client Detection Rate by Defense Method')
        ax.set_ylim(0, 100)
        
        for bar, val in zip(bars, detection_data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/detection_rate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved detection rate plot")
    
    # 5. Baseline vs Attack Comparison (NEW - Important for proving defense works)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get baseline accuracy (no attack)
    baseline_df = df[df['Attack_Type'] == 'none']
    baseline_by_defense = baseline_df.groupby('Defense_Method')['Final_Accuracy'].agg(['mean', 'std'])
    
    # 5a. Bar chart: Baseline vs With Attack for each defense
    ax = axes[0]
    x = np.arange(len(defense_methods))
    width = 0.35
    
    baseline_means = [baseline_by_defense.loc[d, 'mean'] if d in baseline_by_defense.index else 0 for d in defense_methods]
    baseline_stds = [baseline_by_defense.loc[d, 'std'] if d in baseline_by_defense.index else 0 for d in defense_methods]
    
    # Find the highest Byzantine fraction available for attack comparison
    attack_df = df[df['Attack_Type'] != 'none']
    if len(attack_df) > 0:
        available_fractions = sorted(attack_df['Byzantine_Fraction'].unique())
        # Use the highest fraction available (or 0.2 if available)
        if 0.2 in available_fractions:
            target_fraction = 0.2
        else:
            target_fraction = max(available_fractions) if available_fractions else 0.1
        
        attack_target_df = attack_df[attack_df['Byzantine_Fraction'] == target_fraction]
        attack_by_defense = attack_target_df.groupby('Defense_Method')['Final_Accuracy'].agg(['mean', 'std'])
        attack_means = [attack_by_defense.loc[d, 'mean'] if d in attack_by_defense.index else 0 for d in defense_methods]
        attack_stds = [attack_by_defense.loc[d, 'std'] if d in attack_by_defense.index else 0 for d in defense_methods]
        attack_label = f'{int(target_fraction*100)}% Byzantine Attack'
    else:
        attack_means = [0] * len(defense_methods)
        attack_stds = [0] * len(defense_methods)
        target_fraction = 0.2
        attack_label = '20% Byzantine Attack'
    
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                   label='No Attack (Baseline)', color='#2ecc71', capsize=4)
    bars2 = ax.bar(x + width/2, attack_means, width, yerr=attack_stds,
                   label=attack_label, color='#e74c3c', capsize=4)
    
    ax.set_xlabel('Defense Method')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Defense Effectiveness: Baseline vs Under Attack ({int(target_fraction*100)}% Byzantine)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in defense_methods])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy preservation percentage
    for i, (base, atk) in enumerate(zip(baseline_means, attack_means)):
        if base > 0 and atk > 0:
            preservation = (atk / base) * 100
            ax.annotate(f'{preservation:.1f}%', xy=(i + width/2, atk + 0.02),
                       ha='center', va='bottom', fontsize=9, color='#e74c3c')
    
    # 5b. Accuracy drop comparison
    ax = axes[1]
    
    drops = []
    for defense in defense_methods:
        base = baseline_by_defense.loc[defense, 'mean'] if defense in baseline_by_defense.index else 0
        atk = attack_by_defense.loc[defense, 'mean'] if defense in attack_by_defense.index else 0
        drop = (base - atk) / base * 100 if base > 0 else 0
        drops.append(drop)
    
    bar_colors = ['#2ecc71' if d < 5 else '#f39c12' if d < 10 else '#e74c3c' for d in drops]
    bars = ax.bar([d.upper() for d in defense_methods], drops, color=bar_colors)
    
    ax.axhline(y=5, color='#2ecc71', linestyle='--', alpha=0.7, label='Good (<5%)')
    ax.axhline(y=10, color='#f39c12', linestyle='--', alpha=0.7, label='Moderate (<10%)')
    
    ax.set_xlabel('Defense Method')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title(f'Accuracy Drop Under {int(target_fraction*100)}% Byzantine Attack\n(Lower = Better Defense)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{drop:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_vs_attack.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved baseline vs attack comparison plot")
    
    # 6. Convergence Curves Under Attack (NEW)
    if convergence_data:
        # Find available Byzantine fractions in convergence data
        conv_fractions = set(d['Byzantine_Fraction'] for d in convergence_data if d['Attack_Type'] != 'none')
        if 0.2 in conv_fractions:
            conv_target_fraction = 0.2
        elif conv_fractions:
            conv_target_fraction = max(conv_fractions)
        else:
            conv_target_fraction = 0.1
        
        # Plot convergence curves for each attack type
        fig, axes = plt.subplots(2, max(1, len(attack_types_no_none)), 
                                figsize=(6*max(1, len(attack_types_no_none)), 10))
        if len(attack_types_no_none) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Convergence Under Byzantine Attacks ({int(conv_target_fraction*100)}% Byzantine Clients)', 
                    fontsize=14, fontweight='bold')
        
        for ax_idx, attack_type in enumerate(attack_types_no_none):
            # Accuracy convergence
            ax_acc = axes[0, ax_idx] if len(attack_types_no_none) > 1 else axes[0, 0]
            ax_loss = axes[1, ax_idx] if len(attack_types_no_none) > 1 else axes[1, 0]
            
            for i, defense in enumerate(defense_methods):
                # Find convergence data for this combination
                matching = [d for d in convergence_data 
                           if d['Attack_Type'] == attack_type 
                           and d['Defense_Method'] == defense
                           and d['Byzantine_Fraction'] == conv_target_fraction]
                
                if matching:
                    acc_hist = matching[0]['Accuracy_History']
                    loss_hist = matching[0]['Loss_History']
                    rounds = range(1, len(acc_hist) + 1)
                    
                    ax_acc.plot(rounds, acc_hist, label=defense.upper(), 
                               color=colors[i], linewidth=2)
                    ax_loss.plot(rounds, loss_hist, label=defense.upper(),
                                color=colors[i], linewidth=2)
            
            ax_acc.set_xlabel('FL Round')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_title(f'{attack_type.upper()} Attack - Accuracy')
            ax_acc.legend(loc='lower right')
            ax_acc.grid(True, alpha=0.3)
            ax_acc.set_ylim(0, 1)
            
            ax_loss.set_xlabel('FL Round')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title(f'{attack_type.upper()} Attack - Loss')
            ax_loss.legend(loc='upper right')
            ax_loss.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_under_attack.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved convergence under attack plot")
    
    # 6. Box Plot Comparison (NEW)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot for accuracy by defense method (under attack only)
    attack_df = df[df['Attack_Type'] != 'none']
    if len(attack_df) > 0:
        ax = axes[0]
        defense_order = defense_methods
        sns.boxplot(data=attack_df, x='Defense_Method', y='Final_Accuracy', 
                   order=defense_order, palette='Set2', ax=ax)
        ax.set_xlabel('Defense Method')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Accuracy Distribution by Defense Method\n(All Attacks Combined)')
        ax.set_ylim(0, 1)
        
        # Add mean markers
        means = attack_df.groupby('Defense_Method')['Final_Accuracy'].mean()
        for i, defense in enumerate(defense_order):
            if defense in means.index:
                ax.scatter(i, means[defense], color='red', s=100, zorder=5, marker='D')
    
    # Box plot for accuracy by attack type
    ax = axes[1]
    if len(attack_df) > 0:
        attack_order = attack_types_no_none
        sns.boxplot(data=attack_df, x='Attack_Type', y='Final_Accuracy',
                   order=attack_order, palette='Set3', ax=ax)
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Accuracy Distribution by Attack Type\n(All Defenses Combined)')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy box plots")
    
    # 7. Statistical Comparison Table (Friedman Test) (NEW)
    try:
        # Prepare data for Friedman test
        friedman_data = []
        for defense in defense_methods:
            defense_accs = attack_df[attack_df['Defense_Method'] == defense]['Final_Accuracy'].values
            if len(defense_accs) > 0:
                friedman_data.append(defense_accs[:min(len(d) for d in [attack_df[attack_df['Defense_Method'] == dm]['Final_Accuracy'].values for dm in defense_methods] if len(d) > 0)])
        
        if len(friedman_data) >= 3 and all(len(d) >= 3 for d in friedman_data):
            # Ensure equal length for Friedman test
            min_len = min(len(d) for d in friedman_data)
            friedman_data = [d[:min_len] for d in friedman_data]
            
            from scipy.stats import friedmanchisquare
            stat, p_value = friedmanchisquare(*friedman_data)
            
            # Calculate mean ranks
            all_data = np.array(friedman_data).T
            ranks = np.apply_along_axis(lambda x: len(x) + 1 - rankdata(x), 1, all_data)
            mean_ranks = np.mean(ranks, axis=0)
            
            # Save statistical results
            stats_results = {
                'Friedman_Statistic': stat,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            }
            for i, defense in enumerate(defense_methods):
                stats_results[f'{defense}_Mean_Rank'] = mean_ranks[i]
            
            with open(f'{output_dir}/friedman_test_results.json', 'w') as f:
                json.dump(stats_results, f, indent=2)
            print(f"✓ Saved Friedman test results (χ²={stat:.2f}, p={p_value:.4f})")
    except Exception as e:
        print(f"Note: Could not perform Friedman test: {e}")
    
    # 8. Summary Statistics Table Image (NEW)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create summary table
    summary_data = []
    for defense in defense_methods:
        for attack in attack_types_no_none:
            subset = df[(df['Defense_Method'] == defense) & (df['Attack_Type'] == attack)]
            if len(subset) > 0:
                summary_data.append({
                    'Defense': defense.upper(),
                    'Attack': attack.upper(),
                    'Accuracy': f"{subset['Final_Accuracy'].mean():.3f} ± {subset['Final_Accuracy'].std():.3f}",
                    'Loss': f"{subset['Final_Loss'].mean():.3f} ± {subset['Final_Loss'].std():.3f}",
                    'Detection': f"{subset['Avg_Detection_Rate'].mean()*100:.1f}%"
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        table = ax.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0']*len(summary_df.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Byzantine Defense Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved summary table image")

def _plot_byzantine_method_comparison(df, attack_types, byzantine_fractions,
                                       defense_methods, selection_methods, output_dir):
    """
    Compare selection methods (e.g. GWO vs DE) head-to-head under Byzantine
    attack. Only meaningful when selection_methods has 2+ entries -- for a
    single-method run (the historical default), this is a no-op.

    This directly tests the manuscript's claim that GWO's lower baseline
    (Part I, clean-training) fitness-optimization variance translates into
    more stable behavior specifically under attack -- that claim was
    previously asserted from clean-training data alone. Produces:
      1. accuracy_vs_fraction_by_method.png -- accuracy vs Byzantine fraction,
         one line per selection method, averaged across defenses, faceted by
         attack type.
      2. method_accuracy_variance.png -- bar chart of run-to-run std of
         Final_Accuracy per selection method, under attack only (fraction>0),
         the direct empirical test of the variance claim.
      3. method_comparison_summary.csv -- mean/std accuracy per
         (selection_method, attack_type, defense_method, byzantine_fraction).
    """
    if len(selection_methods) < 2:
        print("Only one selection method in this run -- skipping method comparison plots.")
        return

    attack_types_no_none = [a for a in attack_types if a != 'none']
    if len(attack_types_no_none) == 0:
        print("Warning: No attack types to plot for method comparison")
        return

    method_colors = plt.cm.tab10(np.linspace(0, 1, len(selection_methods)))

    # --- 1. Accuracy vs Byzantine fraction, one line per method, per attack type ---
    n_attack_plots = max(1, len(attack_types_no_none))
    fig, axes = plt.subplots(1, n_attack_plots, figsize=(6 * n_attack_plots, 5))
    if n_attack_plots == 1:
        axes = [axes]

    fig.suptitle('Selection Method Comparison Under Byzantine Attack\n'
                 '(Accuracy vs Byzantine Fraction, averaged across defenses)',
                 fontsize=14, fontweight='bold')

    for ax_idx, attack_type in enumerate(attack_types_no_none):
        ax = axes[ax_idx]
        attack_df = df[df['Attack_Type'] == attack_type]

        for i, method in enumerate(selection_methods):
            method_df = attack_df[attack_df['Selection_Method'] == method]

            means, stds = [], []
            for frac in byzantine_fractions:
                frac_df = method_df[method_df['Byzantine_Fraction'] == frac]
                if len(frac_df) > 0:
                    means.append(frac_df['Final_Accuracy'].mean())
                    stds.append(frac_df['Final_Accuracy'].std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            ax.errorbar([f * 100 for f in byzantine_fractions], means, yerr=stds,
                       marker='o', label=method, color=method_colors[i],
                       linewidth=2, capsize=4)

        ax.set_xlabel('Byzantine Fraction (%)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{attack_type.upper()} Attack')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_vs_fraction_by_method.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\u2713 Saved accuracy vs Byzantine fraction by selection method")

    # --- 2. Variance comparison under attack -- the core empirical test ---
    attacked_df = df[df['Byzantine_Fraction'] > 0]
    variance_by_method = attacked_df.groupby('Selection_Method')['Final_Accuracy'].std()
    mean_by_method = attacked_df.groupby('Selection_Method')['Final_Accuracy'].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    methods_present = [m for m in selection_methods if m in variance_by_method.index]
    bars = ax.bar(methods_present, [variance_by_method[m] for m in methods_present],
                  color=method_colors[:len(methods_present)])
    ax.set_ylabel('Std Dev of Final Accuracy (across runs, all attacked configs)')
    ax.set_title('Selection-Method Stability Under Byzantine Attack\n'
                 '(Lower = more consistent behavior when attacked)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, m in zip(bars, methods_present):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               f'\u03c3={variance_by_method[m]:.4f}\n\u03bc={mean_by_method[m]:.3f}',
               ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/method_accuracy_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    variance_summary_parts = []
    for m in methods_present:
        variance_summary_parts.append(f"{m}: sigma={variance_by_method[m]:.4f}")
    print("\u2713 Saved method stability-under-attack comparison ("
         + ", ".join(variance_summary_parts) + ")")

    # --- 3. Full summary table ---
    summary = (df.groupby(['Selection_Method', 'Attack_Type', 'Defense_Method', 'Byzantine_Fraction'])
               ['Final_Accuracy'].agg(['mean', 'std', 'count']).reset_index())
    summary.to_csv(f'{output_dir}/method_comparison_summary.csv', index=False)
    print(f"\u2713 Saved method_comparison_summary.csv ({len(summary)} rows)")


def generate_byzantine_summary(df, output_dir):
    """Generate summary statistics for Byzantine experiment"""
    
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("BYZANTINE RESILIENCE EXPERIMENT SUMMARY")
    summary_lines.append("=" * 60)
    
    summary_lines.append("\n1. OVERALL STATISTICS")
    summary_lines.append("-" * 40)
    
    for defense in df['Defense_Method'].unique():
        defense_df = df[df['Defense_Method'] == defense]
        summary_lines.append(f"\n{defense.upper()} Defense:")
        summary_lines.append(f"  Avg Accuracy: {defense_df['Final_Accuracy'].mean():.4f} ± {defense_df['Final_Accuracy'].std():.4f}")
        summary_lines.append(f"  Best: {defense_df['Final_Accuracy'].max():.4f}")
        summary_lines.append(f"  Worst: {defense_df['Final_Accuracy'].min():.4f}")
    
    summary_lines.append("\n\n2. BEST DEFENSE PER ATTACK")
    summary_lines.append("-" * 40)
    
    for attack in df['Attack_Type'].unique():
        if attack == 'none':
            continue
        attack_df = df[df['Attack_Type'] == attack]
        best_defense = attack_df.groupby('Defense_Method')['Final_Accuracy'].mean().idxmax()
        best_acc = attack_df.groupby('Defense_Method')['Final_Accuracy'].mean().max()
        summary_lines.append(f"{attack.upper()}: {best_defense.upper()} (Acc: {best_acc:.4f})")
    
    summary_text = '\n'.join(summary_lines)
    with open(f'{output_dir}/byzantine_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"✓ Saved Byzantine summary")
    print("\n" + summary_text)
