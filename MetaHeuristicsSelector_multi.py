# -*- coding: utf-8 -*-
"""
CORRECTED MetaHeuristicsSelector for FL Client Selection
Implements proper subset-based fitness optimization as per the PDF formula

Key Differences from Previous Version:
1. Binary solution encoding (not continuous ranking)
2. Evaluates SUBSET fitness (not individual client scores)
3. Proper constraint handling (exactly K clients)
4. Multi-objective weighted sum from PDF
"""

from collections import defaultdict
import numpy as np
from OptimizersV2 import optimizers


class ClientSelector:
    
    def __init__(self, clients_pool, k=10, env=None,fitness_mode="learning"):
        """
        Initialize the client selector
        
        Args:
            clients_pool: Either Environment object (DynaEdgeSim) or dict of FL clients
            k: Number of clients to select
            env: Optional Environment for accessing mobility/kinetic data
        """
        self.clients_pool = clients_pool
        self.k = k
        self.fitness_mode = fitness_mode  # "resource", "learning", "exploration"
        self.env = env if env is not None else clients_pool
        self.optimizer_name = None
        self.optimizer_config = None
        self.optimizer_params = None
        self.progress_callback = None  # Callback for progress updates

        # PERF: per-round cache of normalized client metrics. Metrics only
        # depend on the client's current state (battery, reliability, etc.),
        # NOT on the candidate solution being evaluated -- so they're
        # identical across all agents_nbr * max_iter fitness evaluations
        # within one select_clients() call. Built once in select_clients(),
        # consumed by subset_fitness() and _calculate_selection_metrics()
        # instead of recomputing per evaluation.
        self._metrics_cache = None
        
        # Fitness weights (from PDF formula)
        self.weights = {
            'alpha': 0.10,   # Learning utility coefficient
            'beta': 0.25,    # Battery level coefficient
            'gamma': 0.20,   # Compute capacity coefficient
            'delta': 0.15,   # Reliability coefficient
            'epsilon': 0.10, # Latency coefficient (negative)
            'zeta': 0.05,    # Energy cost coefficient (negative)
            'lambda': 0.10,  # Data diversity coefficient
            'mu': 0.05       # Freshness coefficient
        }

    def set_weights(self, **kwargs):
        """
        Update fitness weights
        
        Usage:
            selector.set_weights(alpha=0.3, beta=0.2, lambda_=0.15)
        """
        valid_weights = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 
                        'zeta', 'lambda', 'mu']
        for key, value in kwargs.items():
            # Handle lambda_ → lambda conversion
            key = key.rstrip('_')
            if key in valid_weights:
                self.weights[key] = value

    def get_optimizer_params(self, active_clients):
        """
        Create optimizer parameters for SUBSET selection
        
        Returns:
            dict: Parameters for optimizer with binary encoding
        """
        return {
            "obj_func": lambda x: self.subset_fitness(x, active_clients),
            "dim": len(active_clients),      # One binary variable per client
            "lb": [0] * len(active_clients),  # Binary: 0 or 1
            "ub": [1] * len(active_clients),
            # PERF: selecting k~3-16 clients out of a pool of ~32 has a
            # small combinatorial space (e.g. C(32,3) ~ 5000). A 20x50
            # budget (1000 evals/round) was massive overkill and was the
            # dominant cost of the whole pipeline. 8x15 (120 evals/round)
            # is still generous for this problem size -- check the
            # convergence_curve in your rate_10 report; if best_fitness
            # plateaus well before iter 15, this can go even lower.
            "agents_nbr": 20,   # was 20
            "max_iter": 50     # was 50
        }
    
    def get_all_optimizer_configs(self):
        """Get configurations for all available optimizers"""
        return {
            # Binary/Discrete optimizers (RECOMMENDED for subset selection)
            "GA": {"method": "GA", "params": {"pc": 0.8, "pm": 0.2, "elitism": True}, 
                   "is_discrete": True},
            "SA": {"method": "SA", "params": {"initial_temp": 1000, "cooling_rate": 0.95, "min_temp": 1e-10}, 
                   "is_discrete": True},
            
            # Continuous optimizers (with threshold decoding)
            "GWO": {"method": "GWO", "params": {}, "is_discrete": False},   
            "PSO": {"method": "PSO", "params": {"w": 0.8, "c1": 1.5, "c2": 1.5}, "is_discrete": False},
            "DE": {"method": "DE", "params": {"F": 0.8, "CR": 0.9}, "is_discrete": False},
            "TSA": {"method": "TSA", "params": {"Pr_exp": 0.7, "Pr_esc": 0.2}, "is_discrete": False},
            
            # Baseline
            "RSELECT": {"method": "RSELECT", "params": {"selection_ratio": 0.3, "replacement_rate": 0.2}, 
                       "is_discrete": True},
        }
    
    def subset_fitness(self, solution, active_clients):
        """
        CORRECTED: Evaluate fitness of a CLIENT SUBSET
        
        Implements the PDF formula:
        F(S_t) = α·ū(S_t) + β·b̄(S_t) + γ·c̄(S_t) + δ·r̄(S_t) 
               - ε·ℓ̄(S_t) - ζ·ē(S_t) + λ·d̄(S_t) + μ·f̄(S_t)
        
        Where:
        - S_t is the selected subset
        - x̄(S_t) denotes the average of metric x over subset S_t
        - Positive weights (α,β,γ,δ,λ,μ): higher is better
        - Negative weights (-ε,-ζ): lower is better (costs)
        
        Args:
            solution: Binary/continuous vector indicating client selection
            active_clients: List of all available clients
            
        Returns:
            float: -F(S_t) for minimization (optimizer minimizes)
        """
        if len(solution) != len(active_clients):
            raise ValueError(f"Solution dimension {len(solution)} doesn't match clients {len(active_clients)}")
        
        # Step 1: Decode solution to get SUBSET of clients
        selected_subset = self._decode_and_repair_solution(solution, active_clients)
        
        # Validation
        if len(selected_subset) == 0:
            return float('inf')  # Worst possible fitness
        
        if len(selected_subset) != self.k:
            # Should not happen with repair, but safety check
            penalty = 1000 * abs(len(selected_subset) - self.k)
            return penalty
        
        # Step 2: Get normalized metrics for ALL clients in subset.
        # PERF: pull from the per-round cache (built once in select_clients)
        # instead of recomputing dict lookups/divisions on every one of the
        # agents_nbr * max_iter evaluations -- metrics don't change within
        # a single round's optimization.
        if self._metrics_cache is not None:
            metrics_list = [self._metrics_cache[id(client)] for client in selected_subset]
        else:
            metrics_list = [self._get_normalized_metrics(client) for client in selected_subset]
        
        # Step 3: Compute SUBSET AVERAGES (x̄(S_t) from PDF)
        # PERF: np.mean() on lists this small (k is often just 3-16 items)
        # is slower than plain Python sum/len due to per-call numpy
        # overhead (dtype checks, array construction). At tens of millions
        # of evaluations total, this adds up.
        n = len(metrics_list)
        u_bar = sum(m['learning_utility'] for m in metrics_list) / n     # ū(S_t)
        b_bar = sum(m['battery_level'] for m in metrics_list) / n        # b̄(S_t)
        c_bar = sum(m['compute_capacity'] for m in metrics_list) / n     # c̄(S_t)
        r_bar = sum(m['reliability'] for m in metrics_list) / n          # r̄(S_t)
        l_bar = sum(m['latency'] for m in metrics_list) / n              # ℓ̄(S_t)
        e_bar = sum(m['energy_cost'] for m in metrics_list) / n          # ē(S_t)
        d_bar = sum(m['data_diversity'] for m in metrics_list) / n       # d̄(S_t)
        f_bar = sum(m['freshness'] for m in metrics_list) / n            # f̄(S_t)
        
        # Step 4: Compute multi-objective fitness using PDF formula
        w = self.weights
        F_St = (
            w['alpha'] * u_bar +      # Learning utility (higher = better)
            w['beta'] * b_bar +       # Battery level (higher = better)
            w['gamma'] * c_bar +      # Compute capacity (higher = better)
            w['delta'] * r_bar +      # Reliability (higher = better)
            - w['epsilon'] * l_bar +  # Latency (LOWER = better, negative weight)
            - w['zeta'] * e_bar +     # Energy cost (LOWER = better, negative weight)
            w['lambda'] * d_bar +     # Data diversity (higher = better)
            w['mu'] * f_bar           # Freshness (higher = better)
        )
        
        # Return negative for minimization
        # Optimizer finds minimum, so we want to minimize -F(S_t) = maximize F(S_t)
        return -F_St
    
    def _decode_and_repair_solution(self, solution, clients):
        """
        Convert optimizer solution to valid client subset with EXACTLY K clients
        
        This is CRITICAL: ensures we always have exactly K clients selected
        
        Args:
            solution: Optimizer output (binary or continuous)
            clients: List of all available clients
            
        Returns:
            list: Selected subset S_t with |S_t| = K
        """
        # Convert to binary based on optimizer type
        if self.optimizer_config.get("is_discrete", False):
            # Already binary {0, 1}
            binary_solution = [int(round(x)) for x in solution]
        else:
            # Continuous [0, 1] → threshold at 0.5
            binary_solution = [1 if x >= 0.5 else 0 for x in solution]
        
        # Get indices of selected clients
        selected_indices = [i for i, val in enumerate(binary_solution) if val == 1]
        
        # REPAIR: Ensure exactly K clients
        if len(selected_indices) > self.k:
            # Too many: keep top K by solution value
            selected_indices = sorted(selected_indices, 
                                    key=lambda i: solution[i], 
                                    reverse=True)[:self.k]
        
        elif len(selected_indices) < self.k:
            # Too few: add best non-selected clients
            not_selected = [i for i in range(len(solution)) if i not in selected_indices]
            
            if not_selected:
                # Sort non-selected by solution value
                additional = sorted(not_selected, 
                                  key=lambda i: solution[i], 
                                  reverse=True)[:self.k - len(selected_indices)]
                selected_indices.extend(additional)
            else:
                # Edge case: if somehow no non-selected, randomly add
                remaining_needed = self.k - len(selected_indices)
                all_indices = list(range(len(clients)))
                additional = np.random.choice([i for i in all_indices if i not in selected_indices],
                                            size=remaining_needed, replace=False)
                selected_indices.extend(additional)
        
        # Return selected client objects
        return [clients[i] for i in selected_indices]
    
    def _get_normalized_metrics(self, client):
        """
        Extract and normalize ALL metrics for a single client
        All values normalized to [0, 1] range for fair comparison
        
        Args:
            client: Client object (either DynaEdgeSim device or FL client)
            
        Returns:
            dict: Normalized metrics in [0, 1]
        """
        # Check client type: DynaEdgeSim device or FL client
        if hasattr(client, 'cpu_capacity'):
            # DynaEdgeSim Device
            return self._get_dynaedge_metrics(client)
        else:
            # FL Client
            return self._get_fl_client_metrics(client)
    
    def _get_dynaedge_metrics(self, device):
        """Get normalized metrics from DynaEdgeSim device"""
        # Distance calculation if environment available
        if hasattr(self, 'env') and self.env is not None:
            distance = self.env.GetDistance(device)
            max_distance = np.sqrt(self.env.parameters['sWidth']**2 + 
                                  self.env.parameters['sHeight']**2)
            normalized_distance = distance / max_distance
        else:
            normalized_distance = 0.5
        
        return {
            'learning_utility': 0.5,  # Not applicable for devices
            'battery_level': device.battery_level / 100.0,  # Assume 0-100 range
            'compute_capacity': min(1.0, device.cpu_capacity / 4.0),  # Normalize by typical max
            'reliability': device.reliability,  # Already [0,1]
            'latency': min(1.0, device.latency / 100.0),  # Normalize by 100ms
            'energy_cost': normalized_distance,  # Use distance as proxy
            'data_diversity': 0.5,  # Not applicable
            'freshness': 0.5,  # Not applicable
        }
    
    def _get_fl_client_metrics(self, client):
        """Get normalized metrics from FL client"""
        # Get training metrics from client
        raw_metrics = client.get_training_metrics()
        
        return {
            'learning_utility': raw_metrics.get('learning_utility', 0.5),
            'battery_level': raw_metrics.get('battery_level', 100) / 100.0,
            'compute_capacity': raw_metrics.get('compute_capacity', 2.0) / 4.0,
            'reliability': raw_metrics.get('reliability', 0.9),
            'latency': raw_metrics.get('latency', 50) / 100.0,
            'energy_cost': raw_metrics.get('energy_cost', 0.5),
            'data_diversity': raw_metrics.get('data_diversity', 0.5),
            'freshness': raw_metrics.get('freshness', 0.5),
        }
    
    def select_clients(self, active_clients, return_curve=False):
        """
        Select optimal client SUBSET using configured optimizer
        
        Args:
            active_clients: List of all available clients
            return_curve: If True, returns convergence curve
            
        Returns:
            If return_curve=False: list of selected clients
            If return_curve=True: (metrics, selected_clients, convergence_curve, best_fitness)
        """
        # PERF: build the normalized-metrics cache once for this round,
        # keyed by object identity (client objects aren't guaranteed
        # hashable/comparable, id() always works and is stable for the
        # lifetime of this call). Covers every branch below, including the
        # early-return paths which also call _calculate_selection_metrics.
        self._metrics_cache = {id(c): self._get_normalized_metrics(c) for c in active_clients}
        try:
            return self._select_clients_impl(active_clients, return_curve)
        finally:
            # Don't let stale per-round data leak into the next round/call
            self._metrics_cache = None

    def _select_clients_impl(self, active_clients, return_curve=False):
        if len(active_clients) < self.k:
            # Not enough clients available
            print(f"Warning: Only {len(active_clients)} clients available, need {self.k}")
            selected = active_clients
            metrics = self._calculate_selection_metrics(selected)
            if return_curve:
                return metrics, selected, np.array([0]), 0.0
            return selected
        
        if len(active_clients) == self.k:
            # Exactly K clients, no optimization needed
            selected = active_clients
            metrics = self._calculate_selection_metrics(selected)
            if return_curve:
                return metrics, selected, np.array([0]), 0.0
            return selected
        
        # Setup optimizer parameters
        self.optimizer_params = self.get_optimizer_params(active_clients)
        
        # Add progress callback if available
        if hasattr(self, 'progress_callback') and self.progress_callback is not None:
            self.optimizer_params['progress_callback'] = self.progress_callback
        
        # Run optimization
        opt = optimizers(self.optimizer_params)
        optimizer_method = getattr(opt, self.optimizer_config["method"])
        
        try:
            best_solution, best_fitness, convergence_curve = optimizer_method(
                **self.optimizer_config["params"]
            )
        except Exception as e:
            print(f"Optimizer error: {e}")
            # Fallback to random selection
            selected = list(np.random.choice(active_clients, size=self.k, replace=False))
            metrics = self._calculate_selection_metrics(selected)
            if return_curve:
                return metrics, selected, np.array([0]), 0.0
            return selected
        
        # Decode solution to client subset
        selected_clients = self._decode_and_repair_solution(best_solution, active_clients)
        
        # Calculate metrics for selected subset
        metrics = self._calculate_selection_metrics(selected_clients)
        
        if return_curve:
            return metrics, selected_clients, convergence_curve, best_fitness
        return selected_clients
    
    def _calculate_selection_metrics(self, selected_clients):
        """
        Calculate aggregate metrics for the selected client subset
        Useful for tracking selection quality over rounds
        
        Args:
            selected_clients: List of selected clients
            
        Returns:
            dict: Aggregate metrics
        """
        if not selected_clients:
            return {}
        
        # Get metrics for all selected clients (from cache when available --
        # see select_clients()/_metrics_cache)
        if self._metrics_cache is not None:
            metrics_list = [self._metrics_cache[id(c)] for c in selected_clients]
        else:
            metrics_list = [self._get_normalized_metrics(c) for c in selected_clients]
        
        # Return subset averages (plain Python -- see subset_fitness for why)
        n = len(metrics_list)
        return {
            'avg_learning_utility': sum(m['learning_utility'] for m in metrics_list) / n,
            'avg_battery': sum(m['battery_level'] for m in metrics_list) / n,
            'avg_compute': sum(m['compute_capacity'] for m in metrics_list) / n,
            'avg_reliability': sum(m['reliability'] for m in metrics_list) / n,
            'avg_latency': sum(m['latency'] for m in metrics_list) / n,
            'avg_energy_cost': sum(m['energy_cost'] for m in metrics_list) / n,
            'avg_data_diversity': sum(m['data_diversity'] for m in metrics_list) / n,
            'avg_freshness': sum(m['freshness'] for m in metrics_list) / n,
            'num_selected': len(selected_clients),
            
            # Overall fitness (same as optimization objective)
            'subset_fitness': -self.subset_fitness(
                [1] * len(selected_clients),  # All selected
                selected_clients
            )
        }


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    """
    Example usage of the corrected ClientSelector
    """
    
    print("="*80)
    print("CORRECTED CLIENT SELECTOR - DEMO")
    print("="*80)
    
    # This is just a demo structure
    # In real usage, you would have actual Client objects from environement library  
    
    class MockClient:
        def __init__(self, client_id):
            self.id = client_id
            
        def get_training_metrics(self):
            return {
                'learning_utility': np.random.rand(),
                'battery_level': 50 + np.random.rand() * 50,
                'compute_capacity': 1 + np.random.rand() * 3,
                'reliability': 0.7 + np.random.rand() * 0.3,
                'latency': 20 + np.random.rand() * 60,
                'energy_cost': np.random.rand(),
                'data_diversity': np.random.rand(),
                'freshness': np.random.rand(),
            }
    
    # Create mock clients
    all_clients = [MockClient(i) for i in range(20)]
    
    # Initialize selector
    selector = ClientSelector(clients_pool=all_clients, k=10)
    
    # Configure optimizer (GA recommended for subset selection)
    configs = selector.get_all_optimizer_configs()
    selector.optimizer_config = configs["GA"]
    
    print(f"\nSelecting {selector.k} clients from {len(all_clients)} available")
    print(f"Using optimizer: {selector.optimizer_config['method']}")
    print(f"Weights: {selector.weights}")
    
    # Select clients
    metrics, selected, curve, fitness = selector.select_clients(
        all_clients, return_curve=True
    )
    
    print(f"\n✓ Selection Complete!")
    print(f"  Selected {len(selected)} clients")
    print(f"  Best fitness: {fitness:.4f}")
    print(f"\nSubset Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("="*80)
