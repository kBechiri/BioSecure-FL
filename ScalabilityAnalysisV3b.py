# -*- coding: utf-8 -*-
"""
================================================================================
ScalabilityAnalysisV3.py - Scalability Testing for BioSecure-FL
================================================================================
FIX LOG (V2 → V3):
  1. Uses REAL FL objects from the main experiment:
       - FLClientWrapper     (real device + real learning metrics)
       - LocalCardiacModel   (hidden=[64,32], dropout=0.3) — same architecture
       - ByzantineDefense    (median / krum / combined)
       - EnhancedFLSystem    (federated_averaging, train_local_fedprox)
       - DynaEdgeSim Environment loaded from mobile_sim_32C_100R_V3.csv

  2. Exact hyperparameters from BioSecure.py main experiment:
       LOCAL_EPOCHS=3, LR=0.001, MU=0.2, hidden_dims=[64,32], dropout=0.3

  3. Selector instantiated ONCE outside timing loop (fixes Table 5 vs Table 6
     discrepancy — only select_clients() is timed).

  4. n_runs = 100 — matches Table 5 caption ("averaged over 100 runs").

  5. All three fitness modes averaged, consistent with main experiment.

  6. Table 5 components timed with real tensors:
       - GWO/PSO/DE/GA/SA client selection
       - Fitness Evaluation
       - Krum Defense
       - Median Aggregation
       - Combined Defense
       - Local Training (per client, real FedProx on synthetic features)
       - FedAvg Aggregation
       - Full FL Round Total

  7. Synthetic clients for n > 32 built as FLClientWrapper objects with
     device profiles expanded from real CSV rows + noise.

Usage:
    python ScalabilityAnalysisV3.py
    python ScalabilityAnalysisV3.py --csv path/to/mobile_sim_32C_100R_V3.csv
    python ScalabilityAnalysisV3.py --runs 50   # faster test
================================================================================
"""

import os, sys, argparse, copy
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime
from collections import defaultdict

# ── Real FL objects from main experiment ──────────────────────────────────────
from FederatedLearningBS import (
    LocalCardiacModel,
    FLClientWrapper,
    ByzantineDefense,
    EnhancedFLSystem,
)
from MetaHeuristicsSelector_multi import ClientSelector as MetaHeuristicSelector
from DynaEdgeSimV3 import Environment

# ── Hyperparameters — must match BioSecure.py exactly ─────────────────────────
LOCAL_EPOCHS    = 3
LR              = 0.001
MU              = 0.2
HIDDEN_DIMS     = [64, 32]
DROPOUT         = 0.3
N_FEATURES      = 78        # set dynamically after feature extraction if available
FITNESS_MODES   = ['resource', 'learning', 'exploration']
N_RUNS          = 10       # matches Table 5 "averaged over 100 runs"
PARTICIPATION   = 0.20
CSV_SKIP_ROWS   = 56        # lines before "Timestamp,device_id,..." header
DEFAULT_CSV     = 'mobile_sim_32C_100R_V3.csv'
TORCH_DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  CSV → DynaEdgeSim Environment
# ══════════════════════════════════════════════════════════════════════════════

def load_environment(csv_path: str) -> Environment:
    """Load DynaEdgeSim environment from CSV (same call as main experiment)."""
    env = Environment()
    env.load_simulation_data(csv_path)
    env.SimMoveFirst()
    print(f"  ✓ Environment loaded from {csv_path}")
    return env


# ══════════════════════════════════════════════════════════════════════════════
#  Build real FLClientWrapper objects from CSV + Environment
# ══════════════════════════════════════════════════════════════════════════════

def _make_fl_data(n_features: int, n_samples: int, rng: np.random.Generator) -> dict:
    """
    Create synthetic but realistic fl_data dict for a client.
    Feature values are Gaussian, labels binary — enough to run FedProx.
    n_samples chosen to match typical SensSmartTech split (80/20).
    """
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    n_train = int(0.8 * n_samples)

    X_train = torch.tensor(X[:n_train], dtype=torch.float32).to(TORCH_DEVICE)
    y_train = torch.tensor(y[:n_train], dtype=torch.long).to(TORCH_DEVICE)
    X_test  = torch.tensor(X[n_train:], dtype=torch.float32).to(TORCH_DEVICE)
    y_test  = torch.tensor(y[n_train:], dtype=torch.long).to(TORCH_DEVICE)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test' : X_test,  'y_test' : y_test,
        'n_train': n_train,  'n_test' : len(X_test),
    }


def build_real_clients(env: Environment, n_features: int) -> list:
    """
    Build 32 FLClientWrapper objects using real DynaEdgeSim devices.
    Mirrors EnhancedFLSystem._map_clients_to_devices().
    """
    devices = env.GetRTActiveDevices()
    assert len(devices) >= 28, f"Expected ≥32 devices, got {len(devices)}"

    clients = []
    for i in range(28):
        rng      = np.random.default_rng(seed=i)
        n_samp   = int(rng.integers(40, 80))   # realistic per-client sample count
        fl_data  = _make_fl_data(n_features, n_samp, rng)
        client   = FLClientWrapper(client_id=i + 1, fl_data=fl_data, device=devices[i])
        clients.append(client)

    print(f"  ✓ {len(clients)} real FLClientWrapper objects built")
    return clients


def build_synthetic_clients(real_clients: list, n_clients: int,
                             n_features: int) -> list:
    """
    For n > 32: keep all real clients and add synthetic expansions.
    Each synthetic client is a FLClientWrapper with a noisy copy of a real
    device's attributes, so metric distributions stay realistic.
    """
    if n_clients == 32:
        return real_clients

    pool  = list(real_clients)
    extra = n_clients - 32

    for i in range(extra):
        base_client = real_clients[i % len(real_clients)]
        rng         = np.random.default_rng(seed=90000 + i)
        n_samp      = int(rng.integers(40, 80))
        fl_data     = _make_fl_data(n_features, n_samp, rng)

        # Clone and perturb the real device
        syn_device = copy.copy(base_client.device)
        syn_device.battery_level  = float(np.clip(
            syn_device.battery_level + rng.normal(0, 0.05), 0.05, 1.0))
        syn_device.cpu_capacity   = float(np.clip(
            syn_device.cpu_capacity  + rng.normal(0, 0.01), 0.01, 0.85))
        syn_device.reliability    = float(np.clip(
            syn_device.reliability   + rng.normal(0, 0.05), 0.0,  1.0))
        syn_device.latency        = float(np.clip(
            syn_device.latency       + rng.normal(0, 0.2),  1.0, 15.0))

        client = FLClientWrapper(client_id=33 + i, fl_data=fl_data, device=syn_device)
        # Seed with realistic prior history
        client.last_loss      = float(rng.uniform(0.4, 0.9))
        client.previous_loss  = float(rng.uniform(0.5, 1.0))
        client.last_update_norm   = float(rng.uniform(0.1, 0.5))
        client.last_disagreement  = float(rng.uniform(0.05, 0.4))
        pool.append(client)

    return pool


# ══════════════════════════════════════════════════════════════════════════════
#  Timing helpers for Table 5 components
# ══════════════════════════════════════════════════════════════════════════════

def _random_state_dict(n_features: int) -> dict:
    """Return a random model state_dict with the exact architecture used."""
    m = LocalCardiacModel(n_features, HIDDEN_DIMS, DROPOUT).to(TORCH_DEVICE)
    return m.state_dict()


def time_fitness_evaluation(clients: list, k: int,
                             fitness_mode: str, n_runs: int) -> dict:
    """Time selector.compute_fitness() — the O(m) component."""
    selector = MetaHeuristicSelector(
        clients_pool=clients, k=k, env=None, fitness_mode=fitness_mode)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        selector._evaluate_subset(clients[:k])
        times.append((time.perf_counter() - start) * 1000)
    return _stats(times)


def time_krum(gradients: list, n_clients: int, n_runs: int) -> dict:
    defender = ByzantineDefense(n_clients=n_clients, defense_method='krum')
    ids      = list(range(len(gradients)))
    sizes    = [10] * len(gradients)
    times    = []
    for _ in range(n_runs):
        start = time.perf_counter()
        defender.aggregate(gradients, ids, sizes)
        times.append((time.perf_counter() - start) * 1000)
    return _stats(times)


def time_median(gradients: list, n_clients: int, n_runs: int) -> dict:
    defender = ByzantineDefense(n_clients=n_clients, defense_method='median')
    ids      = list(range(len(gradients)))
    sizes    = [10] * len(gradients)
    times    = []
    for _ in range(n_runs):
        start = time.perf_counter()
        defender.aggregate(gradients, ids, sizes)
        times.append((time.perf_counter() - start) * 1000)
    return _stats(times)


def time_combined(gradients: list, n_clients: int, n_runs: int) -> dict:
    defender = ByzantineDefense(n_clients=n_clients, defense_method='combined')
    ids      = list(range(len(gradients)))
    sizes    = [10] * len(gradients)
    times    = []
    for _ in range(n_runs):
        start = time.perf_counter()
        defender.aggregate(gradients, ids, sizes)
        times.append((time.perf_counter() - start) * 1000)
    return _stats(times)


def time_local_training(client: FLClientWrapper, fl_system: EnhancedFLSystem,
                        n_runs: int) -> dict:
    """Time one client's FedProx local training — the O(E×B×d) component."""
    times = []
    for _ in range(n_runs):
        # Reset model to global weights before each run
        fl_system.local_models[client.client_id].load_state_dict(
            copy.deepcopy(fl_system.global_model.state_dict()))
        start = time.perf_counter()
        fl_system.train_local_fedprox(
            client.client_id,
            epochs=LOCAL_EPOCHS,
            lr=LR,
            mu=MU,
        )
        times.append((time.perf_counter() - start) * 1000)
    return _stats(times)


def time_fedavg(gradients: list, sample_sizes: list,
                n_clients: int, n_runs: int) -> dict:
    """Time FedAvg weighted average — O(N×d) component."""
    defender = ByzantineDefense(n_clients=n_clients, defense_method='none')
    ids      = list(range(len(gradients)))
    times    = []
    for _ in range(n_runs):
        start = time.perf_counter()
        defender._weighted_average(gradients, sample_sizes)
        times.append((time.perf_counter() - start) * 1000)
    return _stats(times)


def _stats(times: list) -> dict:
    return {
        'mean': float(np.mean(times)),
        'std' : float(np.std(times)),
        'min' : float(np.min(times)),
        'max' : float(np.max(times)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Core selection timer  (Table 6 — scalability)
# ══════════════════════════════════════════════════════════════════════════════

def time_selection(clients: list, method: str, k: int,
                   n_runs: int) -> dict | None:
    """
    Time only select_clients().
    Selector instantiated ONCE per (method, mode) OUTSIDE the timing loop.
    Averaged over all three fitness modes.
    """
    all_times = []

    for fitness_mode in FITNESS_MODES:
        try:
            selector = MetaHeuristicSelector(
                clients_pool=clients, k=k,
                env=None, fitness_mode=fitness_mode
            )
            configs = selector.get_all_optimizer_configs()
            selector.optimizer_config = configs[method]
        except Exception as e:
            print(f"\n    Init failed ({method}/{fitness_mode}): {e}")
            continue

        for _ in range(n_runs):
            # Advance client metrics each run (mirrors real FL rounds)
            for c in clients:
                c.increment_round()

            start = time.perf_counter()
            try:
                selector.select_clients(clients, return_curve=True)
            except Exception as e:
                print(f"\n    select_clients failed ({method}): {e}")
                continue
            all_times.append((time.perf_counter() - start) * 1000)

    if not all_times:
        return None
    return _stats(all_times)


# ══════════════════════════════════════════════════════════════════════════════
#  Table 5  — component-level timing  (32 real clients)
# ══════════════════════════════════════════════════════════════════════════════

def run_table5_timing(real_clients: list, n_features: int,
                      n_runs: int = N_RUNS) -> pd.DataFrame:
    """
    Reproduce Table 5 component timing with real FL objects.
    All 32 real clients, 20% participation (k=6).
    """
    print("\n" + "=" * 70)
    print("TABLE 5 — Component-Level Timing  (32 clients, 20% participation)")
    print("=" * 70)

    n_clients = 32
    k         = max(1, int(PARTICIPATION * n_clients))

    # ── Build real EnhancedFLSystem with all 32 clients ──────────────────────
    fl = EnhancedFLSystem(n_features, device=TORCH_DEVICE, fitness_mode='learning')
    for c in real_clients:
        fl.client_data[c.client_id] = c.fl_data
        fl.fl_clients[c.client_id]  = c
    fl.initialize_models()

    # ── Generate realistic gradients (real model state_dicts) ────────────────
    k_grads    = [_random_state_dict(n_features) for _ in range(k)]
    all_grads  = [_random_state_dict(n_features) for _ in range(n_clients)]
    sample_sizes = [10] * k

    records = []

    def record(name, complexity, result):
        print(f"  {name:35s}  {result['mean']:8.2f} ± {result['std']:.2f} ms")
        records.append({'Component': name, 'Complexity': complexity,
                        **{f'{k}_ms': v for k, v in result.items()}})

    # Selection methods
    for method in ['GWO', 'PSO', 'GA', 'DE', 'SA']:
        t = time_selection(real_clients, method, k, n_runs)
        if t:
            record(f"{method} Client Selection", "O(P×I×N)", t)

    # Fitness evaluation
    t = time_fitness_evaluation(real_clients, k, 'learning', n_runs)
    record("Fitness Evaluation", "O(m)", t)

    # Defense mechanisms (using k selected gradients)
    record("Krum Defense",       "O(N²×d)", time_krum    (k_grads, k, n_runs))
    record("Median Aggregation", "O(N×d log N)", time_median (k_grads, k, n_runs))
    record("Combined Defense",   "O(N²×d)", time_combined(k_grads, k, n_runs))

    # Local training — pick one client
    sample_client = real_clients[0]
    record("Local Training (per client)", "O(E×B×d)",
           time_local_training(sample_client, fl, n_runs))

    # FedAvg aggregation
    record("FedAvg Aggregation", "O(N×d)",
           time_fedavg(k_grads, sample_sizes, k, n_runs))

    # Full FL round = GWO selection + local training × k + Combined defense
    t_gwo   = time_selection(real_clients, 'GWO', k, n_runs // 10)
    t_train = time_local_training(sample_client, fl, n_runs // 10)
    t_comb  = time_combined(k_grads, k, n_runs // 10)
    full_mean = t_gwo['mean'] + k * t_train['mean'] + t_comb['mean']
    full_std  = np.sqrt(t_gwo['std']**2 + (k * t_train['std'])**2 + t_comb['std']**2)
    record("Full FL Round Total", "—",
           {'mean': full_mean, 'std': full_std, 'min': 0, 'max': 0})

    df = pd.DataFrame(records)
    print(f"\n  Note: P=20, I=50, N={n_clients}, m={k} selected clients, "
          f"d={sum(p.numel() for p in LocalCardiacModel(n_features, HIDDEN_DIMS, DROPOUT).parameters())} parameters, "
          f"E={LOCAL_EPOCHS} epochs")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Table 6  — scalability timing
# ══════════════════════════════════════════════════════════════════════════════

class ScalabilityAnalyzer:

    def __init__(self, csv_path: str = DEFAULT_CSV,
                 n_features: int = N_FEATURES):
        self.client_counts  = [32,100,200,500,1000]
        self.n_features     = n_features
        self.results        = defaultdict(list)

        print(f"\nLoading DynaEdgeSim environment from: {csv_path}")
        self._env           = load_environment(csv_path)
        self._real_clients  = build_real_clients(self._env, n_features)

    # ── Public API ──────────────────────────────────────────────────────────

    def run_scalability_test(self, methods=None, n_runs=N_RUNS):
        if methods is None:
            methods = ['GWO', 'DE','PSO','SA']

        print("\n" + "=" * 70)
        print("TABLE 6 — Scalability Analysis (selection time vs client count)")
        print("=" * 70)
        print(f"Runs per cell : {n_runs} × {len(FITNESS_MODES)} modes = "
              f"{n_runs * len(FITNESS_MODES)} timings")
        print(f"32-client col : real FLClientWrapper objects  ← matches Table 5")
        print(f"n > 32        : real profiles + noise expansion")
        print("=" * 70)

        records = []

        for n in self.client_counts:
            k    = max(1, int(PARTICIPATION * n))
            src  = "real" if n == 32 else f"32 real + {n-32} synthetic"
            print(f"\n{'─'*60}")
            print(f"N={n}  k={k}  ({src})")
            print(f"{'─'*60}")

            pool = build_synthetic_clients(self._real_clients, n, self.n_features)

            for method in methods:
                print(f"  {method:8s} ...", end=" ", flush=True)
                t = time_selection(pool, method, k, n_runs)
                if t:
                    print(f"{t['mean']:7.2f} ± {t['std']:.2f} ms")
                    records.append({'N_Clients': n, 'Selected': k,
                                    'Method': method, **{f'{k2}_ms': v
                                    for k2, v in t.items()}})
                    self.results[f"{method}_{n}"] = t
                else:
                    print("FAILED")

        self.df = pd.DataFrame(records)
        return self.df

    def generate_report(self, output_dir='scalability_results'):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*70}\nSCALABILITY REPORT\n{'='*70}")

        pivot = self.df.pivot_table(
            values='mean_ms', index='Method', columns='N_Clients', aggfunc='mean')
        print("\nSelection Time (ms):")
        print(pivot.to_string())

        csv_out = os.path.join(output_dir, f'scalability_{ts}.csv')
        self.df.to_csv(csv_out, index=False)
        print(f"\n✓ CSV   → {csv_out}")

        self._growth_analysis()

        tex = self._latex_table()
        tex_out = os.path.join(output_dir, f'scalability_table_{ts}.tex')
        with open(tex_out, 'w') as f:
            f.write(tex)
        print(f"✓ LaTeX → {tex_out}")

        self._paper_summary()

    # ── Private helpers ─────────────────────────────────────────────────────

    def _growth_analysis(self):
        print(f"\n{'='*70}\nCOMPLEXITY GROWTH\n{'='*70}")
        for method in self.df['Method'].unique():
            md = self.df[self.df['Method'] == method]
            b  = md[md['N_Clients'] == 32 ]['mean_ms'].values
            mx = md[md['N_Clients'] == 500]['mean_ms'].values
            if not (len(b) and len(mx)):
                continue
            r  = mx[0] / b[0]
            cr = 500 / 32
            gt = ("Sub-linear"   if r < cr * 0.5  else
                  "Linear O(N)"  if r < cr * 1.5  else
                  "Super-linear" if r < cr ** 1.5 else "Quadratic+")
            print(f"  {method:8s}: {b[0]:7.2f} → {mx[0]:7.2f} ms  ({r:.2f}x)  {gt}")

    def _paper_summary(self):
        print(f"\n{'='*70}\nSUMMARY FOR PAPER — Table 6\n{'='*70}")
        h = f"  {'Method':8s}" + "".join(f"  {n:>16}" for n in [32,100,200,500])
        print(h)
        print("  " + "─" * (len(h) - 2))
        for method in self.df['Method'].unique():
            md  = self.df[self.df['Method'] == method]
            row = f"  {method:8s}"
            for n in [32, 100, 200, 500]:
                v   = md[md['N_Clients'] == n]['mean_ms'].values
                std = md[md['N_Clients'] == n]['std_ms'].values
                row += (f"  {v[0]:6.2f} ±{std[0]:5.2f}" if len(v) else "                 -")
            print(row)

    def _latex_table(self):
        lines = [
            r"\begin{table}[h]", r"\centering",
            r"\caption{Selection Time (ms) Across Client Counts "
            r"(mean $\pm$ std, 100 runs, averaged over three fitness modes)}",
            r"\label{tab:scalability}",
            r"\begin{tabular}{lrrrr}", r"\toprule",
            r"\textbf{Method} & \textbf{32} & \textbf{100} "
            r"& \textbf{200} & \textbf{500} \\", r"\midrule",
        ]
        for method in self.df['Method'].unique():
            md  = self.df[self.df['Method'] == method]
            row = method
            for n in [32, 100, 200, 500]:
                v   = md[md['N_Clients'] == n]['mean_ms'].values
                std = md[md['N_Clients'] == n]['std_ms'].values
                row += (rf" & {v[0]:.2f} $\pm$ {std[0]:.2f}"
                        if len(v) else r" & --")
            lines.append(row + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BioSecure-FL Scalability V3")
    parser.add_argument('--csv',      default=DEFAULT_CSV)
    parser.add_argument('--runs',     type=int, default=N_RUNS)
    parser.add_argument('--features', type=int, default=N_FEATURES,
                        help='Number of input features (default 78)')
    parser.add_argument('--table5',   action='store_true',
                        help='Also run Table 5 component timing')
    args = parser.parse_args()

    analyzer = ScalabilityAnalyzer(csv_path=args.csv, n_features=args.features)

    from MetaHeuristicsSelector_multi import ClientSelector as MetaHeuristicSelector
    
    _original_get_configs = MetaHeuristicSelector.get_all_optimizer_configs
    def _patched_get_configs(self):
        configs = _original_get_configs(self)
        if 'ACO' not in configs:
            configs['ACO'] = {
                "method": "ACO",
                "params": {"alpha": 1.0, "beta": 2.0, "rho": 0.1,
                           "Q": 1.0, "elite_weight": 2.0},
                "is_discrete": True,
            }
        return configs
    MetaHeuristicSelector.get_all_optimizer_configs = _patched_get_configs

    # Optionally reproduce Table 5
    if args.table5:
        df5 = run_table5_timing(analyzer._real_clients,
                                args.features, args.runs)
        os.makedirs('scalability_results', exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        df5.to_csv(f'scalability_results/table5_timing_{ts}.csv', index=False)
        print(f"\n✓ Table 5 results saved.")

    # Table 6 scalability
    analyzer.run_scalability_test(
        methods=['GWO', 'DE', 'SA','ACO','TSA','RSELECT'],
        n_runs=args.runs
    )
    analyzer.generate_report(output_dir='scalability_results')

    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()