# -*- coding: utf-8 -*-
"""
fedmccs_selection.py - FedMCCS baseline, adapted for SecureFL-HeartGuard

Reference:
    AbdulRahman, S., Tout, H., Mourad, A., & Talhi, C. (2021).
    FedMCCS: Multicriteria Client Selection Model for Optimal IoT
    Federated Learning. IEEE Internet of Things Journal, 8(6), 4723-4735.
    https://doi.org/10.1109/JIOT.2020.3028742

FedMCCS is a GREEDY, PREDICTION-DRIVEN baseline -- fundamentally
different in character from the metaheuristics in Optimizers.py. It does
not search a fitness landscape; it (1) ranks clients by a priority
signal, then (2) walks down that ranking adding clients as long as their
PREDICTED next-round resource usage stays within a budget, stopping at a
target count or when no more eligible clients remain.

Design choice (per revision plan, Option A): FedMCCS is run ONCE PER
PARTICIPATION RATE, independent of the Resource/Learning/Exploration
fitness-mode axis used elsewhere in the paper. It has no native fitness
mode of its own -- forcing it into that 3-mode grid would misrepresent
the original algorithm. Report it as a standalone baseline row in the
participation-rate comparison (Table 4-style), with a footnote pointing
out that its resource-budget philosophy is closest in spirit to the
Resource fitness mode, for readers who want a targeted comparison.

--------------------------------------------------------------------------
STATED ADAPTATIONS (report these explicitly in the methods section)
--------------------------------------------------------------------------
1. CPU and Memory, tracked separately in the original paper, are folded
   into the single `compute_capacity` metric already used elsewhere in
   this codebase -- this framework does not track them independently.
2. The original budget check is "predicted utilization < max budget."
   This codebase's metrics are capacity/goodness-scored (higher =
   better), not raw utilization, so the check is reframed as "predicted
   available capacity >= minimum budget" for compute/battery-style
   metrics, and "predicted cost <= maximum budget" for latency/energy
   -style metrics. The exclusion logic (drop clients predicted to be
   insufficiently resourced) is preserved; only the direction of the
   inequality is adapted to match this codebase's metric semantics.
3. RUPred-LR (the paper's linear-regression resource predictor) is
   originally fit against GROWING IoT DATASET SIZE. Clients in this
   study have fixed local dataset sizes across rounds, which would make
   that regression degenerate (zero variance in x). This implementation
   instead regresses each metric against ROUND INDEX, capturing
   resource drift over time (e.g. battery depletion, thermal
   throttling) rather than drift with data volume. This substitution
   should be stated explicitly wherever FedMCCS results are reported.
4. The original paper prioritizes clients by "event rate" (class-
   imbalance representation in each client's local data), a signal tied
   to their intrusion-detection use case. No equivalent label exists in
   this framework's per-client metrics. `learning_utility` is used as
   the substitute priority signal (closest available proxy for "how
   much would this client's participation help the global model"),
   exposed as a swappable parameter (`priority_metric`) rather than
   hard-coded, so the substitution is visible and easy to change.
--------------------------------------------------------------------------
"""

import math
import numpy as np


# ==============================================================================
# RUPred-LR: Resource Utilization Prediction based on Linear Regression
# Mirrors Eqs. 3, 8, 9 of AbdulRahman et al. (2021), Section V-D, exactly,
# with round index substituted for dataset size (see adaptation #3 above).
# ==============================================================================

def _rupred_lr(history_values):
    """
    Predict the next value of a per-client resource metric via simple
    linear regression over its own history, following the paper's
    RUPred-LR least-squares formulation.

    Args:
        history_values: list/array of past observed values for ONE
            metric, ONE client, ordered by round (oldest first). The
            implicit x-axis is the round index [0, 1, 2, ..., n-1].

    Returns:
        float: predicted value for the next round (round index n).
               Falls back to the last observed value if fewer than 2
               history points are available (cold start -- regression
               is undefined with < 2 points).
    """
    n = len(history_values)
    if n == 0:
        return None
    if n < 2:
        # Cold start: no meaningful trend yet, use latest observation.
        return float(history_values[-1])

    x = np.arange(n, dtype=float)
    y = np.asarray(history_values, dtype=float)

    x_bar = x.mean()
    y_bar = y.mean()

    denom = np.sum((x - x_bar) ** 2)
    if denom == 0:
        return float(y_bar)

    alpha_hat = np.sum((x - x_bar) * (y - y_bar)) / denom   # Eq. 8
    beta_hat = y_bar - alpha_hat * x_bar                     # Eq. 9

    x_next = n  # predict one round ahead
    return float(alpha_hat * x_next + beta_hat)              # Eq. 3


# ==============================================================================
# Metric extraction -- mirrors ClientSelector._get_normalized_metrics in
# MetaHeuristicsSelector_multi.py, duplicated here so this module stays
# standalone and independent of the metaheuristic selection pipeline.
# ==============================================================================

def _get_normalized_metrics(client, env=None):
    """Extract normalized [0,1] metrics for one client (device or FL client)."""
    if hasattr(client, 'cpu_capacity'):
        return _get_dynaedge_metrics(client, env)
    else:
        return _get_fl_client_metrics(client)


def _get_dynaedge_metrics(device, env=None):
    if env is not None:
        distance = env.GetDistance(device)
        max_distance = math.sqrt(env.parameters['sWidth'] ** 2 + env.parameters['sHeight'] ** 2)
        normalized_distance = distance / max_distance if max_distance > 0 else 0.5
    else:
        normalized_distance = 0.5

    return {
        'learning_utility': 0.5,
        'battery_level': device.battery_level / 100.0,
        'compute_capacity': min(1.0, device.cpu_capacity / 4.0),
        'reliability': device.reliability,
        'latency': min(1.0, device.latency / 100.0),
        'energy_cost': normalized_distance,
        'data_diversity': 0.5,
        'freshness': 0.5,
    }


def _get_fl_client_metrics(client):
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


def _client_key(client):
    """Best-effort stable identifier for a client, for history lookups."""
    return getattr(client, 'id', None) or getattr(client, 'subj_id', None) or id(client)


# ==============================================================================
# FedMCCS selector
# ==============================================================================

class FedMCCSSelector:
    """
    Greedy, prediction-driven client selection (FedMCCS baseline).

    Unlike ClientSelector in MetaHeuristicsSelector_multi.py, this class
    has NO fitness_mode and NO metaheuristic optimizer underneath -- it
    is deterministic given the same inputs (client priority ranking +
    predicted resource sufficiency).

    Args:
        k: target number of clients to select == ceil(N_active * rate).
           Unlike the metaheuristics, FedMCCS treats this as a CEILING,
           not a guarantee -- fewer clients may be returned if not
           enough clear the resource budget (see `fulfillment_rate` in
           the diagnostics dict returned by select_clients).
        env: optional DynaEdgeSim Environment, passed through to metric
            extraction for device-based clients (mirrors ClientSelector).
        priority_metric: which normalized metric to rank candidates by
            before applying the resource-budget filter (see adaptation
            #4 above). Defaults to 'learning_utility'.
        budget_compute_min: minimum predicted compute_capacity required
            to keep a client eligible (folds CPU+Memory, adaptation #1).
        budget_energy_max: maximum predicted energy_cost allowed.
        budget_latency_max: maximum predicted latency allowed.
        history_window: number of past rounds of history to keep per
            client for the RUPred-LR regression (older entries dropped).
    """

    def __init__(self, k, env=None, priority_metric='learning_utility',
                 budget_compute_min=0.3, budget_energy_max=0.8,
                 budget_latency_max=0.8, history_window=10):
        self.k = k
        self.env = env
        self.priority_metric = priority_metric
        self.budget_compute_min = budget_compute_min
        self.budget_energy_max = budget_energy_max
        self.budget_latency_max = budget_latency_max
        self.history_window = history_window

    def select_clients(self, active_clients, client_history=None, return_diagnostics=False):
        """
        Args:
            active_clients: list of client objects available this round.
            client_history: optional dict {client_key: {metric_name: [past values]}}.
                If None (or a client has no entries yet), falls back to
                using the client's CURRENT-round metrics directly as the
                "prediction" (equivalent to RUPred-LR's cold-start case).
            return_diagnostics: if True, also return a dict with
                target_k, selected_count, fulfillment_rate, and the
                number of candidates skipped for insufficient resources.

        Returns:
            list of selected clients (length <= self.k), or
            (selected_clients, diagnostics) if return_diagnostics=True.
        """
        if len(active_clients) == 0:
            selected = []
            if return_diagnostics:
                return selected, self._diagnostics(0, 0)
            return selected

        client_history = client_history or {}

        # --- Step 1: current-round metrics + predicted resource state ---
        candidates = []
        for client in active_clients:
            current = _get_normalized_metrics(client, self.env)
            key = _client_key(client)
            hist = client_history.get(key, {})

            predicted_compute = self._predict_or_fallback(
                hist.get('compute_capacity'), current['compute_capacity'])
            predicted_energy = self._predict_or_fallback(
                hist.get('energy_cost'), current['energy_cost'])
            predicted_latency = self._predict_or_fallback(
                hist.get('latency'), current['latency'])

            candidates.append({
                'client': client,
                'priority': current[self.priority_metric],
                'predicted_compute': predicted_compute,
                'predicted_energy': predicted_energy,
                'predicted_latency': predicted_latency,
            })

        # --- Step 2: rank by priority signal, descending (paper's event-rate-first order) ---
        candidates.sort(key=lambda c: c['priority'], reverse=True)

        # --- Step 3: greedy walk, admit while resource budget holds ---
        selected = []
        skipped_insufficient = 0
        for c in candidates:
            if len(selected) >= self.k:
                break
            if self._sufficient_resources(c):
                selected.append(c['client'])
            else:
                skipped_insufficient += 1

        if return_diagnostics:
            diag = self._diagnostics(len(selected), skipped_insufficient)
            return selected, diag
        return selected

    def _predict_or_fallback(self, history_values, current_value):
        if not history_values:
            return current_value
        recent = history_values[-self.history_window:]
        pred = _rupred_lr(recent)
        return pred if pred is not None else current_value

    def _sufficient_resources(self, candidate):
        return (
            candidate['predicted_compute'] >= self.budget_compute_min and
            candidate['predicted_energy'] <= self.budget_energy_max and
            candidate['predicted_latency'] <= self.budget_latency_max
        )

    def _diagnostics(self, selected_count, skipped_insufficient):
        fulfillment_rate = (selected_count / self.k) if self.k > 0 else 0.0
        return {
            'target_k': self.k,
            'selected_count': selected_count,
            'fulfillment_rate': fulfillment_rate,
            'skipped_insufficient_resources': skipped_insufficient,
        }


# ==============================================================================
# Convenience entry point: one call per (round, participation_rate),
# matching Option A -- run once per rate, independent of fitness_mode.
# ==============================================================================

def run_fedmccs_baseline(active_clients, participation_rate, client_history=None,
                          env=None, priority_metric='learning_utility',
                          budget_compute_min=0.3, budget_energy_max=0.8,
                          budget_latency_max=0.8):
    """
    Run FedMCCS for one round at a given participation rate.

    Args:
        active_clients: list of available client objects this round.
        participation_rate: float in (0, 1], e.g. 0.1, 0.2, 0.3, 0.5 --
            matches the `rate` values already used elsewhere in the
            experiment grid (BioSecure.py's `combinations` loop).
        client_history: optional per-client metric history (see
            FedMCCSSelector.select_clients).
        env: optional DynaEdgeSim Environment.
        priority_metric, budget_*: see FedMCCSSelector.

    Returns:
        (selected_clients, diagnostics) where diagnostics includes
        target_k, selected_count, fulfillment_rate, and
        skipped_insufficient_resources -- log fulfillment_rate per rate
        for the "selection fulfillment" analysis discussed alongside
        R1-8 (participation distribution) in the revision plan.
    """
    # NOTE: matches EnhancedFLSystem._select_clients_metaheuristic's exact
    # k formula (truncating int(), not ceiling) so FedMCCS targets the
    # SAME client count as every other method at a given participation
    # rate -- required for a fair comparison in the pipeline.
    k = max(1, int(len(active_clients) * participation_rate))
    selector = FedMCCSSelector(
        k=k, env=env, priority_metric=priority_metric,
        budget_compute_min=budget_compute_min,
        budget_energy_max=budget_energy_max,
        budget_latency_max=budget_latency_max,
    )
    return selector.select_clients(active_clients, client_history=client_history,
                                    return_diagnostics=True)


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FedMCCS BASELINE (adapted) -- DEMO")
    print("=" * 80)

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

    np.random.seed(0)
    all_clients = [MockClient(i) for i in range(32)]

    for rate in [0.10, 0.20, 0.30, 0.50]:
        selected, diag = run_fedmccs_baseline(all_clients, participation_rate=rate)
        print(f"\nRate {rate:.0%}: target_k={diag['target_k']}, "
              f"selected={diag['selected_count']}, "
              f"fulfillment={diag['fulfillment_rate']:.2f}, "
              f"skipped={diag['skipped_insufficient_resources']}")

    print("\n" + "=" * 80)
