# -*- coding: utf-8 -*-
"""
================================================================================
FederatedLearningBS.py - Federated Learning Library for BioSecure
================================================================================
Contains all federated learning related classes and functions:
- Byzantine Defense mechanisms
- FL Client Wrapper
- Enhanced FL System
- Models (LocalCardiacModel, PersonalizedModel)
- Data Loading and Feature Extraction
- Multi-Method Experiment Runner
================================================================================
"""

import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    torch._dynamo.config.suppress_errors = True
except:
    pass

from collections import defaultdict
from datetime import datetime
import copy
import time

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")

# Import MetaHeuristic Selector and DynaEdgeSim
from MetaHeuristicsSelector_multi import ClientSelector as MetaHeuristicSelector
from DynaEdgeSimV3 import Environment, Device

# Import FedMCCS baseline selector (published-method comparison, R3-11)
from fedmccs_selection import FedMCCSSelector


# ==============================================================================
# FITNESS MODE DEFINITIONS
# ==============================================================================
"""
THREE FITNESS VARIANTS:

🟢 RESOURCE (conservative) - Use when energy is scarce, devices unreliable
   F = 0.30*battery + 0.25*reliability - 0.20*latency - 0.15*energy + 0.10*update_norm

🔵 LEARNING (recommended) - Use when accuracy matters, heterogeneity is high  
   F = 0.25*loss_improvement + 0.20*disagreement + 0.15*update_norm + 
       0.15*battery + 0.15*reliability - 0.10*latency

🔴 EXPLORATION (anti-bias) - Use in early rounds, strong non-IID, personalization
   F = 0.35*disagreement + 0.25*update_norm + 0.20*(1-selection_freq) + 
       0.10*battery - 0.10*latency
"""

FITNESS_MODES = {
    'resource': {
        'description': 'Resource-Aware (conservative)',
        'short_name': 'RES',
        'color': '#2ecc71',  # Green
        'weights': {
            'battery': 0.30,
            'reliability': 0.25,
            'latency': -0.20,
            'energy': -0.15,
            'update_norm': 0.10
        }
    },
    'learning': {
        'description': 'Learning-Aware (recommended)',
        'short_name': 'LRN',
        'color': '#3498db',  # Blue
        'weights': {
            'loss_improvement': 0.25,
            'disagreement': 0.20,
            'update_norm': 0.15,
            'battery': 0.15,
            'reliability': 0.15,
            'latency': -0.10
        }
    },
    'exploration': {
        'description': 'Exploration-Driven (anti-bias)',
        'short_name': 'EXP',
        'color': '#e74c3c',  # Red
        'weights': {
            'disagreement': 0.35,
            'update_norm': 0.25,
            'freshness': 0.20,
            'battery': 0.10,
            'latency': -0.10
        }
    }
}


# ==============================================================================
# BYZANTINE FAULT TOLERANCE MODULE
# ==============================================================================

class ByzantineDefense:
    """
    Byzantine-resilient aggregation with multiple defense mechanisms
    """
    def __init__(self, n_clients, defense_method='median', 
                 anomaly_threshold=2.5, trust_decay=0.9, max_byzantine_fraction=0.3):
        self.n_clients = n_clients
        self.defense_method = defense_method
        self.anomaly_threshold = anomaly_threshold
        self.trust_decay = trust_decay
        self.max_byzantine_fraction = max_byzantine_fraction
        
        self.trust_scores = {i: 1.0 for i in range(n_clients)}
        self.detection_history = []
        self.aggregation_history = []
        
    def compute_gradient_norms(self, client_gradients):
        norms = []
        for grad_dict in client_gradients:
            total_norm = 0.0
            for key, grad in grad_dict.items():
                total_norm += torch.norm(grad.float()).item() ** 2
            norms.append(np.sqrt(total_norm))
        return np.array(norms)
    
    def detect_anomalies(self, client_gradients, client_ids):
        if len(client_gradients) < 3:
            return {cid: False for cid in client_ids}, {'detected': 0, 'total': len(client_ids)}
        
        norms = self.compute_gradient_norms(client_gradients)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) + 1e-8
        
        z_scores = (norms - mean_norm) / std_norm
        
        anomalies = {}
        detected_count = 0
        
        for i, cid in enumerate(client_ids):
            is_anomalous = abs(z_scores[i]) > self.anomaly_threshold
            anomalies[cid] = is_anomalous
            if is_anomalous:
                detected_count += 1
                self.trust_scores[cid] = self.trust_decay * self.trust_scores[cid]
            else:
                self.trust_scores[cid] = self.trust_decay * self.trust_scores[cid] + (1 - self.trust_decay)
        
        metrics = {
            'detected': detected_count,
            'total': len(client_ids),
            'detection_rate': detected_count / len(client_ids),
            'z_scores': z_scores.tolist(),
            'norms': norms.tolist()
        }
        
        self.detection_history.append(metrics)
        return anomalies, metrics
    
    def coordinate_wise_median(self, client_gradients):
        aggregated = {}
        keys = list(client_gradients[0].keys())
        
        for key in keys:
            stacked = torch.stack([g[key].float() for g in client_gradients])
            aggregated[key] = torch.median(stacked, dim=0).values
        
        return aggregated
    
    def krum_selection(self, client_gradients, client_ids, m=None):
        n = len(client_gradients)
        f = int(n * self.max_byzantine_fraction)
        
        if m is None:
            m = max(1, n - f - 2)
        m = min(m, n)
        
        flat_gradients = []
        for grad_dict in client_gradients:
            flat = torch.cat([g.float().flatten() for g in grad_dict.values()])
            flat_gradients.append(flat)
        
        n_clients = len(flat_gradients)
        distances = torch.zeros(n_clients, n_clients)
        
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = torch.norm(flat_gradients[i] - flat_gradients[j]).item()
                distances[i, j] = dist
                distances[j, i] = dist
        
        scores = []
        n_neighbors = max(1, n - f - 2)
        
        for i in range(n_clients):
            sorted_dists = torch.sort(distances[i])[0]
            score = torch.sum(sorted_dists[1:n_neighbors+1]).item()
            scores.append(score)
        
        selected_indices = np.argsort(scores)[:m]
        
        aggregated = {}
        keys = list(client_gradients[0].keys())
        
        for key in keys:
            stacked = torch.stack([client_gradients[i][key].float() for i in selected_indices])
            aggregated[key] = torch.mean(stacked, dim=0)
        
        return selected_indices, aggregated
    
    def trimmed_mean(self, client_gradients, trim_fraction=0.1):
        aggregated = {}
        keys = list(client_gradients[0].keys())
        n = len(client_gradients)
        trim_count = int(n * trim_fraction)
        
        for key in keys:
            stacked = torch.stack([g[key].float() for g in client_gradients])
            
            if trim_count > 0 and n > 2 * trim_count:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[trim_count:n-trim_count]
                aggregated[key] = torch.mean(trimmed, dim=0)
            else:
                aggregated[key] = torch.mean(stacked, dim=0)
        
        return aggregated
    
    def aggregate(self, client_gradients, client_ids, sample_sizes):
        if len(client_gradients) == 0:
            return None, {'error': 'No gradients to aggregate'}
        
        anomalies, detection_metrics = self.detect_anomalies(client_gradients, client_ids)
        
        filtered_gradients = []
        filtered_ids = []
        filtered_sizes = []
        
        for i, cid in enumerate(client_ids):
            if not anomalies[cid]:
                filtered_gradients.append(client_gradients[i])
                filtered_ids.append(cid)
                filtered_sizes.append(sample_sizes[i])
        
        if len(filtered_gradients) < len(client_gradients) * 0.5:
            filtered_gradients = client_gradients
            filtered_ids = client_ids
            filtered_sizes = sample_sizes
        
        if self.defense_method == 'median':
            aggregated = self.coordinate_wise_median(filtered_gradients)
        elif self.defense_method == 'krum':
            _, aggregated = self.krum_selection(filtered_gradients, filtered_ids)
        elif self.defense_method == 'trimmed_mean':
            aggregated = self.trimmed_mean(filtered_gradients)
        elif self.defense_method == 'combined':
            if len(filtered_gradients) > 3:
                selected_idx, _ = self.krum_selection(filtered_gradients, filtered_ids)
                krum_gradients = [filtered_gradients[i] for i in selected_idx]
                aggregated = self.coordinate_wise_median(krum_gradients)
            else:
                aggregated = self.coordinate_wise_median(filtered_gradients)
        else:
            aggregated = self._weighted_average(filtered_gradients, filtered_sizes)
        
        metrics = {
            'method': self.defense_method,
            'total_clients': len(client_gradients),
            'filtered_clients': len(filtered_gradients),
            'anomalies_detected': detection_metrics['detected'],
            'detection_rate': detection_metrics['detection_rate']
        }
        
        self.aggregation_history.append(metrics)
        return aggregated, metrics
    
    def _weighted_average(self, client_gradients, sample_sizes):
        total = sum(sample_sizes)
        aggregated = {}
        keys = list(client_gradients[0].keys())
        
        for key in keys:
            aggregated[key] = torch.zeros_like(client_gradients[0][key], dtype=torch.float32)
            for i, grad in enumerate(client_gradients):
                w = sample_sizes[i] / total
                aggregated[key] += w * grad[key].float()
        
        return aggregated
    
    def get_trust_scores(self):
        return self.trust_scores.copy()
    
    def get_statistics(self):
        if not self.detection_history:
            return {}
        
        total_detected = sum(h['detected'] for h in self.detection_history)
        total_clients = sum(h['total'] for h in self.detection_history)
        
        return {
            'total_rounds': len(self.detection_history),
            'total_anomalies_detected': total_detected,
            'average_detection_rate': total_detected / total_clients if total_clients > 0 else 0,
            'trust_scores': self.trust_scores
        }


class ByzantineAttackSimulator:
    """
    Simulate various Byzantine attacks for testing defense mechanisms
    """
    def __init__(self, attack_type='none', attack_strength=1.0, target_fraction=0.2):
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        self.target_fraction = target_fraction
        self.compromised_clients = set()
        self.attack_history = []
        
    def select_byzantine_clients(self, client_ids, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        n_byzantine = int(len(client_ids) * self.target_fraction)
        self.compromised_clients = set(np.random.choice(client_ids, size=n_byzantine, replace=False))
        return self.compromised_clients
    
    def is_byzantine(self, client_id):
        return client_id in self.compromised_clients
    
    def apply_attack(self, gradient_dict, client_id):
        if not self.is_byzantine(client_id) or self.attack_type == 'none':
            return gradient_dict
        
        attacked_gradient = {}
        
        for key, grad in gradient_dict.items():
            if self.attack_type == 'random':
                attacked_gradient[key] = torch.randn_like(grad) * self.attack_strength * torch.std(grad)
            elif self.attack_type == 'sign_flip':
                attacked_gradient[key] = -grad * self.attack_strength
            elif self.attack_type == 'scaling':
                attacked_gradient[key] = grad * self.attack_strength * 100
            elif self.attack_type == 'gaussian_noise':
                noise = torch.randn_like(grad) * torch.std(grad) * self.attack_strength
                attacked_gradient[key] = grad + noise
            elif self.attack_type == 'zero':
                attacked_gradient[key] = torch.zeros_like(grad)
            else:
                attacked_gradient[key] = grad
        
        self.attack_history.append({
            'client_id': client_id,
            'attack_type': self.attack_type,
            'round': len(self.attack_history)
        })
        
        return attacked_gradient
    
    def get_attack_statistics(self):
        return {
            'attack_type': self.attack_type,
            'attack_strength': self.attack_strength,
            'target_fraction': self.target_fraction,
            'n_compromised': len(self.compromised_clients),
            'compromised_clients': list(self.compromised_clients),
            'total_attacks': len(self.attack_history)
        }


# ==============================================================================
# DATA LOADING
# ==============================================================================

class SensSmartTechLoader:
    def __init__(self, csv_dir, demographics_path):
        self.csv_dir = csv_dir
        self.demographics_path = demographics_path
        self.demographics = None
        self.volunteer_data = {}
        
    def load_demographics(self):
        print("=" * 40)
        print("LOADING DEMOGRAPHICS")
        print("=" * 40)
        
        self.demographics = pd.read_csv(self.demographics_path, skiprows=1, usecols=range(14))
        self.demographics.columns = [
            'file_number', 'subject_number', 'recording_time', 'gender',
            'age', 'height', 'weight', 'bmi', 'ecg_file', 'ppg_file', 
            'pcg_file', 'acc_file', 'status', 'heart_rate'
        ]
        
        self.demographics = self.demographics[
            pd.to_numeric(self.demographics['subject_number'], errors='coerce').notna()
        ].copy()
        
        for col in ['subject_number']:
            self.demographics[col] = self.demographics[col].astype(int)
        for col in ['age', 'height', 'weight', 'bmi', 'heart_rate']:
            self.demographics[col] = pd.to_numeric(self.demographics[col], errors='coerce')
        
        print(f"✓ Loaded {len(self.demographics)} recordings from {self.demographics['subject_number'].nunique()} subjects")
        return self.demographics
    
    def load_signal_file(self, filename):
        filepath = os.path.join(self.csv_dir, filename + '.csv')
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            return df.iloc[:, 1:].values if df.shape[1] > 1 else df.values
        except:
            return None
    
    def load_all_data(self):
        print("\n" + "=" * 40)
        print("LOADING ALL SIGNAL DATA")
        print("=" * 40)
        
        if self.demographics is None:
            self.load_demographics()
        
        recordings_loaded = 0
        
        for subject_id in self.demographics['subject_number'].unique():
            subject_recs = self.demographics[self.demographics['subject_number'] == subject_id]
            
            self.volunteer_data[subject_id] = {
                'recordings': {},
                'demographics': {
                    'age': subject_recs['age'].iloc[0],
                    'gender': subject_recs['gender'].iloc[0],
                    'bmi': subject_recs['bmi'].iloc[0]
                }
            }
            
            for _, row in subject_recs.iterrows():
                signals = {}
                for sig_type in ['ecg', 'ppg', 'acc']:
                    data = self.load_signal_file(row[f'{sig_type}_file'])
                    if data is not None:
                        signals[sig_type] = data
                
                if signals:
                    signals['metadata'] = {
                        'status': row['status'],
                        'heart_rate': row['heart_rate']
                    }
                    self.volunteer_data[subject_id]['recordings'][row['file_number']] = signals
                    recordings_loaded += 1
        
        self.volunteer_data = {k: v for k, v in self.volunteer_data.items() if len(v['recordings']) > 0}
        
        print(f"✓ Loaded {recordings_loaded} recordings from {len(self.volunteer_data)} subjects")
        return self.volunteer_data


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

class FeatureExtractor:
    def extract_statistical_features(self, sig, prefix=''):
        default = {f'{prefix}_{s}': 0.0 for s in ['mean','std','min','max','range','skew','kurtosis','rms','median','iqr']}
        if sig is None or len(sig) == 0:
            return default
        sig = sig.flatten() if len(sig.shape) > 1 else sig
        sig = np.nan_to_num(sig, nan=0.0)
        if len(sig) == 0 or np.std(sig) == 0:
            return default
        return {
            f'{prefix}_mean': float(np.mean(sig)), f'{prefix}_std': float(np.std(sig)),
            f'{prefix}_min': float(np.min(sig)), f'{prefix}_max': float(np.max(sig)),
            f'{prefix}_range': float(np.ptp(sig)), f'{prefix}_skew': float(skew(sig)),
            f'{prefix}_kurtosis': float(kurtosis(sig)), f'{prefix}_rms': float(np.sqrt(np.mean(sig**2))),
            f'{prefix}_median': float(np.median(sig)), 
            f'{prefix}_iqr': float(np.percentile(sig, 75) - np.percentile(sig, 25))
        }
    
    def extract_ecg_features(self, ecg_signal, fs=500):
        features = {}
        default_hrv = {'ecg_hr': 0.0, 'hrv_sdnn': 0.0, 'hrv_rmssd': 0.0, 'hrv_pnn50': 0.0}
        
        if ecg_signal is None or len(ecg_signal) == 0:
            features.update(self.extract_statistical_features(None, 'ecg'))
            features.update(default_hrv)
            return features
        
        ecg = ecg_signal[:, 0].flatten() if len(ecg_signal.shape) > 1 else ecg_signal.flatten()
        features.update(self.extract_statistical_features(ecg, 'ecg'))
        
        try:
            nyq = fs / 2
            b, a = signal.butter(2, [5/nyq, 15/nyq], btype='band')
            ecg_filt = signal.filtfilt(b, a, ecg)
            peaks, _ = signal.find_peaks(ecg_filt, distance=int(0.3*fs), 
                                         height=np.mean(ecg_filt)+0.5*np.std(ecg_filt))
            if len(peaks) > 2:
                rr = np.diff(peaks) / fs * 1000
                rr = rr[(rr > 300) & (rr < 2000)]
                if len(rr) > 1:
                    features['ecg_hr'] = float(60000 / np.mean(rr))
                    features['hrv_sdnn'] = float(np.std(rr))
                    features['hrv_rmssd'] = float(np.sqrt(np.mean(np.diff(rr)**2)))
                    features['hrv_pnn50'] = float(np.sum(np.abs(np.diff(rr)) > 50) / len(np.diff(rr)) * 100)
                else:
                    features.update(default_hrv)
            else:
                features.update(default_hrv)
        except:
            features.update(default_hrv)
        return features
    
    def extract_ppg_features(self, ppg_signal, fs=100):
        features = {}
        default = {'ppg_hr': 0.0, 'ppg_amplitude': 0.0, 'ppg_pulse_width': 0.0}
        
        if ppg_signal is None or len(ppg_signal) == 0:
            features.update(self.extract_statistical_features(None, 'ppg'))
            features.update(default)
            return features
        
        ppg = ppg_signal[:, 0].flatten() if len(ppg_signal.shape) > 1 else ppg_signal.flatten()
        features.update(self.extract_statistical_features(ppg, 'ppg'))
        
        try:
            peaks, _ = signal.find_peaks(ppg, distance=int(0.3*fs))
            if len(peaks) > 2:
                pp = np.diff(peaks) / fs * 1000
                pp = pp[(pp > 300) & (pp < 2000)]
                if len(pp) > 0:
                    features['ppg_hr'] = float(60000 / np.mean(pp))
                    features['ppg_amplitude'] = float(np.mean(ppg[peaks]) - np.mean(ppg))
                    features['ppg_pulse_width'] = float(np.mean(pp))
                else:
                    features.update(default)
            else:
                features.update(default)
        except:
            features.update(default)
        return features
    
    def extract_acc_features(self, acc_signal):
        features = {}
        if acc_signal is None or len(acc_signal) == 0:
            features.update(self.extract_statistical_features(None, 'acc'))
            features['acc_energy'] = 0.0
            features['acc_activity'] = 0.0
            return features
        
        if len(acc_signal.shape) > 1 and acc_signal.shape[1] >= 3:
            acc = np.sqrt(np.sum(acc_signal[:, :3]**2, axis=1))
        else:
            acc = acc_signal.flatten()
        
        features.update(self.extract_statistical_features(acc, 'acc'))
        features['acc_energy'] = float(np.sum(acc**2) / len(acc))
        features['acc_activity'] = float(np.std(acc))
        return features
    
    def extract_all_features(self, recording, demographics=None):
        features = {}
        features.update(self.extract_ecg_features(recording.get('ecg')))
        features.update(self.extract_ppg_features(recording.get('ppg')))
        features.update(self.extract_acc_features(recording.get('acc')))
        features['meta_hr'] = float(recording.get('metadata', {}).get('heart_rate', 0) or 0)
    
        if demographics is not None:
            features['demo_age'] = float(demographics.get('age', 0) or 0)
            features['demo_height'] = float(demographics.get('height', 0) or 0)
            features['demo_weight'] = float(demographics.get('weight', 0) or 0)
            features['demo_bmi'] = float(demographics.get('bmi', 0) or 0)
            g = str(demographics.get('gender', '')).strip().lower()
            features['demo_gender_male'] = 1.0 if g in ['m', 'male', '1'] else 0.0
            features['demo_gender_female'] = 1.0 if g in ['f', 'female', '0'] else 0.0
    
        return features


# ==============================================================================
# MODELS
# ==============================================================================

class LocalCardiacModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


class PersonalizedModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.personal = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 2)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.personal(shared_out)
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


# ==============================================================================
# FL CLIENT WRAPPER
# ==============================================================================

class FLClientWrapper:
    """
    Wrapper that associates an FL client with a DynaEdgeSim device
    Provides training metrics for MetaHeuristicSelector
    Supports THREE FITNESS MODES: resource, learning, exploration
    """
    def __init__(self, client_id, fl_data, device=None):
        self.id = client_id
        self.client_id = client_id
        self.fl_data = fl_data
        self.device = device
        
        self.last_loss = 1.0
        self.previous_loss = 1.0
        self.participation_count = 0
        self.total_rounds = 0
        
        self.last_update_norm = 0.0
        self.last_disagreement = 0.0
        self.loss_improvement = 0.0
        
    def get_training_metrics(self):
        n_samples = self.fl_data['n_train']
        y_train = self.fl_data['y_train'].cpu().numpy()
        
        if len(y_train) > 0:
            class_ratio = np.mean(y_train)
            data_diversity = 1 - abs(0.5 - class_ratio) * 2
        else:
            data_diversity = 0.5
        
        learning_utility = min(1.0, n_samples / 50) * (1 - min(1.0, self.last_loss / 5))
        
        if self.total_rounds > 0:
            selection_freq = self.participation_count / self.total_rounds
            freshness = 1 - selection_freq
        else:
            freshness = 1.0
            selection_freq = 0.0
        
        if self.previous_loss > 0:
            raw_improvement = (self.previous_loss - self.last_loss) / self.previous_loss
            loss_improvement = max(0, min(1, (raw_improvement + 0.5)))
        else:
            loss_improvement = 0.5
        
        if self.device is not None and hasattr(self.device, 'battery_level'):
            return {
                'learning_utility': learning_utility,
                'battery_level': self.device.battery_level * 100,
                'battery': self.device.battery_level,
                'compute_capacity': self.device.cpu_capacity * 4,
                'reliability': self.device.reliability,
                'latency': min(1.0, self.device.latency / 100.0),
                'energy_cost': self._compute_energy_cost(),
                'energy': self._compute_energy_cost(),
                'data_diversity': data_diversity,
                'freshness': freshness,
                'loss_improvement': loss_improvement,
                'disagreement': self.last_disagreement,
                'update_norm': min(1.0, self.last_update_norm),
                'selection_freq': selection_freq,
            }
        else:
            return {
                'learning_utility': learning_utility,
                'battery_level': 80 + np.random.rand() * 20,
                'battery': 0.8 + np.random.rand() * 0.2,
                'compute_capacity': 2 + np.random.rand() * 2,
                'reliability': 0.85 + np.random.rand() * 0.15,
                'latency': 0.1 + np.random.rand() * 0.4,
                'energy_cost': 0.3 + np.random.rand() * 0.4,
                'energy': 0.3 + np.random.rand() * 0.4,
                'data_diversity': data_diversity,
                'freshness': freshness,
                'loss_improvement': loss_improvement,
                'disagreement': self.last_disagreement,
                'update_norm': min(1.0, self.last_update_norm),
                'selection_freq': selection_freq,
            }
    
    def _compute_energy_cost(self):
        if self.device is None:
            return 0.5
        base_cost = self.device.idle_power + self.device.cpu_power_factor
        max_cost = 5.0
        return min(1.0, base_cost / max_cost)
    
    def update_after_training(self, loss, update_norm=0.0, disagreement=0.0):
        self.previous_loss = self.last_loss
        self.last_loss = loss
        self.last_update_norm = update_norm
        self.last_disagreement = disagreement
        self.loss_improvement = max(0, (self.previous_loss - loss) / max(self.previous_loss, 1e-6))
        self.participation_count += 1
    
    def increment_round(self):
        self.total_rounds += 1


# ==============================================================================
# ENHANCED FL SYSTEM
# ==============================================================================

class EnhancedFLSystem:
    def __init__(self, n_features, device='cpu', fitness_mode='learning',
                 byzantine_defense='none', byzantine_threshold=2.5):
        self.n_features = n_features
        self.device = device
        self.fitness_mode = fitness_mode
        self.global_model = None
        self.local_models = {}
        self.personalized_models = {}
        self.client_data = {}
        self.scaler = StandardScaler()
        self.client_alphas = {}
        
        self.global_X_train = None
        self.global_y_train = None
        self.global_X_test = None
        self.global_y_test = None
        self.client_indices = {}
        
        self.byzantine_defense_method = byzantine_defense
        self.byzantine_threshold = byzantine_threshold
        self.byzantine_defender = None
        self.attack_simulator = None

        # PERF: class_weights in train_local_fedprox only depend on a
        # client's y_train labels, which never change once client_data is
        # set -- yet they were recomputed (np.unique + loop) on every
        # single round for every selected client. Cache per client_id,
        # computed lazily on first use.
        self._class_weights_cache = {}
        
        self.training_history = {
            'global_acc': [],
            'global_loss': [],
            'selected_client_ids_per_round': [],  # R1-8 instrumentation: per-round selected client IDs, for selection-distribution/fairness analysis
            'local_loss': {},
            'selection_metrics': [],
            'energy_consumed': [],
            'avg_battery_level': [],
            'avg_reliability': [],
            'avg_latency': [],
            'clients_selected_per_round': [],
            'active_clients_per_round': [],
            'total_clients': 0,
            'unique_clients_selected': set(),
            'convergence_round': None,
            'selection_fitness': [],
            'metaheuristic_curves': [],
            'avg_loss_improvement': [],
            'avg_update_norm': [],
            'avg_disagreement': [],
            'byzantine_detections': [],
            'aggregation_metrics': [],
        }
        
        self.fl_clients = {}
        self.edge_env = None

        # FedMCCS baseline state (R3-11 published-method comparison).
        # _fedmccs_history feeds RUPred-LR's per-client regression (round
        # index vs. resource metric, see fedmccs_selection.py adaptation
        # #3); fedmccs_diagnostics_per_round logs fulfillment_rate and
        # skipped_insufficient_resources for the CSV/plot pipeline.
        self._fedmccs_history = {}
        self.fedmccs_diagnostics_per_round = []
        
    def setup_edge_simulation(self, sim_file=None, params=None):
        print("\n" + "=" * 40)
        print("SETTING UP EDGE SIMULATION")
        print("=" * 40)
        
        n_clients = len(self.client_data)
        
        if sim_file and os.path.exists(sim_file):
            self.edge_env = Environment()
            self.edge_env.load_simulation_data(sim_file)
            self.edge_env.SimMoveFirst()
            print(f"✓ Loaded simulation from {sim_file}")
        else:
            default_params = {
                'MaxTime': 200,
                'Device_count': n_clients,
                'sWidth': 100,
                'sHeight': 100,
                'sDepth': 50,
                'Battery_level_min': 0.5,
                'Battery_level_max': 1.0,
                'Reliability_min': 0.7,
                'Reliability_max': 1.0,
                'Computation_power': 3.0,
            }
            if params:
                default_params.update(params)
            
            self.edge_env = Environment(params=default_params)
            self.edge_env.create_random_devices()
            self.edge_env.run_simulation(print_infos=False)
            self.edge_env.SimMoveFirst()
            print(f"✓ Created new simulation with {n_clients} devices")
        
        self._map_clients_to_devices()
        
    def _map_clients_to_devices(self, force_remap=False):
        if self.edge_env is None:
            return
        
        if not force_remap:
            first_client = next(iter(self.fl_clients.values()), None)
            if first_client is not None and hasattr(first_client, 'device') and first_client.device is not None:
                return
            
        devices = self.edge_env.GetRTActiveDevices()
        client_ids = list(self.client_data.keys())
        
        for i, cid in enumerate(client_ids):
            if i < len(devices):
                self.fl_clients[cid].device = devices[i]
        
    def prepare_client_data(self, volunteer_data, feature_extractor, test_size=0.2, 
                            min_samples_per_client=16, augment_data=True, 
                            max_augment_ratio=2.0):
        print("\n" + "=" * 40)
        print("PREPARING DATA (OPTIMIZED)")
        print("=" * 40)
        
        all_features, all_labels = [], []
        client_indices = {}
        current_idx = 0
        
        for subj_id, data in volunteer_data.items():
            recordings = data.get('recordings', {})
            if len(recordings) < 2:
                continue
            
            subj_features, subj_labels = [], []
            for rec in recordings.values():
                demo = data.get('demographics', {})
                feat = feature_extractor.extract_all_features(rec, demographics=demo)
                subj_features.append(np.array(list(feat.values()), dtype=np.float32))
                subj_labels.append(1 if rec.get('metadata', {}).get('status') == 'A' else 0)
            
            if len(subj_features) >= 2:
                subj_features = np.nan_to_num(np.array(subj_features), nan=0.0)
                subj_labels = np.array(subj_labels, dtype=np.int64)
                
                all_features.append(subj_features)
                all_labels.append(subj_labels)
                client_indices[subj_id] = {'start': current_idx, 'end': current_idx + len(subj_features)}
                current_idx += len(subj_features)
                
                print(f"  Subject {subj_id:2d}: {len(subj_features)} samples (B={np.sum(subj_labels==0)}, A={np.sum(subj_labels==1)})")
        
        X_all = self.scaler.fit_transform(np.vstack(all_features)).astype(np.float32)
        y_all = np.concatenate(all_labels)
        self.n_features = X_all.shape[1]
        
        print(f"\n✓ Total: {len(X_all)} samples, {self.n_features} features")
        
        def augment_samples_balanced(X, y, min_target, max_ratio, noise_std=0.02):
            n_original = len(X)
            target_from_min = min_target
            target_from_ratio = int(n_original * max_ratio)
            actual_target = min(target_from_min, target_from_ratio)
            actual_target = max(actual_target, n_original)
            
            if n_original >= actual_target:
                return X, y, 0
            
            n_synthetic = actual_target - n_original
            feature_std = np.std(X, axis=0, keepdims=True)
            feature_std = np.where(feature_std == 0, 1.0, feature_std)
            
            X_aug_list = [X]
            y_aug_list = [y]
            
            unique_labels, label_counts = np.unique(y, return_counts=True)
            label_indices = {label: np.where(y == label)[0] for label in unique_labels}
            
            for i in range(n_synthetic):
                label = np.random.choice(unique_labels, p=label_counts/label_counts.sum())
                idx = np.random.choice(label_indices[label])
                noise = np.random.randn(1, X.shape[1]) * noise_std * feature_std
                new_sample = X[idx:idx+1] + noise
                X_aug_list.append(new_sample.astype(np.float32))
                y_aug_list.append(np.array([y[idx]]))
            
            return np.vstack(X_aug_list), np.concatenate(y_aug_list), n_synthetic
        
        client_data_info = {}
        total_train_samples = 0
        total_test_samples = 0
        augmentation_stats = []
        
        for subj_id, idx in client_indices.items():
            X = X_all[idx['start']:idx['end']]
            y = y_all[idx['start']:idx['end']]
            n_original = len(X)
            
            if len(X) >= 4 and len(np.unique(y)) > 1:
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
                except:
                    split = int(len(X) * 0.8)
                    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
            else:
                X_tr, X_te, y_tr, y_te = X, X, y, y
            
            n_train_original = len(X_tr)
            
            if augment_data:
                X_tr, y_tr, n_synthetic = augment_samples_balanced(
                    X_tr, y_tr, 
                    min_target=min_samples_per_client,
                    max_ratio=max_augment_ratio
                )
                if n_synthetic > 0:
                    synthetic_pct = n_synthetic / len(X_tr) * 100
                    augmentation_stats.append({
                        'client': subj_id,
                        'original': n_train_original,
                        'augmented': len(X_tr),
                        'synthetic': n_synthetic,
                        'synthetic_pct': synthetic_pct
                    })
            
            client_data_info[subj_id] = {
                'X_train': X_tr, 'y_train': y_tr,
                'X_test': X_te, 'y_test': y_te,
                'train_start': total_train_samples,
                'test_start': total_test_samples,
                'n_original': n_original
            }
            total_train_samples += len(X_tr)
            total_test_samples += len(X_te)
        
        print(f"\n→ Creating global data store...")
        print(f"  Train: {total_train_samples} samples")
        print(f"  Test:  {total_test_samples} samples")
        
        self.global_X_train = torch.zeros(total_train_samples, self.n_features, 
                                          dtype=torch.float32, device=self.device)
        self.global_y_train = torch.zeros(total_train_samples, 
                                          dtype=torch.long, device=self.device)
        self.global_X_test = torch.zeros(total_test_samples, self.n_features, 
                                         dtype=torch.float32, device=self.device)
        self.global_y_test = torch.zeros(total_test_samples, 
                                         dtype=torch.long, device=self.device)
        
        self.client_indices = {}
        
        for subj_id, info in client_data_info.items():
            train_start = info['train_start']
            train_end = train_start + len(info['X_train'])
            self.global_X_train[train_start:train_end] = torch.FloatTensor(info['X_train'])
            self.global_y_train[train_start:train_end] = torch.LongTensor(info['y_train'])
            
            test_start = info['test_start']
            test_end = test_start + len(info['X_test'])
            self.global_X_test[test_start:test_end] = torch.FloatTensor(info['X_test'])
            self.global_y_test[test_start:test_end] = torch.LongTensor(info['y_test'])
            
            self.client_indices[subj_id] = {
                'train': slice(train_start, train_end),
                'test': slice(test_start, test_end),
                'n_train': train_end - train_start,
                'n_test': test_end - test_start
            }
            
            self.client_data[subj_id] = {
                'X_train': self.global_X_train[train_start:train_end],
                'y_train': self.global_y_train[train_start:train_end],
                'X_test': self.global_X_test[test_start:test_end],
                'y_test': self.global_y_test[test_start:test_end],
                'n_train': train_end - train_start
            }
            self.client_alphas[subj_id] = 0.5
            self.fl_clients[subj_id] = FLClientWrapper(subj_id, self.client_data[subj_id])
        
        print(f"\n✓ Global data store created (zero-copy client access)")
        print(f"✓ Prepared {len(self.client_data)} clients")
        
        if augment_data and augmentation_stats:
            print(f"\n→ Augmentation summary (max_ratio={max_augment_ratio}x):")
            print(f"  {'Client':>6} | {'Original':>8} | {'Final':>8} | {'Synthetic':>9} | {'%':>5}")
            print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*5}")
            for stat in augmentation_stats:
                print(f"  {stat['client']:>6} | {stat['original']:>8} | {stat['augmented']:>8} | "
                      f"+{stat['synthetic']:>8} | {stat['synthetic_pct']:>4.0f}%")
    
    def get_client_data(self, client_id):
        if client_id not in self.client_indices:
            raise KeyError(f"Client {client_id} not found")
        
        idx = self.client_indices[client_id]
        return {
            'X_train': self.global_X_train[idx['train']],
            'y_train': self.global_y_train[idx['train']],
            'X_test': self.global_X_test[idx['test']],
            'y_test': self.global_y_test[idx['test']],
            'n_train': idx['n_train']
        }
    
    def compute_global_loss(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for data in self.client_data.values():
                outputs = self.global_model(data['X_test'])
                loss = criterion(outputs, data['y_test'])
                total_loss += loss.item() * len(data['y_test'])
                total_samples += len(data['y_test'])
        return total_loss / max(total_samples, 1)
    
    def initialize_models(self):
        print("\n" + "=" * 40)
        print("INITIALIZING MODELS")
        print("=" * 40)
        
        self.global_model = LocalCardiacModel(self.n_features).to(self.device)
        
        for cid in self.client_data:
            self.local_models[cid] = LocalCardiacModel(self.n_features).to(self.device)
            self.local_models[cid].load_state_dict(self.global_model.state_dict())
            self.personalized_models[cid] = PersonalizedModel(self.n_features).to(self.device)
        
        if isinstance(self.fitness_mode, str) and self.fitness_mode in FITNESS_MODES:
            print(f"✓ Fitness Mode: {self.fitness_mode.upper()}")
        else:
            print(f"✓ Fitness Mode: {self.fitness_mode}")
        
    def train_local_fedprox(self, client_id, epochs=5, lr=0.01, mu=0.01, 
                            sample_rate='adaptive', min_samples_per_round=8,
                            class_balance='weighted_loss'):
        model = self.local_models[client_id]
        data = self.client_data[client_id]
        n_total = len(data['X_train'])
        
        if sample_rate == 'adaptive':
            if n_total < 20:
                actual_rate = 1.0
            elif n_total < 50:
                actual_rate = 0.7
            else:
                actual_rate = 0.5
        elif isinstance(sample_rate, (int, float)):
            actual_rate = float(sample_rate)
        else:
            actual_rate = 1.0
        
        n_samples = max(min_samples_per_round, int(n_total * actual_rate))
        n_samples = min(n_samples, n_total)
        
        y_np = data['y_train'].cpu().numpy()
        unique_labels, label_counts = np.unique(y_np, return_counts=True)
        n_classes = len(unique_labels)

        # PERF: class_weights depend only on this client's y_train, which
        # never changes -- cache per client_id instead of recomputing
        # np.unique + the weighting loop on every round of every run.
        if client_id in self._class_weights_cache:
            class_weights = self._class_weights_cache[client_id].to(self.device)
        else:
            class_weights = torch.zeros(n_classes, device=self.device)
            total_samples = len(y_np)
            for i, (label, count) in enumerate(zip(unique_labels, label_counts)):
                class_weights[label] = total_samples / (n_classes * count)
            class_weights = class_weights / class_weights.sum() * n_classes
            self._class_weights_cache[client_id] = class_weights.detach().cpu()
        
        if class_balance == 'balanced_batch' or class_balance == 'oversample':
            label_indices = {label: np.where(y_np == label)[0] for label in unique_labels}
            max_count = max(label_counts)
            
            balanced_indices = []
            for label in unique_labels:
                indices = label_indices[label]
                if len(indices) < max_count:
                    oversampled = np.random.choice(indices, size=max_count, replace=True)
                    balanced_indices.extend(oversampled)
                else:
                    balanced_indices.extend(indices)
            
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            n_balanced = len(balanced_indices)
            n_samples_balanced = max(min_samples_per_round, int(n_balanced * actual_rate))
            n_samples_balanced = min(n_samples_balanced, n_balanced)
            
            selected_indices = balanced_indices[:n_samples_balanced]
            X_train_round = data['X_train'][selected_indices]
            y_train_round = data['y_train'][selected_indices]
            n_samples = len(selected_indices)
            
        elif n_samples < n_total:
            selected_indices = []
            for label in unique_labels:
                label_indices = np.where(y_np == label)[0]
                n_per_class = max(1, int(len(label_indices) * n_samples / n_total))
                n_per_class = min(n_per_class, len(label_indices))
                selected = np.random.choice(label_indices, size=n_per_class, replace=False)
                selected_indices.extend(selected)
            
            selected_indices = np.array(selected_indices)
            np.random.shuffle(selected_indices)
            
            X_train_round = data['X_train'][selected_indices]
            y_train_round = data['y_train'][selected_indices]
        else:
            X_train_round = data['X_train']
            y_train_round = data['y_train']
            n_samples = n_total
        
        global_params = {k: v.clone() for k, v in self.global_model.state_dict().items()}
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        if class_balance == 'weighted_loss':
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
        else:
            criterion = nn.CrossEntropyLoss()
        
        batch_size = min(16, max(4, n_samples // 2))

        # PERF: DataLoader's dataset/collate machinery is overhead-heavy
        # for datasets this small (often tens of rows). Manual index-based
        # batching does the same shuffle-then-slice with far less
        # per-round setup cost, since this runs for every selected client,
        # every round, every run.
        n_train = len(X_train_round)

        # PERF: also cache the global_params -> device tensors once per
        # call rather than re-transferring inside the prox penalty loop
        # (they were already .clone()'d above but .to(self.device) still
        # ran per-parameter, per-batch, per-epoch below).
        global_params_dev = {k: v.to(self.device) for k, v in global_params.items()}

        total_loss_tensor = torch.zeros((), device=self.device)
        for _ in range(epochs):
            perm = torch.randperm(n_train, device=X_train_round.device)
            for start in range(0, n_train, batch_size):
                batch_idx = perm[start:start + batch_size]
                X, y = X_train_round[batch_idx], y_train_round[batch_idx]

                optimizer.zero_grad()
                loss = criterion(model(X), y)
                prox = 0
                for name, param in model.named_parameters():
                    prox += ((param - global_params_dev[name]) ** 2).sum()
                loss += (mu / 2) * prox
                loss.backward()
                optimizer.step()
                # PERF: accumulate on-device, only sync to CPU once at the end
                # instead of calling .item() (a GPU sync point) every batch.
                total_loss_tensor += loss.detach()
        total_loss = total_loss_tensor.item()
        
        update_norm = 0.0
        final_params = model.state_dict()
        for k in initial_params:
            update_norm += torch.norm(final_params[k].float() - initial_params[k].float()).item()
        update_norm = min(1.0, update_norm / 10.0)
        
        disagreement = self._compute_disagreement(client_id)
        
        return model.state_dict(), n_samples, total_loss, update_norm, disagreement
    
    def _compute_disagreement(self, client_id):
        data = self.client_data[client_id]
        
        self.local_models[client_id].eval()
        self.global_model.eval()
        
        with torch.no_grad():
            local_probs = self.local_models[client_id].predict_proba(data['X_test'])
            global_probs = self.global_model.predict_proba(data['X_test'])
            
            kl_div = torch.sum(local_probs * (torch.log(local_probs + 1e-8) - torch.log(global_probs + 1e-8)))
            disagreement = min(1.0, kl_div.item() / 10.0)
        
        return disagreement
    
    def federated_averaging(self, client_weights, use_byzantine_defense=True):
        if len(client_weights) == 0:
            return None
        
        gradients = [w[0] for w in client_weights]
        sample_sizes = [w[1] for w in client_weights]
        client_ids = list(range(len(client_weights)))
        
        if use_byzantine_defense and self.byzantine_defender is not None and self.byzantine_defense_method != 'none':
            aggregated, metrics = self.byzantine_defender.aggregate(gradients, client_ids, sample_sizes)
            self.training_history['aggregation_metrics'].append(metrics)
            self.training_history['byzantine_detections'].append(metrics.get('anomalies_detected', 0))
            return aggregated
        else:
            total = sum(n for _, n, _, _, _ in client_weights)
            aggregated = {k: torch.zeros_like(v, dtype=torch.float32) 
                         for k, v in client_weights[0][0].items()}
            
            for state, n, _, _, _ in client_weights:
                w = n / total
                for k in aggregated:
                    aggregated[k] += w * state[k].float()
            
            return aggregated
    
    def setup_byzantine_defense(self, n_clients=None):
        if n_clients is None:
            n_clients = len(self.client_data)
        
        if self.byzantine_defense_method != 'none':
            self.byzantine_defender = ByzantineDefense(
                n_clients=n_clients,
                defense_method=self.byzantine_defense_method,
                anomaly_threshold=self.byzantine_threshold
            )
            print(f"✓ Byzantine defense enabled: {self.byzantine_defense_method} (threshold={self.byzantine_threshold})")
    
    def setup_attack_simulation(self, attack_type='none', attack_strength=1.0, target_fraction=0.2):
        self.attack_simulator = ByzantineAttackSimulator(
            attack_type=attack_type,
            attack_strength=attack_strength,
            target_fraction=target_fraction
        )
        
        if self.client_data:
            client_ids = list(self.client_data.keys())
            self.attack_simulator.select_byzantine_clients(client_ids)
            print(f"✓ Attack simulation: {attack_type} (strength={attack_strength}, {int(target_fraction*100)}% clients)")
            print(f"  Compromised clients: {self.attack_simulator.compromised_clients}")
    
    def _select_clients_metaheuristic(self, method_name, participation_rate, show_progress=False):
        client_wrappers = list(self.fl_clients.values())
        k = max(1, int(participation_rate * len(client_wrappers)))
        
        if self.edge_env is not None:
            self._map_clients_to_devices()
        
        selector = MetaHeuristicSelector(
            clients_pool=client_wrappers,
            k=k,
            env=self.edge_env,
            fitness_mode=self.fitness_mode
        )
        
        configs = selector.get_all_optimizer_configs()
        
        if method_name.upper() in configs:
            selector.optimizer_config = configs[method_name.upper()]
        else:
            selector.optimizer_config = configs.get("RSELECT", {"method": "RSELECT", "params": {}, "is_discrete": True})
        
        try:
            metrics, selected_clients, curve, fitness = selector.select_clients(
                client_wrappers, return_curve=True
            )
            
            self.training_history['selection_metrics'].append(metrics)
            if curve is not None and len(curve) > 0:
                self.training_history['metaheuristic_curves'].append(curve)
            selected_ids = [c.client_id for c in selected_clients]
            self.training_history['selected_client_ids_per_round'].append(selected_ids)  # R1-8 instrumentation
            return selected_ids, fitness
            
        except Exception as e:
            print(f"  Warning: MetaHeuristic selection failed ({e}), using random")
            selected = np.random.choice(list(self.client_data.keys()), size=k, replace=False)
            selected = list(selected)
            self.training_history['selected_client_ids_per_round'].append(selected)  # R1-8 instrumentation
            return selected, 0.0
    
    def _select_clients_fedmccs(self, participation_rate):
        """
        FedMCCS baseline selection (AbdulRahman et al., 2021 -- see
        fedmccs_selection.py for full adaptation notes). Unlike
        _select_clients_metaheuristic, this is a greedy, prediction-driven
        selector with NO fitness_mode of its own and NO guarantee of
        hitting the target k -- it may select FEWER clients if not
        enough clear the resource budget. That under-fulfillment is
        genuine FedMCCS behavior, not an error, so it is NOT padded back
        up to k here (contrast with the except-block fallback below,
        which does force exactly k since that path is a genuine error
        recovery, not FedMCCS's own selection logic).

        Uses the SAME k formula as _select_clients_metaheuristic
        (truncating int(), not ceiling) so both selection paths target
        an identical client count at a given participation rate --
        required for a fair comparison.
        """
        client_wrappers = list(self.fl_clients.values())
        k = max(1, int(participation_rate * len(client_wrappers)))

        if self.edge_env is not None:
            self._map_clients_to_devices()

        selector = FedMCCSSelector(
            k=k,
            env=self.edge_env,
            priority_metric='learning_utility',
        )

        try:
            selected_clients, diag = selector.select_clients(
                client_wrappers,
                client_history=self._fedmccs_history,
                return_diagnostics=True,
            )

            # Update per-client history with this round's metrics (all
            # active clients, not just the ones selected) so RUPred-LR
            # has a growing series of (round_index -> metric) pairs to
            # regress against next round.
            for client in client_wrappers:
                key = client.client_id
                metrics = client.get_training_metrics()
                if key not in self._fedmccs_history:
                    self._fedmccs_history[key] = {
                        'compute_capacity': [], 'energy_cost': [], 'latency': []
                    }
                self._fedmccs_history[key]['compute_capacity'].append(
                    metrics.get('compute_capacity', 2.0) / 4.0)
                self._fedmccs_history[key]['energy_cost'].append(
                    metrics.get('energy_cost', 0.5))
                self._fedmccs_history[key]['latency'].append(
                    metrics.get('latency', 0.5))

            self.fedmccs_diagnostics_per_round.append(diag)
            self.training_history['selection_metrics'].append(diag)

            selected_ids = [c.client_id for c in selected_clients]
            self.training_history['selected_client_ids_per_round'].append(selected_ids)  # R1-8 instrumentation
            # Reuse the framework's existing fitness formula (under the
            # current fitness_mode) purely for LOGGING comparability
            # against the metaheuristics -- FedMCCS did not optimize for
            # this score, it's reported alongside for reference only.
            fitness = self._compute_selection_fitness(selected_ids)

            return selected_ids, fitness

        except Exception as e:
            print(f"  Warning: FedMCCS selection failed ({e}), using random")
            selected = np.random.choice(list(self.client_data.keys()), size=k, replace=False)
            return list(selected), 0.0

    def _select_clients_random(self, participation_rate):
        k = max(1, int(participation_rate * len(self.client_data)))
        selected = list(np.random.choice(list(self.client_data.keys()), size=k, replace=False))
        fitness = self._compute_selection_fitness(selected)
        return selected, fitness
    
    def _compute_selection_fitness(self, selected_clients):
        if not selected_clients:
            return 0.0
        
        total_fitness = 0.0
        
        if isinstance(self.fitness_mode, str) and self.fitness_mode in FITNESS_MODES:
            weights = FITNESS_MODES[self.fitness_mode]['weights']
        else:
            weights = FITNESS_MODES['learning']['weights']
        
        for cid in selected_clients:
            client = self.fl_clients.get(cid)
            if client:
                metrics = client.get_training_metrics()
                fitness = 0.0
                for metric_name, weight in weights.items():
                    metric_value = metrics.get(metric_name, 0.5)
                    fitness += weight * metric_value
                total_fitness += fitness
        
        return -total_fitness / len(selected_clients)
    
    def _compute_round_energy(self, selected_clients):
        total_energy = 0.0
        battery_levels = []
        reliabilities = []
        latencies = []
        loss_improvements = []
        update_norms = []
        disagreements = []
        
        for cid in selected_clients:
            client = self.fl_clients.get(cid)
            if client is None:
                continue
                
            metrics = client.get_training_metrics()
            
            compute_energy = 1.0 - (metrics.get('compute_capacity', 2) / 4.0)
            comm_energy = metrics.get('latency', 0.5)
            battery_factor = 1.0 + (1.0 - metrics.get('battery', 0.8))
            client_energy = (compute_energy + comm_energy) * battery_factor * metrics.get('energy', 0.5)
            total_energy += client_energy
            
            battery_levels.append(metrics.get('battery_level', 80))
            reliabilities.append(metrics.get('reliability', 0.9))
            latencies.append(metrics.get('latency', 0.5) * 100)
            loss_improvements.append(metrics.get('loss_improvement', 0.5))
            update_norms.append(metrics.get('update_norm', 0.5))
            disagreements.append(metrics.get('disagreement', 0.5))
        
        return {
            'total_energy': total_energy,
            'avg_battery': np.mean(battery_levels) if battery_levels else 0,
            'avg_reliability': np.mean(reliabilities) if reliabilities else 0,
            'avg_latency': np.mean(latencies) if latencies else 0,
            'avg_loss_improvement': np.mean(loss_improvements) if loss_improvements else 0,
            'avg_update_norm': np.mean(update_norms) if update_norms else 0,
            'avg_disagreement': np.mean(disagreements) if disagreements else 0,
            'n_clients': len(selected_clients)
        }
    
    def train_federated(self, selection_method, participation_rate, n_rounds=50, 
                       local_epochs=5, lr=0.01, mu=0.01, verbose=True, show_progress=True,
                       progress_position=0, leave_progress=True):
        if verbose:
            print("\n" + "=" * 40)
            print(f"FEDERATED LEARNING (FedProx, μ={mu})")
            print(f"Selection Method: {selection_method}")
            print(f"Fitness Mode: {self.fitness_mode.upper()}")
            print("=" * 40)
        
        prev_acc = 0
        stable_rounds = 0
        convergence_threshold = 0.001
        convergence_patience = 5
        round_times = []
        
        total_clients = len(self.client_data)
        self.training_history['total_clients'] = total_clients
        
        if show_progress and TQDM_AVAILABLE:
            round_iterator = tqdm(
                range(n_rounds), 
                desc=f"    FL Rounds ({selection_method[:5]}|{self.fitness_mode[:3].upper()})", 
                unit="rnd", 
                ncols=120, 
                leave=leave_progress,
                position=progress_position,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        else:
            round_iterator = range(n_rounds)
        
        for rnd in round_iterator:
            round_start = time.time()
            
            for client in self.fl_clients.values():
                client.increment_round()
            
            for cid in self.local_models:
                self.local_models[cid].load_state_dict(self.global_model.state_dict())
            
            if self.edge_env is not None:
                active_devices = self.edge_env.GetRTActiveDevices()
                n_active = len(active_devices)
            else:
                n_active = total_clients
            
            if selection_method == 'FedMCCS':
                selected, selection_fitness = self._select_clients_fedmccs(participation_rate)
            else:
                selected, selection_fitness = self._select_clients_metaheuristic(selection_method, participation_rate)
            n_selected = len(selected)
            
            self.training_history['active_clients_per_round'].append(n_active)
            self.training_history['selection_fitness'].append(selection_fitness)
            self.training_history['unique_clients_selected'].update(selected)
            self.training_history['clients_selected_per_round'].append(n_selected)
            
            energy_metrics = self._compute_round_energy(selected)
            self.training_history['energy_consumed'].append(energy_metrics['total_energy'])
            self.training_history['avg_battery_level'].append(energy_metrics['avg_battery'])
            self.training_history['avg_reliability'].append(energy_metrics['avg_reliability'])
            self.training_history['avg_latency'].append(energy_metrics['avg_latency'])
            self.training_history['avg_loss_improvement'].append(energy_metrics['avg_loss_improvement'])
            self.training_history['avg_update_norm'].append(energy_metrics['avg_update_norm'])
            self.training_history['avg_disagreement'].append(energy_metrics['avg_disagreement'])
            
            if self.edge_env is not None:
                self.edge_env.SimMoveNext(is_FL_Round=True,Loop=True)
            
            weights = []
            decay_factor = 0.8 * (1 + np.cos(np.pi * rnd / n_rounds))
            current_lr = lr * decay_factor + 1e-5
            for cid in selected:
                w, n, loss, update_norm, disagreement = self.train_local_fedprox(cid, local_epochs, current_lr, mu)
                
                if self.attack_simulator is not None:
                    w = self.attack_simulator.apply_attack(w, cid)
                
                weights.append((w, n, loss, update_norm, disagreement))
                
                avg_loss = loss / (local_epochs * max(1, n // 16))
                self.fl_clients[cid].update_after_training(avg_loss, update_norm, disagreement)
                
                if cid not in self.training_history['local_loss']:
                    self.training_history['local_loss'][cid] = []
                self.training_history['local_loss'][cid].append(loss)
            
            aggregated = self.federated_averaging(weights, use_byzantine_defense=True)
            if aggregated is not None:
                self.global_model.load_state_dict(aggregated)
            
            acc = self.evaluate_global()
            loss = self.compute_global_loss()
            self.training_history['global_acc'].append(acc)
            self.training_history['global_loss'].append(loss)
            
            if abs(acc - prev_acc) < convergence_threshold:
                stable_rounds += 1
                if stable_rounds >= convergence_patience and self.training_history['convergence_round'] is None:
                    self.training_history['convergence_round'] = rnd + 1 - convergence_patience
            else:
                stable_rounds = 0
            prev_acc = acc
            
            round_time = time.time() - round_start
            round_times.append(round_time)
            
            if show_progress and TQDM_AVAILABLE:
                round_iterator.set_postfix({
                    'Acc': f'{acc:.3f}', 
                    'Loss': f'{loss:.3f}',
                    'Act/Sel': f'{n_active}/{n_selected}'
                }, refresh=True)
            elif verbose and ((rnd + 1) % 10 == 0 or rnd == 0):
                print(f"  Round {rnd+1:3d}/{n_rounds}: Acc={acc:.4f}, Loss={loss:.4f}, "
                      f"Active={n_active}, Selected={n_selected}, Fitness={selection_fitness:.4f}")
        
        self.training_history['total_energy'] = sum(self.training_history['energy_consumed'])
        self.training_history['client_utilization'] = len(self.training_history['unique_clients_selected']) / len(self.client_data)
        
        if verbose:
            total_train_time = sum(round_times) if round_times else 0
            print(f"✓ FL complete! ({total_train_time:.1f}s, {np.mean(round_times):.2f}s/rnd)" if round_times else "✓ FL complete!")
        
        return self.training_history
    
    def personalize_models(self, epochs=10, lr=0.005, verbose=True):
        if verbose:
            print("\n" + "=" * 40)
            print("PERSONALIZING MODELS")
            print("=" * 40)
        
        for cid, data in self.client_data.items():
            model = self.personalized_models[cid]
            global_state = self.global_model.state_dict()
            model_state = model.state_dict()
            
            for i in range(4):
                if f'network.{i}.weight' in global_state:
                    model_state[f'shared.{i}.weight'] = global_state[f'network.{i}.weight'].clone()
                if f'network.{i}.bias' in global_state:
                    model_state[f'shared.{i}.bias'] = global_state[f'network.{i}.bias'].clone()
            
            model.load_state_dict(model_state, strict=False)
            
            model.train()
            for param in model.shared.parameters():
                param.requires_grad = False
            
            optimizer = optim.Adam(model.personal.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            loader = DataLoader(TensorDataset(data['X_train'], data['y_train']),
                              batch_size=min(16, len(data['X_train'])), shuffle=True)
            
            for _ in range(epochs):
                for X, y in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    optimizer.step()
            
            for param in model.shared.parameters():
                param.requires_grad = True
        
        if verbose:
            print(f"✓ Personalized {len(self.personalized_models)} models")
    
    def learn_adaptive_alpha(self, verbose=True):
        if verbose:
            print("\n" + "=" * 40)
            print("LEARNING ADAPTIVE ALPHA")
            print("=" * 40)
        
        for cid, data in self.client_data.items():
            best_alpha, best_acc = 0.5, 0
            
            for alpha in np.arange(0.0, 1.05, 0.1):
                preds = self.predict_combined(cid, data['X_test'], alpha)
                acc = accuracy_score(data['y_test'].cpu().numpy(), 
                                   (preds['combined_weighted'] > 0.5).astype(int))
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha
            
            self.client_alphas[cid] = best_alpha
            if verbose:
                print(f"  Subject {cid:2d}: α = {best_alpha:.1f} (acc = {best_acc:.3f})")
        
        if verbose:
            print(f"\n✓ Mean α = {np.mean(list(self.client_alphas.values())):.2f}")
    
    def evaluate_global(self):
        self.global_model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for data in self.client_data.values():
                out = self.global_model(data['X_test'])
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(data['y_test'].cpu().numpy())
        return accuracy_score(labels, preds)
    
    def predict_combined(self, client_id, X, alpha=None):
        if alpha is None:
            alpha = self.client_alphas.get(client_id, 0.5)
        
        self.local_models[client_id].eval()
        self.global_model.eval()
        
        with torch.no_grad():
            local_p = self.local_models[client_id].predict_proba(X)[:, 1]
            global_p = self.global_model.predict_proba(X)[:, 1]
            combined = alpha * local_p + (1 - alpha) * global_p
        
        return {
            'local': local_p.cpu().numpy(),
            'global': global_p.cpu().numpy(),
            'combined_weighted': combined.cpu().numpy(),
            'alpha': alpha
        }
    
    def evaluate_all(self, verbose=True):
        if verbose:
            print("\n" + "=" * 40)
            print("COMPREHENSIVE EVALUATION")
            print("=" * 40)
        
        results = {}
        
        g_preds, g_labels = [], []
        with torch.no_grad():
            for data in self.client_data.values():
                out = self.global_model(data['X_test'])
                g_preds.extend(torch.argmax(out, 1).cpu().numpy())
                g_labels.extend(data['y_test'].cpu().numpy())
        
        results['global'] = {
            'accuracy': accuracy_score(g_labels, g_preds),
            'f1': f1_score(g_labels, g_preds, average='weighted', zero_division=0)
        }
        
        local_accs = []
        for cid, data in self.client_data.items():
            self.local_models[cid].eval()
            with torch.no_grad():
                preds = torch.argmax(self.local_models[cid](data['X_test']), 1).cpu().numpy()
            local_accs.append(accuracy_score(data['y_test'].cpu().numpy(), preds))
        
        results['local'] = {'mean_accuracy': np.mean(local_accs), 'std': np.std(local_accs)}
        
        c_preds, c_labels = [], []
        for cid, data in self.client_data.items():
            pred = self.predict_combined(cid, data['X_test'], alpha=0.5)
            c_preds.extend((pred['combined_weighted'] > 0.5).astype(int))
            c_labels.extend(data['y_test'].cpu().numpy())
        
        results['combined_fixed'] = {
            'accuracy': accuracy_score(c_labels, c_preds),
            'f1': f1_score(c_labels, c_preds, average='weighted', zero_division=0)
        }
        
        a_preds, a_labels = [], []
        for cid, data in self.client_data.items():
            pred = self.predict_combined(cid, data['X_test'])
            a_preds.extend((pred['combined_weighted'] > 0.5).astype(int))
            a_labels.extend(data['y_test'].cpu().numpy())
        
        results['combined_adaptive'] = {
            'accuracy': accuracy_score(a_labels, a_preds),
            'f1': f1_score(a_labels, a_preds, average='weighted', zero_division=0)
        }
        
        p_preds, p_labels = [], []
        for cid, data in self.client_data.items():
            self.personalized_models[cid].eval()
            with torch.no_grad():
                preds = torch.argmax(self.personalized_models[cid](data['X_test']), 1).cpu().numpy()
            p_preds.extend(preds)
            p_labels.extend(data['y_test'].cpu().numpy())
        
        results['personalized'] = {
            'accuracy': accuracy_score(p_labels, p_preds),
            'f1': f1_score(p_labels, p_preds, average='weighted', zero_division=0)
        }
        
        if verbose:
            print(f"\n--- Global Model ---")
            print(f"  Accuracy: {results['global']['accuracy']:.4f}, F1: {results['global']['f1']:.4f}")
            print(f"\n--- Local Models ---")
            print(f"  Mean Accuracy: {results['local']['mean_accuracy']:.4f} ± {results['local']['std']:.4f}")
            print(f"\n--- Combined (Adaptive α) ---")
            print(f"  Accuracy: {results['combined_adaptive']['accuracy']:.4f}")
            print(f"\n--- Personalized ---")
            print(f"  Accuracy: {results['personalized']['accuracy']:.4f}")
        
        return results
    
    def evaluate_all_with_predictions(self, verbose=True):
        if verbose:
            print("\n" + "=" * 40)
            print("COMPREHENSIVE EVALUATION WITH PREDICTIONS")
            print("=" * 40)
        
        results = {}
        predictions = {}
        
        g_preds, g_labels, g_probs = [], [], []
        self.global_model.eval()
        with torch.no_grad():
            for data in self.client_data.values():
                out = self.global_model(data['X_test'])
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                g_preds.extend(torch.argmax(out, 1).cpu().numpy())
                g_labels.extend(data['y_test'].cpu().numpy())
                g_probs.extend(probs)
        
        results['global'] = {
            'accuracy': accuracy_score(g_labels, g_preds),
            'f1': f1_score(g_labels, g_preds, average='weighted', zero_division=0),
            'precision': precision_score(g_labels, g_preds, average='weighted', zero_division=0),
            'recall': recall_score(g_labels, g_preds, average='weighted', zero_division=0),
        }
        predictions['global'] = {
            'y_true': np.array(g_labels),
            'y_pred': np.array(g_preds),
            'y_prob': np.array(g_probs)
        }
        
        local_accs = []
        l_preds_all, l_labels_all = [], []
        for cid, data in self.client_data.items():
            self.local_models[cid].eval()
            with torch.no_grad():
                preds = torch.argmax(self.local_models[cid](data['X_test']), 1).cpu().numpy()
            local_accs.append(accuracy_score(data['y_test'].cpu().numpy(), preds))
            l_preds_all.extend(preds)
            l_labels_all.extend(data['y_test'].cpu().numpy())
        
        results['local'] = {'mean_accuracy': np.mean(local_accs), 'std': np.std(local_accs)}
        predictions['local'] = {'y_true': np.array(l_labels_all), 'y_pred': np.array(l_preds_all)}
        
        c_preds, c_labels, c_probs = [], [], []
        for cid, data in self.client_data.items():
            pred = self.predict_combined(cid, data['X_test'], alpha=0.5)
            c_preds.extend((pred['combined_weighted'] > 0.5).astype(int))
            c_labels.extend(data['y_test'].cpu().numpy())
            c_probs.extend(pred['combined_weighted'])
        
        results['combined_fixed'] = {
            'accuracy': accuracy_score(c_labels, c_preds),
            'f1': f1_score(c_labels, c_preds, average='weighted', zero_division=0)
        }
        predictions['combined_fixed'] = {
            'y_true': np.array(c_labels),
            'y_pred': np.array(c_preds),
            'y_prob': np.array(c_probs)
        }
        
        a_preds, a_labels, a_probs = [], [], []
        for cid, data in self.client_data.items():
            pred = self.predict_combined(cid, data['X_test'])
            a_preds.extend((pred['combined_weighted'] > 0.5).astype(int))
            a_labels.extend(data['y_test'].cpu().numpy())
            a_probs.extend(pred['combined_weighted'])
        
        results['combined_adaptive'] = {
            'accuracy': accuracy_score(a_labels, a_preds),
            'f1': f1_score(a_labels, a_preds, average='weighted', zero_division=0),
            'precision': precision_score(a_labels, a_preds, average='weighted', zero_division=0),
            'recall': recall_score(a_labels, a_preds, average='weighted', zero_division=0),
        }
        predictions['combined_adaptive'] = {
            'y_true': np.array(a_labels),
            'y_pred': np.array(a_preds),
            'y_prob': np.array(a_probs)
        }
        
        p_preds, p_labels, p_probs = [], [], []
        for cid, data in self.client_data.items():
            self.personalized_models[cid].eval()
            with torch.no_grad():
                out = self.personalized_models[cid](data['X_test'])
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(out, 1).cpu().numpy()
            p_preds.extend(preds)
            p_labels.extend(data['y_test'].cpu().numpy())
            p_probs.extend(probs)
        
        results['personalized'] = {
            'accuracy': accuracy_score(p_labels, p_preds),
            'f1': f1_score(p_labels, p_preds, average='weighted', zero_division=0),
            'precision': precision_score(p_labels, p_preds, average='weighted', zero_division=0),
            'recall': recall_score(p_labels, p_preds, average='weighted', zero_division=0),
        }
        predictions['personalized'] = {
            'y_true': np.array(p_labels),
            'y_pred': np.array(p_preds),
            'y_prob': np.array(p_probs)
        }
        
        if verbose:
            print(f"\n--- Global Model ---")
            print(f"  Accuracy: {results['global']['accuracy']:.4f}")
            print(f"\n--- Personalized ---")
            print(f"  Accuracy: {results['personalized']['accuracy']:.4f}")
        
        return results, predictions
