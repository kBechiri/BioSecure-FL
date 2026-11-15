# BioSecure-FL (SecureFL-HeartGuard)

Code for the paper *SecureFL-HeartGuard: Bio-Inspired Client Selection for
Byzantine-Resilient Federated Learning in IoMT Wearable Cardiac Monitoring*.

## Requirements

Python 3.10 or later.

    pip install -r requirements.txt

## Dataset

The experiments use the SensSmartTech database (PhysioNet):
https://doi.org/10.13026/fy9p-n277

The dataset is not included in this repository. Download it and place the
signal CSV files in `dataset/csv/` and the demographics file at
`dataset/Demographics.csv` (the default locations). To use another location,
set two environment variables before running:

    BIOSECURE_CSV_DIR=/path/to/csv
    BIOSECURE_DEMOGRAPHICS=/path/to/Demographics.csv

## Running the experiments

    python BioSecureV3.py

Two switches near the end of `BioSecureV3.py` choose what is run:

- `run_experiment`: client-selection comparison across participation rates,
  followed by the FedMCCS baseline.
- `run_byzantine`: Byzantine-resilience experiment (attack types, Byzantine
  fractions, defense methods).

Results (CSV files and figures) are written under `Results_revision1/`.

## Files

Main pipeline

| File | Role |
|---|---|
| `BioSecureV3.py` | Main program: experiment configuration and orchestration |
| `FederatedLearningBS.py` | Data loading, feature extraction, models, FedProx training, Byzantine defenses and attack simulator |
| `MetaHeuristicsSelector_multi.py` | Client-selection fitness function and selector |
| `OptimizersV2.py` | Metaheuristic optimizers |
| `ResultBS.py` | Experiment runner, statistics and plotting |
| `fedmccs_selection.py`, `fedmccs_experiment.py`, `fedmccs_plots.py` | FedMCCS baseline, adapted to this framework |

Edge-device simulation

| File | Role |
|---|---|
| `DynaEdgeSimV3.py` | DynaEdgeSim edge-device simulator |
| `Generate_CineticV3.py` | Script that generates the device-mobility trajectory |
| `mobile_sim_32C_100R_V3.csv` | Trajectory file used in the experiments (32 devices, 100 rounds) |

Analysis scripts

| File | Role |
|---|---|
| `friedman_wilcoxon_test.py` | Friedman and Wilcoxon (Holm) tests, critical-difference diagram |
| `pareto_plot_with_fedmccs.py` | Accuracy versus energy Pareto plot including FedMCCS |
| `weight_sensitivity_ablation.py` | Fitness-weight sensitivity study (±20%) |
| `weight_ablation_study.py` | Fitness-weight ablation (one weight set to zero) |
| `fedprox_mu_sensitivity.py` | FedProx μ sensitivity study |
| `client_selection_distribution.py` | Per-client selection counts and Jain's fairness index |
| `client_heterogeneity_analysis.py` | Label-distribution EMD and feature-space divergence across clients |
| `class_distribution_check.py` | Class distribution of the dataset |
| `ScalabilityAnalysisV3b.py` | Selection-time scalability and component timing (`--table5`) |

## License and contact

Free for educational and non-commercial use; see `LICENSE`.
Contact: Karim Bechiri, karim.bechiri@university-constantine2.dz
