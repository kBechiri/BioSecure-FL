# -*- coding: utf-8 -*-
"""
class_distribution_check.py - Quick one-off script to report the actual
class distribution (label 0 vs. label 1) across the SensSmartTech
dataset, for the reproducibility block (R3-1--4).

Run this once with your real CSV_DIR / DEMOGRAPHICS_PATH (same values
you use in BioSecureV3.py) and paste the printed output back -- that's
all that's needed to finalize the class-distribution sentence.

Usage:
    python class_distribution_check.py --csv-dir /path/to/csvs --demographics-path /path/to/demographics.csv
"""

import argparse
import os
from collections import Counter

from FederatedLearningBS import SensSmartTechLoader


def main():
    CSV_DIR = os.environ.get("BIOSECURE_CSV_DIR", os.path.join("dataset", "csv"))
    DEMOGRAPHICS_PATH = os.environ.get("BIOSECURE_DEMOGRAPHICS", os.path.join("dataset", "Demographics.csv"))
    parser = argparse.ArgumentParser()
    #parser.add_argument('--csv-dir', required=True)
    #parser.add_argument('--demographics-path', required=True)
    #args = parser.parse_args()

    loader = SensSmartTechLoader(CSV_DIR, DEMOGRAPHICS_PATH)
    volunteer_data = loader.load_all_data()

    total_recordings = 0
    status_counts = Counter()
    per_subject_counts = {}

    for subj_id, subj in volunteer_data.items():
        recordings = subj.get('recordings', {})
        subj_status_counts = Counter()
        for rec_id, rec in recordings.items():
            status = rec.get('metadata', {}).get('status')
            status_counts[status] += 1
            subj_status_counts[status] += 1
            total_recordings += 1
        per_subject_counts[subj_id] = dict(subj_status_counts)

    n_abnormal = status_counts.get('A', 0)
    n_normal = total_recordings - n_abnormal

    print("=" * 60)
    print("CLASS DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"Total recordings: {total_recordings}")
    print(f"Total volunteers/clients: {len(volunteer_data)}")
    print()
    print(f"Raw 'status' value counts: {dict(status_counts)}")
    print()
    print(f"Label 1 (abnormal response, status=='A'): {n_abnormal} "
          f"({100*n_abnormal/total_recordings:.1f}%)")
    print(f"Label 0 (normal response, status!='A'):   {n_normal} "
          f"({100*n_normal/total_recordings:.1f}%)")
    print()
    print("Per-client breakdown (for checking non-IID severity):")
    n_single_class_clients = 0
    for subj_id, counts in per_subject_counts.items():
        labels_present = len(counts)
        if labels_present == 1:
            n_single_class_clients += 1
        print(f"  Client {subj_id}: {counts}")
    print()
    print(f"Clients with only ONE class present: {n_single_class_clients} / {len(volunteer_data)}")
    print("(relevant for R3-1--4: single-class clients affect the 80/20")
    print(" stratified split fallback described in Section 3.1.1)")


if __name__ == "__main__":
    main()
