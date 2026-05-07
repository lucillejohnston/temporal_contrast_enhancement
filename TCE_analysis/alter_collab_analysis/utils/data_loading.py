"""Data loading and combining utilities"""
import json
import pandas as pd
import sqlite3

# Constants
KNEEOA_SUBJECT_OFFSET = 1000
CLBP_SUBJECT_OFFSET = 2000

def load_dataset(dataset_name, base_path):
    """
    Load a single dataset's trial metrics and optionally time series.
    
    Parameters:
        dataset_name: 'kneeOA', 'plosONE', 'cLBP'
        base_path: Base directory path
    
    Returns:
        metrics_df: DataFrame of trial metrics
    """
    # From combining_datasets.py lines 20-50
    trial_metrics_path = f'{base_path}/{dataset_name}_trial_metrics.json'
    
    with open(trial_metrics_path, 'r') as f:
        metrics_data = json.load(f)
    
    # Convert structured metrics to flat DataFrame
    metrics_records = []
    for subject_id, trials in metrics_data.items():
        for trial_num, trial_data in trials.items():
            record = {
                'dataset': dataset_name,
                'subject': int(subject_id),
                'trial_num': int(trial_num),
                **trial_data
            }
            metrics_records.append(record)
    
    return pd.DataFrame(metrics_records)


def combine_datasets(dataset_names, base_path):
    """
    Load and combine multiple datasets with subject ID offsets.
    
    Returns:
        combined_df: Combined DataFrame with all datasets
    """
    # From combining_datasets.py lines 20-170
    # Full implementation here
    pass


def add_group_labels(trial_metrics, sql_path):
    """Add clinical group labels from SQL database."""
    # From combining_datasets.py lines 115-140
    pass