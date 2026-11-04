"""Trial sequence analysis functions"""
import pandas as pd

def get_preceding_value(row, col, df):
    """
    Get value from preceding trial for a given metric.
    
    Used by: trial_sequences.py, combining_datasets.py
    """
    prev_trial_num = row['trial_num'] - 1
    subject = row['subject']
    prev_row = df[
        (df['subject'] == subject) & 
        (df['trial_num'] == prev_trial_num)
    ]
    if not prev_row.empty:
        return prev_row.iloc[0][col]
    return None


def add_preceding_metrics(data, metrics_dict):
    """
    Add preceding trial metrics to dataframe.
    
    Parameters:
        data: DataFrame with trial data
        metrics_dict: Dict mapping new column names to source columns
                     e.g., {'preceding_abs_max_val': 'abs_max_val'}
    
    Returns:
        DataFrame with added preceding columns
    
    Used by: trial_sequences.py, combining_datasets.py
    """
    data_copy = data.copy()
    
    for new_col, source_col in metrics_dict.items():
        data_copy[new_col] = data_copy.apply(
            lambda row: get_preceding_value(row, source_col, data), axis=1
        )
    
    return data_copy