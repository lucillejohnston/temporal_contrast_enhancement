"""Trajectory classification - habituators vs sensitizers"""
from scipy import stats
import numpy as np
import pandas as pd

def calculate_pain_trajectory(subject_data):
    """
    Calculate pain trajectory using max pain over all trials.
    
    Used by: habituators_sensitizers.py, combining_datasets.py
    """

    clean_data = subject_data.dropna(subset=['abs_max_val', 'trial_num']).sort_values('trial_num')
    
    if len(clean_data) < 3:
        return np.nan, np.nan, np.nan
    
    if clean_data['abs_max_val'].var() == 0:
        return 0.0, np.nan, np.nan
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            clean_data['trial_num'],
            clean_data['abs_max_val']
        )
        return slope, r_value, p_value
    except:
        return np.nan, np.nan, np.nan


def classify_subject(row, mean_slope=None, std_slope=None):
    """
    Classify subject as habituator/sensitizer/no_trend based on slope.
    
    Used by: habituators_sensitizers.py, combining_datasets.py
    """
    # From combining_datasets.py lines 651-660
    if pd.isna(row['slope']):
        return 'insufficient_data'
    
    # Calculate thresholds if not provided
    if mean_slope is None or std_slope is None:
        # This would need the full trajectory_df passed in
        # Or calculate inline
        pass
        
    lower_thresh = mean_slope - std_slope
    upper_thresh = mean_slope + std_slope
    
    if row['slope'] < lower_thresh:
        return 'habituator'
    elif row['slope'] > upper_thresh:
        return 'sensitizer'
    else:
        return 'no_trend'


def calculate_all_trajectories(data):
    """
    Calculate trajectories for all subjects in dataset.
    Returns DataFrame with trajectory classification.
    
    Used by: habituators_sensitizers.py, combining_datasets.py
    """
    # From combining_datasets.py lines 585-620
    subject_trajectories = []
    
    for subject, subj_df in data.groupby('subject'):
        slope, r_val, p_val = calculate_pain_trajectory(subj_df)
        
        group_label = subj_df['group_label'].iloc[0] if 'group_label' in subj_df.columns else 'unknown'
        dataset = subj_df['dataset'].iloc[0] if 'dataset' in subj_df.columns else 'unknown'
        
        subject_trajectories.append({
            'subject': subject,
            'slope': slope,
            'r_value': r_val,
            'p_value': p_val,
            'group_label': group_label,
            'dataset': dataset,
            'n_trials': len(subj_df)
        })
    
    trajectory_df = pd.DataFrame(subject_trajectories)
    trajectory_df = trajectory_df.dropna(subset=['slope'])
    
    # Classify subjects
    mean_slope = trajectory_df['slope'].mean()
    std_slope = trajectory_df['slope'].std()
    
    trajectory_df['trajectory_classification'] = trajectory_df.apply(
        lambda row: classify_subject(row, mean_slope, std_slope), axis=1
    )
    
    return trajectory_df