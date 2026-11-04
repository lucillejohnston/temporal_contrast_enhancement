# -*- coding: utf-8 -*-
"""
Hold-trial habituation metrics and the habituator / sensitizer classification.

Habituator / sensitizer HERE means the slope of pain across period C of the
hold trials -- not the session trajectory (max pain regressed on trial number),
which session_trajectory.py computes and which is a different definition.

These functions moved out of
single_dataset_analysis/habituator_sensitizers_ControlTrial.py unchanged.
That file is a #%% notebook with `dataset = 'cLBP'` and a full analysis at
module level, and `from x import f` runs all of x -- so every script that
wanted these five helpers was running a whole cLBP analysis to get them, and
inheriting its failures. A module holding only function definitions imports
harmlessly, and none of these care which dataset they are given.

Used by: habituator_sensitizers_ControlTrial.py, combining_datasets.py,
combining_datasets_zscore.py, PosterFigures/FENS_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Primary metric: slope of period C of hold trials
def calculate_slope(trial_data, C_start=None, C_end=None):
    period_c_data = trial_data[(trial_data['aligned_time'] >= C_start) & (trial_data['aligned_time'] <= C_end)]
    period_c_data = period_c_data.dropna(subset=['pain']) # drop rows where pain is NaN (like in cLBP data)
    x = period_c_data['aligned_time'].to_numpy()
    y = period_c_data['pain'].to_numpy()
    if len(np.unique(x)) < 2:
        return np.nan
    slope, intercept, r, p, se = stats.linregress(x, y)
    return slope


# Sensitivity analysis 1: windowed early vs. late difference (mean of last 5 s of hold - mean of first 5s of hold)
def calculate_windowed_difference(trial_data, window_size=5, A_start=None, C_end=None):
    early_window = trial_data[(trial_data['aligned_time'] >= A_start) & (trial_data['aligned_time'] < A_start + window_size)]
    late_window = trial_data[(trial_data['aligned_time'] <= C_end) & (trial_data['aligned_time'] > C_end - window_size)]

    if len(early_window) == 0 or len(late_window) == 0:
        return np.nan

    early_mean = early_window['pain'].mean()
    late_mean = late_window['pain'].mean()

    return late_mean - early_mean


# Sensitivity analysis 2: AUC last 10s - AUC first 10s of hold
def calculate_auc_difference(trial_data, window_size=10, A_start=None, C_end=None):

    early_window = trial_data[(trial_data['aligned_time'] >= A_start) & (trial_data['aligned_time'] < A_start + window_size)]
    late_window = trial_data[(trial_data['aligned_time'] <= C_end) & (trial_data['aligned_time'] > C_end - window_size)]
    early_window = early_window.dropna(subset=['pain'])
    late_window = late_window.dropna(subset=['pain'])

    if len(early_window) == 0 or len(late_window) == 0:
        return np.nan

    early_auc = np.trapezoid(early_window['pain'], early_window['aligned_time'])
    late_auc = np.trapezoid(late_window['pain'], late_window['aligned_time'])

    return late_auc - early_auc


# Sensitivity analysis 3: normalized time-aware pain change (second extrema - first extrema) / max
def calculate_normalized_time_aware_change(row, max_floor=5):
    min_val = row.get('abs_min_val')
    max_val = row.get('abs_max_val')
    min_time = row.get('abs_min_time')
    max_time = row.get('abs_max_time')
    if pd.isna(min_val) or pd.isna(max_val) or pd.isna(min_time) or pd.isna(max_time):
        return np.nan

    if max_val <= max_floor or max_val == 0:
        return np.nan

    # first extrema in time, second extrema in time
    if min_time <= max_time:
        first_ext = min_val
        second_ext = max_val
    else:
        first_ext = max_val
        second_ext = min_val

    return (second_ext - first_ext) / max_val * 100


# Bootstrapping approach to determine classification based on any metric
def bootstrap_classification(metric_values, n_boot=10000, ci=95, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    metric_values = metric_values.dropna().values
    if len(metric_values) == 0:
        print("Warning: No valid metric values for classification. Returning 'unclassified'.")
        return 'unclassified', np.nan, np.nan, np.nan

    # Calculate observed mean
    observed_mean = np.mean(metric_values)

    # Generate bootstrap samples and calculate means
    bootstrap_means = []
    for _ in range(n_boot):
        sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
        bootstrap_means.append(np.mean(sample))

    # Calculate confidence interval
    lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

    # Classify based on whether CI is above or below zero
    if lower_bound > 0:
        classification = 'sensitizer'
    elif upper_bound < 0:
        classification = 'habituator'
    else:
        classification = 'no trend'
    print(".")
    return classification, observed_mean, lower_bound, upper_bound


def plot_hold_trials_for_subject(subject_id, hold_metrics_df, time_series_df, title=None,
                                 time_window=None, savepath=None, ax1=None):
    subject_hold_trials = hold_metrics_df[hold_metrics_df['subject'] == subject_id].copy()
    if subject_hold_trials.empty:
        print(f"No hold trials found for subject {subject_id}")
        return

    trial_nums = subject_hold_trials['trial_num'].unique()

    subj_ts = time_series_df[
        (time_series_df['subject'] == subject_id) &
        (time_series_df['trial_num'].isin(trial_nums))
    ].copy()

    if subj_ts.empty:
        print(f"No time series data found for subject {subject_id}")
        return

    all_times = subj_ts['aligned_time'].dropna().to_numpy()
    time_grid = np.linspace(np.min(all_times), np.max(all_times), 400)

    pain_curves = []

    standalone = ax1 is None
    if standalone:
        fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    for trial_num, trial_df in subj_ts.groupby('trial_num'):
        trial_df = trial_df.sort_values('aligned_time')

        x = trial_df['aligned_time'].to_numpy()
        pain = trial_df['pain'].to_numpy()
        temp = trial_df['temperature'].to_numpy()

        if len(x) < 2:
            continue

        pain_interp = np.interp(time_grid, x, pain, left=np.nan, right=np.nan)
        pain_curves.append(pain_interp)

        # Plot individual pain traces
        ax1.plot(x, pain, color='gray', alpha=0.25, linewidth=1)

        # Plot individual temperature traces
        ax2.plot(x, temp, color='red', alpha=0.8, linewidth=1)

    # Mean pain trace
    pain_curves = np.array(pain_curves)
    mean_pain = np.nanmean(pain_curves, axis=0)
    ax1.plot(time_grid, mean_pain, color='black', linewidth=3, label='Mean pain')

    ax1.set_xlabel('Aligned time (s)')
    ax1.set_xlim(0, 60)
    ax1.set_ylabel('Pain')
    ax2.set_ylabel('Temperature')
    ax1.grid(True, alpha=0.3)

    if time_window is not None:
        ax1.set_xlim(time_window)

    if title is None:
        title = f"Subject {subject_id} hold trials"

    ax1.set_title(title)

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='best')

    if standalone:
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, format='svg', bbox_inches='tight')

        plt.show()
