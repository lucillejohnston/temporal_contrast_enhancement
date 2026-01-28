#%%
# ========================================================
# CONFIGURATION
# ========================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os, json
from scipy import stats
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_metrics.json'
TRIAL_DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json'

# Load the metrics data
with open(TRIAL_METRICS_PATH, 'r') as f:
    metrics_data = json.load(f)

with open(TRIAL_DATA_PATH, 'r') as f:
    time_series_trial_data = json.load(f)

# Convert to DataFrame
trial_metrics_df = {}
records = []
for subject_id, trials in metrics_data.items():
    for trial_num, trial_data in trials.items():
        record = {
            'subject': int(subject_id),
            'trial_num': int(trial_num),
            **trial_data
        }
        records.append(record)

# Convert to DataFrame
trial_metrics_df = pd.DataFrame(records)
time_series_df = pd.DataFrame(time_series_trial_data)

#%%
# ============================================================================
# Compare Metrics to Each Other 
# auc_total vs max_val; 
# abs_max_val vs time_yoked_max_val; 
# abs_normalized_pain_change vs time_yoked_abs_normalized_pain_change)
# ============================================================================

###################################################################################################### Overall distributions of auc_total and abs_max_val
# Plot histrograms of auc_total for all trial_types
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=trial_metrics_df,
    x='auc_total',
    hue='trial_type',
    multiple='stack',
    bins=30,
    kde=True
)
plt.title('Distribution of AUC Total by Trial Type')
plt.xlabel('AUC Total')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot histrograms of abs_max_val for all trial_types
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=trial_metrics_df,
    x='abs_max_val',
    hue='trial_type',
    multiple='stack',
    bins=30,
    kde=True    
)
plt.title("Distribution of Absolute Max Val by Trial Type")
plt.xlabel('Absolute Max Val')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot: auc_total vs. abs_max_val for all trials
create_correlation_scatter(
    trial_metrics_df,
    x_col='auc_total',
    y_col='abs_max_val',
    title='AUC Total vs. Absolute Max Val for All Trials',
    xlabel='AUC Total',
    ylabel='Absolute Max Val'
)

############################################################################################### Compare abs_max_val and time_yoked_max_val for control trials
control_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['t1_hold', 't2_hold'])].copy()
# Create temporary DataFrame for max values
temp_max_df = pd.DataFrame({
    'time_yoked_max_combined': pd.concat([
        control_trials['time_yoked_max_val_offset'], 
        control_trials['time_yoked_max_val_inv']
    ]).dropna(),
    'abs_max_combined': pd.concat([
        control_trials.loc[control_trials['time_yoked_max_val_offset'].notna(), 'abs_max_val'],
        control_trials.loc[control_trials['time_yoked_max_val_inv'].notna(), 'abs_max_val']
    ])
}).reset_index(drop=True)

create_correlation_scatter(
    temp_max_df, 
    x_col='time_yoked_max_combined', 
    y_col='abs_max_combined',
    title='Control Trials: Absolute Max Val vs. Time-Yoked Max Val',
    xlabel='Time-Yoked Max Val',
    ylabel='Absolute Max Val'
)

# Create temporary DataFrame for min values
temp_min_df = pd.DataFrame({
    'time_yoked_min_combined': pd.concat([
        control_trials['time_yoked_min_val_offset'], 
        control_trials['time_yoked_min_val_inv']
    ]).dropna(),
    'abs_min_combined': pd.concat([
        control_trials.loc[control_trials['time_yoked_min_val_offset'].notna(), 'abs_min_val'],
        control_trials.loc[control_trials['time_yoked_min_val_inv'].notna(), 'abs_min_val']
    ])
}).reset_index(drop=True)

create_correlation_scatter(
    temp_min_df,
    x_col='time_yoked_min_combined',  
    y_col='abs_min_combined',  
    title='Control Trials: Absolute Min Val vs. Time-Yoked Min Val',
    xlabel='Time-Yoked Min Val',
    ylabel='Absolute Min Val'
)

############################################################################################ Compare abs_normalized_pain_change and time_yoked_normalized_pain_change for control trials
# Separate t1_hold and t2_hold trials
t1_hold_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 't1_hold'].copy()
t2_hold_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 't2_hold'].copy()

# Compare time-yoked vs absolute normalized pain change for t1_hold trials
create_correlation_scatter(
    t1_hold_trials,
    x_col='time_yoked_normalized_pain_change',
    y_col='abs_normalized_pain_change',
    title='T1_Hold Trials: Time-Yoked vs Absolute Normalized Pain Change',
    xlabel='Time-Yoked Normalized Pain Change (%)',
    ylabel='Absolute Normalized Pain Change (%)'
)

# Compare time-yoked vs absolute normalized pain change for t2_hold trials
create_correlation_scatter(
    t2_hold_trials,
    x_col='time_yoked_normalized_pain_change',
    y_col='abs_normalized_pain_change',
    title='T2_Hold Trials: Time-Yoked vs Absolute Normalized Pain Change',
    xlabel='Time-Yoked Normalized Pain Change (%)',
    ylabel='Absolute Normalized Pain Change (%)'
)
# Plot them together because it looks cute
plt.figure(figsize=(10, 8))
for trial_type, color in [('t1_hold', 'blue'), ('t2_hold', 'red')]:
    subset = control_trials[control_trials['trial_type'] == trial_type]
    mask = subset['time_yoked_normalized_pain_change'].notnull() & subset['abs_normalized_pain_change'].notnull()
    x = subset.loc[mask, 'time_yoked_normalized_pain_change']
    y = subset.loc[mask, 'abs_normalized_pain_change']
    
    plt.scatter(x, y, alpha=0.7, label=trial_type, color=color)

plt.xlabel('Time-Yoked Normalized Pain Change (%)')
plt.ylabel('Absolute Normalized Pain Change (%)')
plt.title('Control Trials: Time-Yoked vs Absolute Normalized Pain Change')
plt.legend()
plt.xlim(-150, 150)
plt.grid(True, alpha=0.3)
plt.show()

############################################################################################## Compare t1_hold vs offset/stepdown and t2_hold vs inv for max and min values
comparisons = [
    ('t1_hold', 'offset', 'max_val', 'T1_Hold vs Offset: Max Value', None),
    ('t1_hold', 'stepdown', 'min_val', 'T1_Hold vs Stepdown: Min Value', None),
    ('t2_hold', 'inv', 'max_val', 'T2_Hold vs Inv: Max Value', None)
]
for trial1, trial2, metric, label, ylims in comparisons:
    df1 = trial_metrics_df[trial_metrics_df['trial_type'] == trial1].copy()
    df2 = trial_metrics_df[trial_metrics_df['trial_type'] == trial2].copy()

    common_subjects = set(df1['subject']) & set(df2['subject'])
    df1 = df1[df1['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
    df2 = df2[df2['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
    
    # Build metric names based on trial type
    if trial1 in ['t1_hold', 't2_hold']:  # Control trials
        if trial1 == 't1_hold' and trial2 in ['offset', 'stepdown']:
            metric1 = f'time_yoked_{metric}_{trial2}'  # e.g., 'time_yoked_min_val_offset'
        elif trial1 == 't2_hold' and trial2 == 'inv':
            metric1 = f'time_yoked_{metric}_inv'       # e.g., 'time_yoked_max_val_inv'
    
    if trial2 in ['offset', 'stepdown', 'inv']:  # Stepped trials
        metric2 = f'abs_{metric}'  # e.g., 'abs_min_val' or 'abs_max_val'
    
    vals1 = df1[metric1]
    vals2 = df2[metric2]
    
    data = pd.DataFrame({trial1: vals1, trial2: vals2}).dropna()
    means = [data[trial1].mean(), data[trial2].mean()]
    cis = [mean_ci(data[trial1]), mean_ci(data[trial2])]
    errors = [[m - ci[1], ci[2] - m] for m, ci in zip(means, cis)]
    
    # Paired t-test
    tstat, pval = stats.ttest_rel(data[trial1], data[trial2])
    print(f"{label}: n={len(data)} subjects, t={tstat:.3f}, p={pval:.4f}")
    
    # Barplot
    plt.figure(figsize=(6, 5))
    plt.bar([0, 1], means, yerr=np.array(errors).T, capsize=8, color=['#4F81BD', '#C0504D'])
    plt.xticks([0, 1], [trial1, trial2])
    plt.ylabel('Pain Rating')
    plt.title(f"{label}\n(p={pval:.4f})")
    if ylims and (ylims[0] is not None or ylims[1] is not None):
        plt.ylim(ylims)
    plt.tight_layout()
    plt.show()



