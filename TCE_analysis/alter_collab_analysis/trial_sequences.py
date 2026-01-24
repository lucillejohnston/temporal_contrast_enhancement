#%%
# ========================================================
# CONFIGURATION
# ========================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os, json
import scipy.stats as stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_metrics.json'
TRIAL_DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json'

# Load the metrics data
with open(TRIAL_METRICS_PATH, 'r') as f:
    metrics_data = json.load(f)

with open(TRIAL_DATA_PATH, 'r') as f:
    trial_data = json.load(f)

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
trial_metrics_df = pd.DataFrame(records)


#%%
# ========================================================
# Calculate preceding trial metrics 
# ========================================================
def get_preceding_value(row, col):
    prev_trial_num = row['trial_num'] - 1
    subject = row['subject']
    prev_row = trial_metrics_df[
        (trial_metrics_df['subject'] == subject) & 
        (trial_metrics_df['trial_num'] == prev_trial_num)
    ]
    if not prev_row.empty:
        return prev_row.iloc[0][col]
    return None

# Define the preceding metrics to calculate
preceding_metrics = {
    'preceding_trial_type': 'trial_type',
    'preceding_abs_max_val': 'abs_max_val',
    'preceding_abs_min_val': 'abs_min_val',
    'preceding_abs_peak_to_peak': 'abs_peak_to_peak',
    'preceding_auc_A': 'auc_A',
    'preceding_auc_B': 'auc_B',
    'preceding_auc_C': 'auc_C',
    'preceding_auc_total': 'auc_total',
    'preceding_abs_normalized_pain_change': 'abs_normalized_pain_change'
}

# Add time-yoked metrics for control trials
for trial_type in trial_metrics_df['trial_type'].unique():
    if trial_type in ['t1_hold', 't2_hold']:
        # Add time-yoked metrics based on trial type
        if trial_type == 't1_hold':
            # t1_hold can reference offset or stepdown
            for ref_type in ['offset', 'stepdown']:
                preceding_metrics[f'preceding_time_yoked_max_val_{ref_type}'] = f'time_yoked_max_val_{ref_type}'
                preceding_metrics[f'preceding_time_yoked_min_val_{ref_type}'] = f'time_yoked_min_val_{ref_type}'
                preceding_metrics[f'preceding_time_yoked_peak_to_peak_{ref_type}'] = f'time_yoked_peak_to_peak_{ref_type}'
        elif trial_type == 't2_hold':
            # t2_hold references inv
            preceding_metrics['preceding_time_yoked_max_val_inv'] = 'time_yoked_max_val_inv'
            preceding_metrics['preceding_time_yoked_min_val_inv'] = 'time_yoked_min_val_inv'
            preceding_metrics['preceding_time_yoked_peak_to_peak_inv'] = 'time_yoked_peak_to_peak_inv'

for new_col, source_col in preceding_metrics.items():
    trial_metrics_df[new_col] = trial_metrics_df.apply(
        lambda row: get_preceding_value(row, source_col), axis=1
    )


#%%
# ========================================================
# Preliminary plots of OA/OH trials by preceding trial type
# ========================================================
# # Filter for OA/OH trials
# oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# # Violin plot of max_val in OA/OH trials, separated by preceding trial type
# plt.figure(figsize=(8, 6))
# sns.violinplot(
#     data=oa_oh_trials,
#     x='preceding_trial_type',
#     y='max_val',
#     hue='trial_type',
#     split=True
# )
# plt.title('OA/OH trial max_val by preceding trial type')
# plt.xlabel('Preceding Trial Type')
# plt.ylabel('max_val')
# plt.tight_layout()
# plt.show()

# # Plot normalized pain change for 'offset' trials
# plt.figure(figsize=(8, 6))
# sns.violinplot(
#     data=oa_oh_trials[oa_oh_trials['trial_type'] == 'offset'],
#     x='preceding_trial_type',
#     y='normalized_pain_change'
# )
# plt.title('Offset trials: Normalized pain change by preceding trial type')
# plt.xlabel('Preceding Trial Type')
# plt.ylabel('Normalized Pain Change (%)')
# plt.tight_layout()
# plt.show()

# # Plot normalized pain change for 'inv' trials
# plt.figure(figsize=(8, 6))
# sns.violinplot(
#     data=oa_oh_trials[oa_oh_trials['trial_type'] == 'inv'],
#     x='preceding_trial_type',
#     y='normalized_pain_change'
# )
# plt.title('Inv trials: Normalized pain change by preceding trial type')
# plt.xlabel('Preceding Trial Type')
# plt.ylabel('Normalized Pain Change (%)')
# plt.tight_layout()
# plt.show()

# %%
# ========================================================================================
# Compare Preceding Trial Metrics to Current Trial Metrics 
# normalized_pain_change (as metric of OH/OA) vs auc_total, abs_max_val, abs_min_val
# ========================================================================================
# Filter for OA/OH trials
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# Scatter plot: preceding_auc_total vs. normalized pain change for all OA/OH trials
for ttype in ['offset', 'inv']:
    create_correlation_scatter(                                                 # preceding AUC total vs. normalized pain change
        oa_oh_trials,
        x_col='preceding_auc_total',
        y_col='abs_normalized_pain_change',
        title=f"{ttype.capitalize()} trials: Normalized pain change vs. preceding AUC Total",
        xlabel='Preceding AUC Total',
        ylabel='Normalized Pain Change (%)',
        filter_col='trial_type',
        filter_val=ttype
    )

    create_correlation_scatter(                                                 # preceding absolute max_val vs. normalized pain change
        oa_oh_trials,
        x_col='preceding_abs_max_val',
        y_col='abs_normalized_pain_change',
        title=f"{ttype.capitalize()} trials: Normalized pain change vs. preceding Absolute Max Val",
        xlabel='Preceding Absolute Max Val',
        ylabel='Normalized Pain Change (%)',
        filter_col='trial_type',
        filter_val=ttype
    )

    create_correlation_scatter(                                                  # preceding peak-to-peak vs current normalized pain change
        oa_oh_trials,
        x_col='preceding_abs_peak_to_peak',
        y_col='abs_normalized_pain_change',
        title=f"{ttype.capitalize()} trials: Normalized pain change vs. preceding Peak-to-Peak",
        xlabel='Preceding Peak-to-Peak',
        ylabel='Normalized Pain Change (%)',
        filter_col='trial_type',
        filter_val=ttype
    )

# Scatter plot: preceding_min_value vs. normalized pain change for OA (offset) trials
create_correlation_scatter(
    oa_oh_trials,
    x_col='preceding_abs_min_val',
    y_col='abs_normalized_pain_change',
    title="Offset trials: Normalized pain change vs. preceding Min Value",
    xlabel='Preceding Min Value',
    ylabel='Normalized Pain Change (%)',
    filter_col='trial_type',
    filter_val='offset'
)

# Scatter plot: preceding_min_value vs. normalized pain change for OH (inv) trials
create_correlation_scatter(
    oa_oh_trials,
    x_col='preceding_abs_min_val',
    y_col='abs_normalized_pain_change',
    title="Inv trials: Normalized pain change vs. preceding Min Value",
    xlabel='Preceding Min Value',
    ylabel='Normalized Pain Change (%)',
    filter_col='trial_type',
    filter_val='inv'
)

# Scatter plot: preceding_abs_normalized_pain_change vs. normalized pain change OH trials
create_correlation_scatter(
    oa_oh_trials,
    x_col='preceding_abs_normalized_pain_change',
    y_col='abs_normalized_pain_change',
    title="OH trials: Normalized pain change vs. preceding Normalized Pain Change",
    xlabel='Preceding Normalized Pain Change (%)',
    ylabel='Normalized Pain Change (%)',
    filter_col='trial_type',
    filter_val='inv'
)

# Scatter plot: preceding_abs_normalized_pain_change vs. normalized pain change OA trials
create_correlation_scatter(
    oa_oh_trials,
    x_col='preceding_abs_normalized_pain_change',
    y_col='abs_normalized_pain_change',
    title="OA trials: Normalized pain change vs. preceding Normalized Pain Change",
    xlabel='Preceding Normalized Pain Change (%)',
    ylabel='Normalized Pain Change (%)',
    filter_col='trial_type',
    filter_val='offset'
)

# %%
# ========================================================================
# Compare Preceding Trial Metrics for FIRST OA/OH Trial Only
# ========================================================================

########################################################################### OFFSET TRIAL 
# Find the first 'offset' trial for each subject
first_offset_trials = (
    oa_oh_trials[oa_oh_trials['trial_type'] == 'offset']
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

# Replace NaN in preceding_trial_type with 'None' for plotting
first_offset_trials['preceding_trial_type'] = first_offset_trials['preceding_trial_type'].fillna('None')

# Violin plot of normalized pain change by preceding trial type
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=first_offset_trials,
    x='preceding_trial_type',
    y='abs_normalized_pain_change'
)
plt.title("First 'offset' trial: Normalized pain change by preceding trial type")
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()


# Scatter plot: preceding_auc_total vs. normalized pain change for first offset trial
create_correlation_scatter(
    first_offset_trials,
    x_col='preceding_auc_total',
    y_col='abs_normalized_pain_change',
    title="First Offset trial: Normalized pain change vs. preceding AUC Total",
    xlabel='Preceding AUC Total',
    ylabel='Normalized Pain Change (%)'
)


# Scatter plot: preceding_max_val vs. normalized pain change for first offset trial
create_correlation_scatter(
    first_offset_trials,
    x_col='preceding_abs_max_val',
    y_col='abs_normalized_pain_change',
    title="First Offset trial: Normalized pain change vs. preceding Max Val",
    xlabel='Preceding Max Val',
    ylabel='Normalized Pain Change (%)'
)

# Scatter plot: preceding_min_val vs. normalized pain change for first offset trial
create_correlation_scatter(
    first_offset_trials,
    x_col='preceding_abs_min_val',
    y_col='abs_normalized_pain_change',
    title="First Offset trial: Normalized pain change vs. preceding Min Val",
    xlabel='Preceding Min Val',
    ylabel='Normalized Pain Change (%)'
)

########################################################################### ONSET/INV TRIAL 

# Find the first 'inv' trial for each subject
first_inv_trials = (
    oa_oh_trials[oa_oh_trials['trial_type'] == 'inv']
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

# Replace NaN in preceding_trial_type with 'None' for plotting
first_inv_trials['preceding_trial_type'] = first_inv_trials['preceding_trial_type'].fillna('None')

# Violin plot of normalized pain change by preceding trial type
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=first_inv_trials,
    x='preceding_trial_type',
    y='abs_normalized_pain_change'
)
plt.title("First 'inv' trial: Normalized pain change by preceding trial type")
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Scatter plot: preceding_auc_total vs. normalized pain change for first inv trial
create_correlation_scatter(
    first_inv_trials,
    x_col='preceding_auc_total',
    y_col='abs_normalized_pain_change',
    title="First Inv trial: Normalized pain change vs. preceding AUC Total",
    xlabel='Preceding AUC Total',
    ylabel='Normalized Pain Change (%)'
)

# Scatter plot: preceding_max_val vs. normalized pain change for first inv trial
create_correlation_scatter(
    first_inv_trials,
    x_col='preceding_abs_max_val',
    y_col='abs_normalized_pain_change',
    title="First Inv trial: Normalized pain change vs. preceding Max Val",
    xlabel='Preceding Max Val',
    ylabel='Normalized Pain Change (%)'
)

# Scatter plot: preceding_min_val vs. normalized pain change for first inv trial
create_correlation_scatter(
    first_inv_trials,
    x_col='preceding_abs_min_val',
    y_col='abs_normalized_pain_change',
    title="First Inv trial: Normalized pain change vs. preceding Min Val",
    xlabel='Preceding Min Val',
    ylabel='Normalized Pain Change (%)'
)


# %%
# ========================================================
# Comprehensive grid plot of normalized pain change vs. preceding min and max values for OA and OH trials
# ========================================================

# PLOT NORMALIZED PAIN CHANGE V MIN / MAX AS A GRID FOR OFFSET AND INV TRIALS
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex='col', sharey='row')

# Define trial types and metrics
trial_types = ['offset', 'inv']
metrics = ['preceding_abs_min_val', 'preceding_abs_max_val']
metric_labels = ['Preceding Min Value', 'Preceding Max Value']

# Color scheme
colors = {'offset': 'blue', 'inv': 'red'}

for row_idx, ttype in enumerate(trial_types):
    subset = oa_oh_trials[oa_oh_trials['trial_type'] == ttype]
    
    for col_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row_idx, col_idx]
        
        # Get data
        mask = subset[metric].notnull() & subset['abs_normalized_pain_change'].notnull()
        x = subset.loc[mask, metric]
        y = subset.loc[mask, 'abs_normalized_pain_change']
        
        if len(x) > 1:
            # Compute regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=50, color=colors[ttype], edgecolors='black', linewidth=0.5)
            
            # Add regression line if significant
            if p_value < 0.05:
                x_vals = np.array([x.min(), x.max()])
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color='black', linewidth=2, linestyle='--')
            
            # Add stats text box
            ax.text(0.05, 0.95,
                   f"$R$ = {r_value:.2f}\np = {p_value:.3g}\nn = {len(x)}",
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor='black'))
        
        # Set labels
        if row_idx == 1:  # Bottom row
            ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
        if col_idx == 0:  # Left column
            ax.set_ylabel('Normalized Pain Change (%)', fontsize=12, fontweight='bold')
        
        # Add title
        title = f"{ttype.capitalize()}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Set different y-axis limits for top and bottom rows
        if row_idx == 0:  # Top row (offset)
            ax.set_ylim(-102, 0) # make -102 to give some space below -100
        else:  # Bottom row (inv)
            ax.set_ylim(0, 102) # make 102 to give some space above 100

# Add overall title
fig.suptitle('Normalized Pain Change vs Preceding Trial Metrics', 
             fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])
# Save figure
plt.savefig('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/pain_change_vs_preceding_metrics_grid.svg', 
            dpi=300, bbox_inches='tight')
plt.show()

# %%