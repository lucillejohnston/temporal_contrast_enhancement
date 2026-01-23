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
# FUNCTIONS
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

def create_correlation_scatter(df, x_col, y_col, title=None, xlabel=None, ylabel=None,
                               filter_col=None, filter_val=None, figsize=(8,6)):
    """
    Create a correlation scatter plot with regression line and statistics
    Parameters:
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    title : str, optional
        Plot title (auto-generated if None)
    xlabel : str, optional
        X-axis label (uses column name if None
    ylabel : str, optional
        Y-axis label (uses column name if None)
    filter_col : str, optional
        Column name to filter data
    filter_val : any, optional
        Value to filter data
    figsize : tuple, optional
        Figure size

    Returns:
    dict: correlation statistics (r_value, p_value, slope, intercept)
    """
    # Apply filter if specified
    if filter_col and filter_val:
        if isinstance(filter_val, list):
            subset = df[df[filter_col].isin(filter_val)]
        else:
            subset = df[df[filter_col] == filter_val]
    else:
        subset = df.copy()

    # Remove null values
    mask = subset[x_col].notnull() & subset[y_col].notnull()
    x = subset.loc[mask, x_col]
    y = subset.loc[mask, y_col]
    
    # Calculate correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"{x_col} vs. {y_col}:")
    print(f"  Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

    # Create scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.7)
    if p_value < 0.05:                                          # If significant, add regression line
        x_vals = np.array([x.min(), x.max()])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='red')
    plt.text(0.05, 0.95,                                        # Add statistics text
             f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.title(title if title else f"{y_col} vs. {x_col}")       # Add titles and labels
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.tight_layout()
    plt.show()
    return {
        "r_value": r_value,
        "p_value": p_value,
        "slope": slope,
        "intercept": intercept
    }
#%%
# ========================================================
# Calculate preceding trial metrics 
# ========================================================
# Add preceding trial metrics 
preceding_metrics = {
    'preceding_trial_type': 'trial_type',
    'preceding_abs_max_val': 'abs_max_val',
    'preceding_abs_min_val': 'abs_min_val',
    'preceding_abs_peak_to_peak': 'abs_peak_to_peak',
    'preceding_auc_A': 'auc_A',
    'preceding_auc_B': 'auc_B',
    'preceding_auc_C': 'auc_C',
    'preceding_auc_total': 'auc_total'
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

# Filter for OA/OH trials
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

#%%
# ========================================================
# Preliminary plots of OA/OH trials by preceding trial type
# ========================================================

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

#%% 
# ========================================================
# Which is better auc_total or max_val?
# Also time_yoked max_val vs. absolute max_val
# ========================================================

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

# Compare abs_max_val and time_yoked_max_val for control trials
control_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['t1_hold', 't2_hold'])].copy()
create_correlation_scatter(
    control_trials,
    x_col='time_yoked_max_val_offset',
    y_col='abs_max_val',
    title='Control Trials: Absolute Max Val vs. Time-Yoked Max Val',
    xlabel='Time-Yoked Max Val',
    ylabel='Absolute Max Val'
)
# Compare abs_min_val and time_yoked_min_val for control trials
create_correlation_scatter(
    control_trials,
    x_col='time_yoked_min_val_offset',
    y_col='abs_min_val',
    title='Control Trials: Absolute Min Val vs. Time-Yoked Min Val',
    xlabel='Time-Yoked Min Val',
    ylabel='Absolute Min Val'
)

# %%
# ========================================================
# Correlation of preceding trial metrics with normalized pain change in OA/OH trials
# ========================================================

# Scatter plot: preceding_auc_total vs. normalized pain change for all OA/OH trials
for ttype in ['offset', 'inv']:
    create_correlation_scatter(                                                 # preceding AUC total vs. normalized pain change
        oa_oh_trials,
        x_col='preceding_auc_total',
        y_col='normalized_pain_change',
        title=f"{ttype.capitalize()} trials: Normalized pain change vs. preceding AUC Total",
        xlabel='Preceding AUC Total',
        ylabel='Normalized Pain Change (%)',
        filter_col='trial_type',
        filter_val=ttype
    )

    create_correlation_scatter(                                                 # preceding time-yoked max_val vs. normalized pain change
        oa_oh_trials,
        x_col='preceding_time_yoked_max_val',
        y_col='normalized_pain_change',
        title=f"{ttype.capitalize()} trials: Normalized pain change vs. preceding Time-Yoked Max Val",
        xlabel='Preceding Time-Yoked Max Val',
        ylabel='Normalized Pain Change (%)',
        filter_col='trial_type',
        filter_val=ttype
    )

    create_correlation_scatter(                                                 # preceding absolute max_val vs. normalized pain change
        oa_oh_trials,
        x_col='preceding_abs_max_val',
        y_col='normalized_pain_change',
        title=f"{ttype.capitalize()} trials: Normalized pain change vs. preceding Absolute Max Val",
        xlabel='Preceding Absolute Max Val',
        ylabel='Normalized Pain Change (%)',
        filter_col='trial_type',
        filter_val=ttype
    )
# Calculate preceding min_val for OA trials
trial_metrics_df['preceding_time_yoked_min_val'] = trial_metrics_df.groupby('subject')['time_yoked_min_val'].shift(1)
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# Scatter plot: preceding_min_value vs. normalized pain change for OA (offset) trials
subset = oa_oh_trials[oa_oh_trials['trial_type'] == 'offset']
mask = subset['preceding_time_yoked_min_val'].notnull() & subset['normalized_pain_change'].notnull()
x = subset.loc[mask, 'preceding_time_yoked_min_val']
y = subset.loc[mask, 'normalized_pain_change']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"OA trials - preceding_min_value:")
print(f"  Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
if p_value < 0.05:
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red')
plt.text(0.05, 0.95,
            f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.title("Offset trials: Normalized pain change vs. preceding Min Value")
plt.xlabel('Preceding Min Value')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()


# Scatter plot: preceding_min_value vs. normalized pain change for OH (inv) trials
subset = oa_oh_trials[oa_oh_trials['trial_type'] == 'inv']
mask = subset['preceding_min_value'].notnull() & subset['normalized_pain_change'].notnull()
x = subset.loc[mask, 'preceding_min_value']
y = subset.loc[mask, 'normalized_pain_change']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"OH trials - preceding_min_value:")
print(f"  Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
if p_value < 0.05:
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red')
plt.text(0.05, 0.95,
            f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.title("Inv trials: Normalized pain change vs. preceding Min Value")
plt.xlabel('Preceding Min Value')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# %%
"""
Looking at OA on the first OA trial only
"""
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
    y='normalized_pain_change'
)
plt.title("First 'offset' trial: Normalized pain change by preceding trial type")
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()


# Scatter plot: preceding_auc_total vs. normalized pain change for first offset trial
slope, intercept, r_value, p_value, std_err = stats.linregress(
    first_offset_trials['preceding_auc_total'].dropna(),
    first_offset_trials['normalized_pain_change'].dropna()
)
print(f"Correlation r: {r_value:.3f}, R^2: {r_value**2:.3f}, p-value: {p_value:.3g}")
plt.figure(figsize=(8, 6))
plt.scatter(
    first_offset_trials['preceding_auc_total'],
    first_offset_trials['normalized_pain_change'],
    alpha=0.7
)
plt.title("First 'offset' trial: Normalized pain change vs. preceding AUC Total")
plt.xlabel('Preceding AUC Total')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()


# Scatter plot: preceding_max_val vs. normalized pain change for first offset trial
mask = first_offset_trials['preceding_max_val'].notnull() & first_offset_trials['normalized_pain_change'].notnull()
x = first_offset_trials.loc[mask, 'preceding_max_val']
y = first_offset_trials.loc[mask, 'normalized_pain_change']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Correlation r: {r_value:.3f}, R^2: {r_value**2:.3f}, p-value: {p_value:.3g}")
plt.figure(figsize=(8, 6))
plt.scatter(
    first_offset_trials['preceding_max_val'],
    first_offset_trials['normalized_pain_change'],
    alpha=0.7
)
plt.title("First 'offset' trial: Normalized pain change vs. preceding Max Val")
plt.xlabel('Preceding Max Val')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Scatter plot: preceding_min_val vs. normalized pain change for first offset trial
mask = first_offset_trials['preceding_min_value'].notnull() & first_offset_trials['normalized_pain_change'].notnull()
x = first_offset_trials.loc[mask, 'preceding_min_value']
y = first_offset_trials.loc[mask, 'normalized_pain_change'] 
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Correlation r: {r_value:.3f}, R^2: {r_value**2:.3f}, p-value: {p_value:.3g}")
plt.figure(figsize=(8, 6))
plt.scatter(
    first_offset_trials['preceding_min_value'],
    first_offset_trials['normalized_pain_change'],
    alpha=0.7
)
plt.title("First 'offset' trial: Normalized pain change vs. preceding Min Value")
plt.xlabel('Preceding Min Value')
plt.ylabel('Normalized Pain Change (%)')
plt.text(0.05, 0.95,
             f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
plt.show()


# %%
"""
Look at OH on the first OH trial only
"""
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
    y='normalized_pain_change'
)
plt.title("First 'inv' trial: Normalized pain change by preceding trial type")
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Scatter plot: preceding_auc_total vs. normalized pain change for first inv trial
mask = first_inv_trials['preceding_auc_total'].notnull() & first_inv_trials['normalized_pain_change'].notnull()
x = first_inv_trials.loc[mask, 'preceding_auc_total']
y = first_inv_trials.loc[mask, 'normalized_pain_change']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

plt.figure(figsize=(8, 6))
plt.scatter(
    first_inv_trials['preceding_auc_total'],
    first_inv_trials['normalized_pain_change'],
    alpha=0.7
)
x_vals = np.array([x.min(), x.max()])
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red', label=f'$R^2$={r_value**2:.2f}')
plt.legend()
plt.title("First 'inv' trial: Normalized pain change vs. preceding AUC Total")
plt.xlabel('Preceding AUC Total')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Scatter plot: preceding_max_val vs. normalized pain change for first inv trial
mask = first_inv_trials['preceding_max_val'].notnull() & first_inv_trials['normalized_pain_change'].notnull()
x = first_inv_trials.loc[mask, 'preceding_max_val']
y = first_inv_trials.loc[mask, 'normalized_pain_change']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")
plt.figure(figsize=(8, 6))
plt.scatter(
    first_inv_trials['preceding_max_val'],
    first_inv_trials['normalized_pain_change'],
    alpha=0.7
)
x_vals = np.array([x.min(), x.max()])
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='red')
plt.text(0.05, 0.95,
             f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.legend()
plt.title("First 'inv' trial: Normalized pain change vs. preceding Max Val")
plt.xlabel('Preceding Max Val')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Find the first 'offset' trial for each subject
first_offset_trials = (
    oa_oh_trials[oa_oh_trials['trial_type'] == 'offset']
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

# Scatter plot: preceding_min_value vs. normalized pain change for first OA trial
mask = first_offset_trials['preceding_min_value'].notnull() & first_offset_trials['normalized_pain_change'].notnull()
x = first_offset_trials.loc[mask, 'preceding_min_value']
y = first_offset_trials.loc[mask, 'normalized_pain_change']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"First OA trial - preceding_min_value:")
print(f"  Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7)
if p_value < 0.05:
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='red')
plt.title("First 'offset' trial: Normalized pain change vs. preceding Min Value")
plt.xlabel('Preceding Min Value')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# # %%
# """
# Comprehensive grid plot of normalized pain change vs. preceding min and max values for OA and OH trials
# """
# # PLOT NORMALIZED PAIN CHANGE V MIN / MAX AS A GRID FOR OFFSET AND INV TRIALS
# fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex='col', sharey='row')

# # Define trial types and metrics
# trial_types = ['offset', 'inv']
# metrics = ['preceding_min_value', 'preceding_max_val']
# metric_labels = ['Preceding Min Value', 'Preceding Max Value']

# # Color scheme
# colors = {'offset': 'blue', 'inv': 'red'}

# for row_idx, ttype in enumerate(trial_types):
#     subset = oa_oh_trials[oa_oh_trials['trial_type'] == ttype]
    
#     for col_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
#         ax = axes[row_idx, col_idx]
        
#         # Get data
#         mask = subset[metric].notnull() & subset['normalized_pain_change'].notnull()
#         x = subset.loc[mask, metric]
#         y = subset.loc[mask, 'normalized_pain_change']
        
#         if len(x) > 1:
#             # Compute regression
#             slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
#             # Scatter plot
#             ax.scatter(x, y, alpha=0.6, s=50, color=colors[ttype], edgecolors='black', linewidth=0.5)
            
#             # Add regression line if significant
#             if p_value < 0.05:
#                 x_vals = np.array([x.min(), x.max()])
#                 y_vals = intercept + slope * x_vals
#                 ax.plot(x_vals, y_vals, color='black', linewidth=2, linestyle='--')
            
#             # Add stats text box
#             ax.text(0.05, 0.95,
#                    f"$R$ = {r_value:.2f}\np = {p_value:.3g}\nn = {len(x)}",
#                    transform=ax.transAxes,
#                    verticalalignment='top',
#                    fontsize=10,
#                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor='black'))
        
#         # Set labels
#         if row_idx == 1:  # Bottom row
#             ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
#         if col_idx == 0:  # Left column
#             ax.set_ylabel('Normalized Pain Change (%)', fontsize=12, fontweight='bold')
        
#         # Add title
#         title = f"{ttype.capitalize()}"
#         ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
#         # Grid
#         ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
#         # Set different y-axis limits for top and bottom rows
#         if row_idx == 0:  # Top row (offset)
#             ax.set_ylim(-102, 0) # make -102 to give some space below -100
#         else:  # Bottom row (inv)
#             ax.set_ylim(0, 102) # make 102 to give some space above 100

# # Add overall title
# fig.suptitle('Normalized Pain Change vs Preceding Trial Metrics', 
#              fontsize=16, fontweight='bold', y=0.995)

# # Adjust layout
# plt.tight_layout(rect=[0, 0, 1, 0.99])
# # Save figure
# plt.savefig('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/pain_change_vs_preceding_metrics_grid.svg', 
#             dpi=300, bbox_inches='tight')
# plt.show()

# %%
"""
Compare time-yoked 'min_val' and 'max_val' to absolute 'abs_min_val' and 'abs_max_val' for t1_hold and t2_hold trials
"""
control_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['t1_hold', 't2_hold'])].copy()

t1_hold_trials = control_trials[control_trials['trial_type'] == 't1_hold'].copy()

# Create comparison plots for t1_hold trials
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Min values comparison
ax = axes[0, 0]
# Remove NaN values for plotting
mask = t1_hold_trials['min_val_offset'].notna() & t1_hold_trials['abs_min_val'].notna()
x = t1_hold_trials.loc[mask, 'min_val_offset']
y = t1_hold_trials.loc[mask, 'abs_min_val']

if len(x) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax.scatter(x, y, alpha=0.7, s=50, color='green', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect correlation)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    
    # Add regression line
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='red', linewidth=2, label=f'R={r_value:.2f}')
    
    ax.text(0.05, 0.95, f"R = {r_value:.2f}\np = {p_value:.3g}\nn = {len(x)}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax.set_xlabel('Time-yoked Min (offset reference)')
ax.set_ylabel('Absolute Min')
ax.set_title('T1_Hold: Min Values Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Max values comparison  
ax = axes[0, 1]
mask = t1_hold_trials['max_val_offset'].notna() & t1_hold_trials['abs_max_val'].notna()
x = t1_hold_trials.loc[mask, 'max_val_offset']
y = t1_hold_trials.loc[mask, 'abs_max_val']

if len(x) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax.scatter(x, y, alpha=0.7, s=50, color='orange', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    
    # Add regression line
    x_vals = np.array([x.min(), x.max()])
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color='red', linewidth=2, label=f'R={r_value:.2f}')
    
    ax.text(0.05, 0.95, f"R = {r_value:.2f}\np = {p_value:.3g}\nn = {len(x)}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax.set_xlabel('Time-yoked Max (offset reference)')
ax.set_ylabel('Absolute Max')
ax.set_title('T1_Hold: Max Values Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Difference plots
ax = axes[1, 0]
min_diff = t1_hold_trials.loc[mask, 'abs_min_val'] - t1_hold_trials.loc[mask, 'min_val_offset']
ax.hist(min_diff.dropna(), bins=20, alpha=0.7, color='green', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.set_xlabel('Absolute Min - Time-yoked Min')
ax.set_ylabel('Frequency')
ax.set_title('T1_Hold: Min Value Differences')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
mask = t1_hold_trials['max_val_offset'].notna() & t1_hold_trials['abs_max_val'].notna()
max_diff = t1_hold_trials.loc[mask, 'abs_max_val'] - t1_hold_trials.loc[mask, 'max_val_offset']
ax.hist(max_diff.dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.set_xlabel('Absolute Max - Time-yoked Max')
ax.set_ylabel('Frequency')
ax.set_title('T1_Hold: Max Value Differences')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('T1_Hold Trials: Time-yoked vs Absolute Extrema', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# %%
