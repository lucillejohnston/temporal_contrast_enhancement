#%%
"""
Section 1: Load the data and import relevant packages
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os
import scipy.stats as stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_metrics.csv'
TRIAL_DATA_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data_cleaned_aligned.json'

# Load the metrics data
trial_metrics_df = pd.read_csv(TRIAL_METRICS_PATH)

# Load the raw trial data (for time series plotting)
df = pd.read_json(TRIAL_DATA_PATH, orient='records')
#%%
"""
Section 2: Trial t-1

For OA/OH trials (offset/inv), examine how the preceding trial affects current trial metrics.
"""
# Add preceding trial type and metrics using groupby + shift
trial_metrics_df['preceding_trial_type'] = trial_metrics_df.groupby('subject')['trial_type'].shift(1)
trial_metrics_df['preceding_max_val'] = trial_metrics_df.groupby('subject')['max_val'].shift(1)
trial_metrics_df['preceding_peak_to_peak'] = trial_metrics_df.groupby('subject')['peak_to_peak'].shift(1)
trial_metrics_df['preceding_auc_A'] = trial_metrics_df.groupby('subject')['auc_A'].shift(1)
trial_metrics_df['preceding_auc_B'] = trial_metrics_df.groupby('subject')['auc_B'].shift(1)
trial_metrics_df['preceding_auc_C'] = trial_metrics_df.groupby('subject')['auc_C'].shift(1)
trial_metrics_df['preceding_auc_total'] = (
    trial_metrics_df['preceding_auc_A'].fillna(0) +
    trial_metrics_df['preceding_auc_B'].fillna(0) +
    trial_metrics_df['preceding_auc_C'].fillna(0) 
)

def get_preceding_value(row, col):
    prev = row['trial_num'] - 1
    subj = row['subject']
    prev_row = trial_metrics_df[(trial_metrics_df['subject'] == subj) & (trial_metrics_df['trial_num'] == prev)]
    if not prev_row.empty:
        # For t1_hold, use *_offset columns if available
        if row['trial_type'] == 't1_hold':
            if col == 'max_val':
                return prev_row.iloc[0].get('max_val_offset', None)
            if col == 'min_val':
                return prev_row.iloc[0].get('min_val_offset', None)
        return prev_row.iloc[0][col]
    return None

# Recompute preceding values using the updated function
trial_metrics_df['preceding_max_val'] = trial_metrics_df.apply(
    lambda row: get_preceding_value(row, 'max_val'), axis=1
)
trial_metrics_df['preceding_min_value'] = trial_metrics_df.apply(
    lambda row: get_preceding_value(row, 'min_val'), axis=1
)

# Filter for OA/OH trials
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# Add preceding trial type and metrics
def get_preceding_value(row, col):
    prev = row['trial_num'] - 1
    subj = row['subject']
    prev_row = trial_metrics_df[(trial_metrics_df['subject'] == subj) & (trial_metrics_df['trial_num'] == prev)]
    if not prev_row.empty:
        return prev_row.iloc[0][col]
    return None

trial_metrics_df['preceding_trial_type'] = trial_metrics_df.apply(
    lambda row: get_preceding_value(row, 'trial_type'), axis=1
)
trial_metrics_df['preceding_max_val'] = trial_metrics_df.apply(
    lambda row: get_preceding_value(row, 'max_val'), axis=1
)
trial_metrics_df['preceding_peak_to_peak'] = trial_metrics_df.apply(
    lambda row: get_preceding_value(row, 'peak_to_peak'), axis=1
)
trial_metrics_df['preceding_auc_total'] = trial_metrics_df.apply(
    lambda row: sum([get_preceding_value(row, auc) or 0 for auc in ['auc_A', 'auc_B', 'auc_C']]), axis=1
)

# Filter for OA/OH trials
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

#%%
"""
Section 3: Plotting the effects of preceding trial type on OA/OH trials
"""
# Violin plot of max_val in OA/OH trials, separated by preceding trial type
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.violinplot(
    data=oa_oh_trials,
    x='preceding_trial_type',
    y='max_val',
    hue='trial_type',
    split=True
)
plt.title('OA/OH trial max_val by preceding trial type')
plt.xlabel('Preceding Trial Type')
plt.ylabel('max_val')
plt.tight_layout()
plt.show()

# Plot normalized pain change for 'offset' trials
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=oa_oh_trials[oa_oh_trials['trial_type'] == 'offset'],
    x='preceding_trial_type',
    y='normalized_pain_change'
)
plt.title('Offset trials: Normalized pain change by preceding trial type')
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Plot normalized pain change for 'inv' trials
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=oa_oh_trials[oa_oh_trials['trial_type'] == 'inv'],
    x='preceding_trial_type',
    y='normalized_pain_change'
)
plt.title('Inv trials: Normalized pain change by preceding trial type')
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

#%% 
"""
Section 4: Plotting AUC and max_val distributions for all trial types
"""
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

plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=trial_metrics_df,
    x='max_val',
    hue='trial_type',
    multiple='stack',
    bins=30,
    kde=True    
)
plt.title("Distribution of Max Val by Trial Type")
plt.xlabel('Max Val')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
# %%
"""
Section 5: Plotting effects of preceding pain on OH/OA trials
"""
# Scatter plot: preceding_auc_total vs. normalized pain change for all OA/OH trials
for ttype in ['offset', 'inv']:
    subset = oa_oh_trials[oa_oh_trials['trial_type'] == ttype]
    mask = subset['preceding_auc_total'].notnull() & subset['normalized_pain_change'].notnull()
    x = subset.loc[mask, 'preceding_auc_total']
    y = subset.loc[mask, 'normalized_pain_change']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"{ttype} trials - preceding_auc_total:")
    print(f"  Correlation r: {r_value:.3f}, R: {r_value:.3f}, p-value: {p_value:.3g}")
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
    plt.title(f"{ttype.capitalize()} trials: Normalized pain change vs. preceding AUC Total")
    plt.xlabel('Preceding AUC Total')
    plt.ylabel('Normalized Pain Change (%)')
    plt.tight_layout()
    plt.savefig(f"{ttype}_trials_preceding_auc_total.svg", dpi=300)
    plt.show()

# Scatter plot: preceding_max_val vs. normalized pain change for all OA/OH trials 
for ttype in ['offset', 'inv']:
    subset = oa_oh_trials[oa_oh_trials['trial_type'] == ttype]
    mask = subset['preceding_max_val'].notnull() & subset['normalized_pain_change'].notnull()
    x = subset.loc[mask, 'preceding_max_val']
    y = subset.loc[mask, 'normalized_pain_change']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"{ttype} trials - preceding_max_val:")
    print(f"  Correlation r: {r_value:.3f}, R: {r_value:.3f}, p-value: {p_value:.3g}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    if p_value < 0.05:
        x_vals = np.array([x.min(), x.max()])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='red')
    #Add text annotation for p-value and R on the figure
    plt.text(0.05, 0.95,
             f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.title(f"{ttype.capitalize()} trials: Normalized pain change vs. preceding Max Val")
    plt.xlabel('Preceding Max Val')
    plt.ylabel('Normalized Pain Change (%)')
    plt.tight_layout()
    plt.savefig(f"{ttype}_trials_preceding_max_val.svg", dpi=300)
    plt.show()

# %%
"""
Section 6: Look at OA on the first OA trial only
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
Section 7: Look at OH on the first OH trial only
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



# %% 
"""
Section 8: Preceding Min Value vs. Normalized Pain Change for OA Trials
"""
# Calculate preceding min_val for OA trials
trial_metrics_df['preceding_min_value'] = trial_metrics_df.groupby('subject')['min_val'].shift(1)
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# Scatter plot: preceding_min_value vs. normalized pain change for OA (offset) trials
subset = oa_oh_trials[oa_oh_trials['trial_type'] == 'offset']
mask = subset['preceding_min_value'].notnull() & subset['normalized_pain_change'].notnull()
x = subset.loc[mask, 'preceding_min_value']
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
plt.savefig(f"offset_trials_preceding_min_value.svg", dpi=300)
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
plt.savefig(f"inv_trials_preceding_min_value.svg", dpi=300)
plt.show()

# Find the first 'offset' trial for each subject
first_offset_trials = (
    oa_oh_trials[oa_oh_trials['trial_type'] == 'offset']
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

# # Scatter plot: preceding_min_value vs. normalized pain change for first OA trial
# mask = first_offset_trials['preceding_min_value'].notnull() & first_offset_trials['normalized_pain_change'].notnull()
# x = first_offset_trials.loc[mask, 'preceding_min_value']
# y = first_offset_trials.loc[mask, 'normalized_pain_change']

# slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
# print(f"First OA trial - preceding_min_value:")
# print(f"  Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, alpha=0.7)
# if p_value < 0.05:
#     x_vals = np.array([x.min(), x.max()])
#     y_vals = intercept + slope * x_vals
#     plt.plot(x_vals, y_vals, color='red')
# plt.title("First 'offset' trial: Normalized pain change vs. preceding Min Value")
# plt.xlabel('Preceding Min Value')
# plt.ylabel('Normalized Pain Change (%)')
# plt.tight_layout()
# plt.show()

# %%
# Correlation between auc_total and max_val for all trials
mask = trial_metrics_df['auc_total'].notnull() & trial_metrics_df['max_val'].notnull()
x = trial_metrics_df.loc[mask, 'auc_total']
y = trial_metrics_df.loc[mask, 'max_val']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Correlation between auc_total and max_val for all trials:")
print(f"  Correlation r: {r_value:.3f}, R^2: {r_value**2:.3f}, p-value: {p_value:.3g}")

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
plt.title("All trials: Max Val vs. AUC Total")
plt.xlabel('AUC Total')
plt.ylabel('Max Val')
plt.tight_layout()
plt.show()
# %%
# PLOT NORMALIZED PAIN CHANGE V MIN / MAX AS A GRID FOR OFFSET AND INV TRIALS
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex='col', sharey='row')

# Define trial types and metrics
trial_types = ['offset', 'inv']
metrics = ['preceding_min_value', 'preceding_max_val']
metric_labels = ['Preceding Min Value', 'Preceding Max Value']

# Color scheme
colors = {'offset': 'blue', 'inv': 'red'}

for row_idx, ttype in enumerate(trial_types):
    subset = oa_oh_trials[oa_oh_trials['trial_type'] == ttype]
    
    for col_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row_idx, col_idx]
        
        # Get data
        mask = subset[metric].notnull() & subset['normalized_pain_change'].notnull()
        x = subset.loc[mask, metric]
        y = subset.loc[mask, 'normalized_pain_change']
        
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
plt.savefig('/userdata/ljohnston/TCE_analysis/data_from_ben/pain_change_vs_preceding_metrics_grid.svg', 
            dpi=300, bbox_inches='tight')
plt.show()

# %%
