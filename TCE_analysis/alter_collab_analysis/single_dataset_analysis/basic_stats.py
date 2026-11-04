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
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/alter_collab_analysis/')
from utils.plotting_functions import plot_trial_comparison, create_correlation_scatter
# Define the dataset
dataset = 'cLBP'  # options: 'plosONE', 'kneeOA', 'cLBP'
# When doing plosONE, be sure to change any "onset" to "inv"
# When doing kneeOA, be sure to change any "inv" to "onset"
# Define dataset-specific trial types and comparisons
if dataset == 'plosONE':
    trial_types = ['inv', 'offset', 't1_hold', 't2_hold', 'stepdown']
    control_trials = ['t1_hold', 't2_hold']
    stepped_trials = ['inv', 'offset', 'stepdown']
    trial_comparisons = [
        ('t1_hold', 'offset', 'max_val', 'T1_Hold vs Offset: Max Value', None),
        ('t1_hold', 'stepdown', 'min_val', 'T1_Hold vs Stepdown: Min Value', None),
        ('t2_hold', 'inv', 'max_val', 'T2_Hold vs Inv: Max Value', None)
    ]
elif dataset == 'kneeOA':
    trial_types = ['onset', 'offset', 't1_hold', 't2_hold', 'innocuous']
    control_trials = ['t1_hold', 't2_hold', 'innocuous']
    stepped_trials = ['onset', 'offset']
    trial_comparisons = [
        ('t1_hold', 'offset', 'max_val', 'T1_Hold vs Offset: Max Value', None),
        ('t2_hold', 'onset', 'max_val', 'T2_Hold vs Onset: Max Value', None),
    ]
elif dataset == 'cLBP':
    trial_types = ['onset', 'offset', 't1_hold', 't2_hold']
    control_trials = ['t1_hold', 't2_hold']
    stepped_trials = ['onset', 'offset']
    trial_comparisons = [
        ('t1_hold', 'offset', 'max_val', 'T1_Hold vs Offset: Max Value', None),
        ('t2_hold', 'onset', 'max_val', 'T2_Hold vs Onset: Max Value', None),
    ]
    
# File paths
TRIAL_METRICS_PATH = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_metrics.json'
TRIAL_DATA_PATH = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_data_cleaned_aligned.json'

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
control_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(control_trials)].copy()
# Create temporary DataFrame for max values
temp_max_df = pd.DataFrame({
    'time_yoked_max_combined': pd.concat([
        control_trials['time_yoked_max_val_offset'], 
        control_trials['time_yoked_max_val_onset'] if dataset in ['kneeOA', 'cLBP'] else control_trials['time_yoked_max_val_inv']
    ]).dropna(),
    'abs_max_combined': pd.concat([
        control_trials.loc[control_trials['time_yoked_max_val_offset'].notna(), 'abs_max_val'],
        control_trials.loc[control_trials['time_yoked_max_val_onset'].notna() if dataset in ['kneeOA', 'cLBP'] else control_trials['time_yoked_max_val_inv'].notna(), 'abs_max_val']
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
        control_trials['time_yoked_min_val_onset'] if dataset in ['kneeOA', 'cLBP'] else control_trials['time_yoked_min_val_inv']
    ]).dropna(),
    'abs_min_combined': pd.concat([
        control_trials.loc[control_trials['time_yoked_min_val_offset'].notna(), 'abs_min_val'],
        control_trials.loc[control_trials['time_yoked_min_val_onset'].notna() if dataset in ['kneeOA', 'cLBP'] else control_trials['time_yoked_min_val_inv'].notna(), 'abs_min_val']
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

############################################################################################ Compare abs_normalized_pain_change in offset and inv trials
# Compare abs_normalized_pain_change between offset and inv trials
offset_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'offset'].copy()
if dataset == 'plosONE':
    inv_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'inv'].copy()
elif dataset in ['kneeOA', 'cLBP']:
    inv_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'onset'].copy()

# Find common subjects
common_subjects = set(offset_trials['subject']) & set(inv_trials['subject'])
offset_data = offset_trials[offset_trials['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True) # average across trials for each subject
inv_data = inv_trials[inv_trials['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True) # average across trials for each subject

vals_offset = offset_data['abs_normalized_pain_change']
vals_inv = inv_data['abs_normalized_pain_change']

data = pd.DataFrame({'offset': vals_offset.abs(), 'onset': vals_inv.abs()}).dropna()

# Paired t-test
tstat, pval = stats.ttest_rel(data['offset'], data['onset'])
print(f"Offset vs Onset (abs_normalized_pain_change): n={len(data)} subjects, t={tstat:.3f}, p={pval:.4f}")

# Plot
plot_data = pd.melt(data.reset_index(), id_vars=['subject'], 
                   value_vars=['offset', 'onset'], 
                   var_name='Trial_Type', value_name='Normalized_Pain_Change')

plt.figure(figsize=(8, 6))
sns.swarmplot(data=plot_data, x='Trial_Type', y='Normalized_Pain_Change', 
              palette=['#4F81BD', '#C0504D'], size=6, alpha=0.8)

# Add connecting lines for paired data
for subject in data.index:
    plt.plot([0, 1], [data.loc[subject, 'offset'], data.loc[subject, 'onset']], 
            'k-', alpha=0.3, linewidth=0.8)

plt.ylabel('Absolute Value of Normalized Pain Change (%)')
plt.title(f'Offset vs Onset: Normalized Pain Change\n(p={pval:.4f})')
plt.tight_layout()
plt.show()

# #%% 
# # ============================================================================
# # Sometimes, there are huge outliers in the time_yoked metrics, so let's look at them
# # ===========================================================================
# for tt in ["t1_hold", "t2_hold"]:
#     sub = trial_metrics_df[trial_metrics_df["trial_type"] == tt].copy()
#     sub["abs_ty"] = sub["time_yoked_normalized_pain_change"].abs()
#     print("\n", tt)
#     print(sub.sort_values("abs_ty", ascending=False)[
#         ["subject", "trial_num", "trial_type", "time_yoked_normalized_pain_change", "abs_normalized_pain_change"]
#     ].head(10))

# # Either loop and plot all trials that have time_yoked_normalized_pain_change > 100
# # Or select a specific subject, trial combo to plot 
# if dataset != 'cLBP': #everyone else is good
#     plot_trial_comparison()
# else: # treat cLBP data different because of the sparse pain ratings
#     plot_trial_comparison_cLBP()





#%%
# ============================================================================
# Replicate basic stats from the paper
# ============================================================================
############################################################################################## Compare t1_hold vs offset/stepdown and t2_hold vs inv for max and min values Like in the paper, using time-yoked min/max
comparisons = trial_comparisons
for trial1, trial2, metric, label, ylims in comparisons:
    df1 = trial_metrics_df[trial_metrics_df['trial_type'] == trial1].copy()
    df2 = trial_metrics_df[trial_metrics_df['trial_type'] == trial2].copy()
    if dataset == 'plosONE':
        metric1 = f'time_yoked_{metric}_{trial2}'  # e.g., 'time_yoked_max_val_offset' or 'time_yoked_min_val_stepdown'
    elif dataset in ['kneeOA', 'cLBP']:
        metric1 = f'control_{metric}_{trial2}' # e.g., 'control_max_val_offset' or 'control_min_val_onset'
    metric2 = f'abs_{metric}'  # e.g., 'abs_max_val' or 'abs_min_val'
    common_subjects = set(df1['subject']) & set(df2['subject'])
    df1 = df1[df1['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
    df2 = df2[df2['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
    
    # Build metric names based on trial type
    if trial1 in ['t1_hold', 't2_hold']:  # Control trials
        if trial1 == 't1_hold' and trial2 in ['offset', 'stepdown']:
            metric1 = f'time_yoked_{metric}_{trial2}'  # e.g., 'time_yoked_min_val_offset'
        elif trial1 == 't2_hold' and trial2 in ['inv','onset']:
            metric1 = f'time_yoked_{metric}_{trial2}'       # e.g., 'time_yoked_max_val_inv'
    
    if trial2 in ['offset', 'stepdown', 'inv', 'onset']:  # Stepped trials
        metric2 = f'abs_{metric}'  # e.g., 'abs_min_val' or 'abs_max_val'
    
    vals1 = df1[metric1]
    vals2 = df2[metric2]
    
    data = pd.DataFrame({trial1: vals1, trial2: vals2}).dropna()
    means = [data[trial1].mean(), data[trial2].mean()]
    sems = [stats.sem(data[trial1]), stats.sem(data[trial2])]
    errors = [[sem, sem] for sem in sems]
    
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

#%%
# ============================================================================
# Compare latency across trial types
# ============================================================================
############################################################################################ Compare latency to max pain across trial types
def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

latency_comparisons = [
    ('t1_hold', 'offset', 'Latency to Max Pain: T1_Hold vs Offset'),
    ('t2_hold', 'inv', 'Latency to Max Pain: T2_Hold vs Inv') if dataset == 'plosONE' else ('t2_hold', 'onset', 'Latency to Max Pain: T2_Hold vs Onset'),
    ('t1_hold', 't2_hold', 'Latency to Max Pain: T1_Hold vs T2_Hold')
]

for trial1, trial2, label in latency_comparisons:
    df1 = trial_metrics_df[trial_metrics_df['trial_type'] == trial1].copy()
    df2 = trial_metrics_df[trial_metrics_df['trial_type'] == trial2].copy()

    common_subjects = set(df1['subject']) & set(df2['subject'])
    df1 = df1[df1['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
    df2 = df2[df2['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)

    vals1 = df1['abs_max_time']
    vals2 = df2['abs_max_time']

    data = pd.DataFrame({trial1: vals1, trial2: vals2}).dropna()

    # Paired t-test
    tstat, pval = stats.ttest_rel(data[trial1], data[trial2])
    stars = get_significance_stars(pval)
    
    print(f"{label}: n={len(data)} subjects, t={tstat:.3f}, p={pval:.4f} {stars}")

    # Reshape for seaborn
    plot_data = pd.melt(data.reset_index(), id_vars=['subject'], 
                       value_vars=[trial1, trial2], 
                       var_name='Trial_Type', value_name='Latency')

    # Violin plot
    plt.figure(figsize=(8, 6))
    
    ax = sns.violinplot(data=plot_data, x='Trial_Type', y='Latency', 
                       palette=['#4F81BD', '#C0504D'], inner=None)
    
    sns.stripplot(data=plot_data, x='Trial_Type', y='Latency', 
                 color='black', size=4, alpha=0.7)
    
    # Add connecting lines for paired data
    for subject in data.index:
        plt.plot([0, 1], [data.loc[subject, trial1], data.loc[subject, trial2]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    # Add significance stars
    if stars != 'ns':
        y_max = plot_data['Latency'].max()
        y_range = plot_data['Latency'].max() - plot_data['Latency'].min()
        plt.text(0.5, y_max + 0.05 * y_range, stars, ha='center', va='bottom', 
                fontsize=20, fontweight='bold')
        plt.plot([0, 1], [y_max + 0.02 * y_range, y_max + 0.02 * y_range], 
                'k-', linewidth=1)
    
    plt.ylabel('Latency to Max Pain (s)')
    plt.title(f"{label}\n(p={pval:.4f})")
    plt.tight_layout()
    plt.show()

# %%
# ======================================================================================
# Compare if abs_normalized_pain_change in offset trials is proportional to inv trials
# ======================================================================================
# Get offset and inv trials for comparison
offset_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'offset'].copy()
if dataset == 'plosONE':
    inv_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'inv'].copy()
else:
    inv_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'onset'].copy()

# Find common subjects between offset and inv trials
common_subjects = set(offset_trials['subject']) & set(inv_trials['subject'])
print(f"Found {len(common_subjects)} subjects with both offset and inv trials")

# Group by subject and get mean values
offset_data = offset_trials[offset_trials['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
inv_data = inv_trials[inv_trials['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)

# Get the normalized pain change values
offset_norm_change = offset_data['abs_normalized_pain_change']
inv_norm_change = inv_data['abs_normalized_pain_change']

# Create combined dataframe
comparison_data = pd.DataFrame({
    'offset_norm_change': offset_norm_change,
    'inv_norm_change': inv_norm_change
}).dropna()

print(f"Final analysis includes {len(comparison_data)} subjects")

# Correlation analysis
r, p_value = stats.pearsonr(comparison_data['offset_norm_change'], comparison_data['inv_norm_change'])
print(f"\nCorrelation Analysis:")
print(f"Pearson r = {r:.4f}, p = {p_value:.4f}")

# Linear regression to see if relationship passes through origin (proportional)
from scipy.stats import linregress
slope, intercept, r_value, p_value_reg, std_err = linregress(
    comparison_data['offset_norm_change'], comparison_data['inv_norm_change'])

print(f"\nLinear Regression:")
print(f"Slope = {slope:.4f}")
print(f"Intercept = {intercept:.4f}")
print(f"R² = {r_value**2:.4f}")
print(f"p-value = {p_value_reg:.4f}")

# Test if intercept is significantly different from 0 (for true proportionality)
# For proportionality, we expect intercept ≈ 0
t_stat_intercept = intercept / std_err
p_intercept = 2 * (1 - stats.t.cdf(abs(t_stat_intercept), len(comparison_data) - 2))
print(f"Intercept t-test (H0: intercept = 0): t = {t_stat_intercept:.4f}, p = {p_intercept:.4f}")

# Create scatter plot with regression line
plt.figure(figsize=(10, 8))
plt.scatter(comparison_data['offset_norm_change'], comparison_data['inv_norm_change'], 
           alpha=0.7, s=60, color='#4F81BD')
x_range = np.linspace(comparison_data['offset_norm_change'].min(), 
                     comparison_data['offset_norm_change'].max(), 100)
y_pred = slope * x_range + intercept
plt.plot(x_range, y_pred, 'r-', linewidth=2, 
         label=f'y = {slope:.3f}x + {intercept:.3f}')
# Add line through origin for comparison (true proportionality)
y_prop = slope * x_range  # Force through origin
plt.plot(x_range, y_prop, 'g--', linewidth=2, alpha=0.7,
         label=f'Proportional: y = {slope:.3f}x')
# Add identity line (1:1 relationship)
max_val = max(comparison_data['offset_norm_change'].max(), 
              comparison_data['inv_norm_change'].max())
plt.plot([0, max_val], [0, max_val], 'k:', linewidth=1, alpha=0.5, label='y = x')
plt.xlim(-110, 0)
plt.xlabel('Offset Trials: Normalized Pain Change (%)')
plt.ylabel('Inv Trials: Normalized Pain Change (%)')
plt.title(f'Proportionality Test: Offset vs Inv Normalized Pain Change\n'
          f'r = {r:.4f}, p = {p_value:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
# Add text box with key statistics
textstr = f'n = {len(comparison_data)}\nr = {r:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()


# Plot ratio distribution
comparison_data['ratio'] = comparison_data['inv_norm_change'] / comparison_data['offset_norm_change']
plt.figure(figsize=(8, 6))
plt.hist(comparison_data['ratio'].dropna(), bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(comparison_data['ratio'].mean(), color='red', linestyle='--', 
           label=f'Mean = {comparison_data['ratio'].mean():.3f}')
plt.xlabel('Ratio (Inv/Offset)')
plt.ylabel('Frequency')
plt.title('Distribution of Inv/Offset Ratios Across Subjects')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Compare Peak to Peak Values
# Compare abs_peak_to_peak between offset and onset trials
offset_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'offset'].copy()
onset_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'onset'].copy()

vals_offset = offset_trials['abs_peak_to_peak']
vals_onset = onset_trials['abs_peak_to_peak']

data = pd.DataFrame({'offset': vals_offset, 'onset': vals_onset}).dropna()

# Paired t-test
tstat, pval = stats.ttest_rel(data['offset'], data['onset'])
print(f"Peak-to-Peak Comparison (Offset vs Onset): n={len(data)} subjects, t={tstat:.3f}, p={pval:.4f}")

# Correlation
r, p_corr = stats.pearsonr(data['offset'], data['onset'])
print(f"Correlation: r = {r:.4f}, p = {p_corr:.4f}")

# Quick scatter plot to see correlation
plt.figure(figsize=(8, 6))
plt.scatter(data['offset'], data['onset'], alpha=0.7, s=60, color='#4F81BD')
plt.plot([0, data[['offset', 'onset']].max().max()], 
         [0, data[['offset', 'onset']].max().max()], 'k--', alpha=0.5, label='y=x')
plt.xlabel('Offset Peak-to-Peak')
plt.ylabel('Onset Peak-to-Peak')
plt.title(f'Peak-to-Peak Correlation: Offset vs Onset\n(r={r:.4f}, p={p_corr:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
