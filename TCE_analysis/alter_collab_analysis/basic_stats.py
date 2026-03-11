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
# Define the dataset
dataset = 'kneeOA'  # options: 'plosONE', 'kneeOA'
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
        control_trials['time_yoked_max_val_onset'] if dataset == 'kneeOA' else control_trials['time_yoked_max_val_inv']
    ]).dropna(),
    'abs_max_combined': pd.concat([
        control_trials.loc[control_trials['time_yoked_max_val_offset'].notna(), 'abs_max_val'],
        control_trials.loc[control_trials['time_yoked_max_val_onset'].notna() if dataset == 'kneeOA' else control_trials['time_yoked_max_val_inv'].notna(), 'abs_max_val']
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
        control_trials['time_yoked_min_val_onset'] if dataset == 'kneeOA' else control_trials['time_yoked_min_val_inv']
    ]).dropna(),
    'abs_min_combined': pd.concat([
        control_trials.loc[control_trials['time_yoked_min_val_offset'].notna(), 'abs_min_val'],
        control_trials.loc[control_trials['time_yoked_min_val_onset'].notna() if dataset == 'kneeOA' else control_trials['time_yoked_min_val_inv'].notna(), 'abs_min_val']
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
elif dataset == 'kneeOA':
    inv_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'onset'].copy()

# Find common subjects
common_subjects = set(offset_trials['subject']) & set(inv_trials['subject'])
offset_data = offset_trials[offset_trials['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)
inv_data = inv_trials[inv_trials['subject'].isin(common_subjects)].groupby('subject').mean(numeric_only=True)

vals_offset = offset_data['abs_normalized_pain_change']
vals_inv = inv_data['abs_normalized_pain_change']

data = pd.DataFrame({'offset': vals_offset, 'onset': vals_inv}).dropna()

# Paired t-test
tstat, pval = stats.ttest_rel(data['offset'], data['onset'])
print(f"Offset vs Onset (abs_normalized_pain_change): n={len(data)} subjects, t={tstat:.3f}, p={pval:.4f}")

# Plot (choose swarm or violin)
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

plt.ylabel('Absolute Normalized Pain Change (%)')
plt.title(f'Offset vs Onset: Normalized Pain Change\n(p={pval:.4f})')
plt.tight_layout()
plt.show()


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
    elif dataset == 'kneeOA':
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
# Compare abs_peak_to_peak between offset and inv trials
offset_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'offset'].copy()
inv_trials = trial_metrics_df[trial_metrics_df['trial_type'] == 'inv'].copy()

# Find common subjects
vals_offset = offset_data['abs_peak_to_peak']
vals_inv = inv_data['abs_peak_to_peak']

data = pd.DataFrame({'offset': vals_offset, 'inv': vals_inv}).dropna()

# Paired t-test
tstat, pval = stats.ttest_rel(data['offset'], data['inv'])
print(f"Peak-to-Peak Comparison (Offset vs Inv): n={len(data)} subjects, t={tstat:.3f}, p={pval:.4f}")

# Correlation
r, p_corr = stats.pearsonr(data['offset'], data['inv'])
print(f"Correlation: r = {r:.4f}, p = {p_corr:.4f}")

# Quick scatter plot to see correlation
plt.figure(figsize=(8, 6))
plt.scatter(data['offset'], data['inv'], alpha=0.7, s=60, color='#4F81BD')
plt.plot([0, data[['offset', 'inv']].max().max()], 
         [0, data[['offset', 'inv']].max().max()], 'k--', alpha=0.5, label='y=x')
plt.xlabel('Offset Peak-to-Peak')
plt.ylabel('Inv Peak-to-Peak')
plt.title(f'Peak-to-Peak Correlation: Offset vs Inv\n(r={r:.4f}, p={p_corr:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
"""
EVERYTHING BELOW STRAIGHT FROM CLAUDE PLEASE VERIFY
"""
# ============================================================================
# TRIAL ORDER EFFECTS ANALYSIS - Linear Mixed Effects Models
# ============================================================================

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRIAL ORDER EFFECTS ANALYSIS")
print("=" * 80)

# Prepare data for LME analysis
lme_data = trial_metrics_df.copy()

# Ensure trial_num is numeric and centered (helps with interpretation)
lme_data['trial_num_centered'] = lme_data['trial_num'] - lme_data['trial_num'].mean()

# Create a combined dataset for stepped trials (offset + inv)
control_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(control_trials)].copy()
stepped_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(stepped_trials)].copy()

print(f"Data summary:")
print(f"- Total trials: {len(lme_data)}")
print(f"- Stepped trials (offset + inv): {len(stepped_trials)}")
print(f"- Control trials (t1_hold + t2_hold): {len(control_trials)}")
print(f"- Subjects: {lme_data['subject'].nunique()}")
print(f"- Trial range: {lme_data['trial_num'].min()} to {lme_data['trial_num'].max()}")

######################################################################################### Trial order effects on primary pain metrics

print("\n" + "="*60)
print("MODEL 1: TRIAL ORDER EFFECTS ON PAIN INTENSITY")
print("="*60)

# Test different pain metrics
pain_metrics = ['abs_max_val', 'abs_normalized_pain_change', 'auc_total']

for metric in pain_metrics:
    print(f"\n--- {metric.upper()} ---")
    
    # Model for stepped trials only (offset + inv)
    if metric in stepped_trials.columns:
        try:
            # Full model: trial_num * trial_type + random intercept + random slope
            model = mixedlm(f"{metric} ~ trial_num_centered * trial_type", 
                          data=stepped_trials,
                          groups=stepped_trials["subject"],
                          re_formula="~trial_num_centered")
            
            result = model.fit(reml=False)
            
            print(f"Model: {metric} ~ trial_num_centered * trial_type + (1 + trial_num_centered | subject)")
            print(f"AIC: {result.aic:.2f}")
            
            # Extract key results
            params = result.params
            pvalues = result.pvalues
            
            print(f"Fixed Effects:")
            for param in ['trial_num_centered', 'trial_type[T.offset]', 
                         'trial_num_centered:trial_type[T.offset]']:
                if param in params.index:
                    coef = params[param]
                    p = pvalues[param]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"  {param}: β = {coef:.4f}, p = {p:.4f} {sig}")
            
            # Test for overall trial order effect (combining both trial types)
            if 'trial_num_centered' in pvalues.index:
                trial_effect_p = pvalues['trial_num_centered']
                if trial_effect_p < 0.05:
                    direction = "increases" if params['trial_num_centered'] > 0 else "decreases"
                    print(f"  → {metric} {direction} over trials (p = {trial_effect_p:.4f})")
            
            # Test for interaction
            interaction_param = 'trial_num_centered:trial_type[T.offset]'
            if interaction_param in pvalues.index:
                interaction_p = pvalues[interaction_param]
                if interaction_p < 0.05:
                    print(f"  → Different trial order effects between offset/inv (p = {interaction_p:.4f})")
                    
        except Exception as e:
            print(f"Model failed for {metric}: {e}")

######################################################################################### Trial order effects on latency measures
print("\n" + "="*60)
print("MODEL 2: TRIAL ORDER EFFECTS ON LATENCY (WITHIN TRIAL TYPE)")
print("="*60)

latency_metrics = ['abs_max_time', 'abs_min_time']

for metric in latency_metrics:
    if metric in stepped_trials.columns:
        print(f"\n--- {metric.upper()} ---")
        
        # Test each trial type separately for trial order effects
        for trial_type in ['offset', 'inv']:
            subset = stepped_trials[stepped_trials['trial_type'] == trial_type]
            
            if len(subset) > 10:  # Need sufficient data
                try:
                    # Model: latency ~ trial_num + (1 + trial_num | subject)
                    model = mixedlm(f"{metric} ~ trial_num_centered", 
                                  data=subset,
                                  groups=subset["subject"],
                                  re_formula="~trial_num_centered")
                    
                    result = model.fit(reml=False)
                    
                    params = result.params
                    pvalues = result.pvalues
                    
                    print(f"\n  {trial_type.upper()} trials (n={len(subset)} trials, {subset['subject'].nunique()} subjects):")
                    
                    if 'trial_num_centered' in pvalues.index:
                        trial_effect_p = pvalues['trial_num_centered']
                        trial_effect_coef = params['trial_num_centered']
                        sig = "***" if trial_effect_p < 0.001 else "**" if trial_effect_p < 0.01 else "*" if trial_effect_p < 0.05 else "ns"
                        
                        print(f"    Trial order effect: β = {trial_effect_coef:.4f}, p = {trial_effect_p:.4f} {sig}")
                        
                        if trial_effect_p < 0.05:
                            direction = "increases" if trial_effect_coef > 0 else "decreases"
                            print(f"    → {metric} {direction} over trials in {trial_type}")
                        else:
                            print(f"    → No significant trial order effect in {trial_type}")
                    
                    # Simple correlation as backup
                    r, p_corr = stats.pearsonr(subset['trial_num'], subset[metric])
                    print(f"    Simple correlation: r = {r:.3f}, p = {p_corr:.4f}")
                            
                except Exception as e:
                    print(f"    Model failed for {trial_type}: {e}")
            else:
                print(f"\n  {trial_type.upper()}: Insufficient data (n={len(subset)})")

# Optional: Test if trial order effects on latency differ between trial types
print(f"\n--- COMPARING TRIAL ORDER EFFECTS BETWEEN OFFSET AND INV ---")
try:
    # This tests the interaction: does trial order affect latency differently in offset vs inv?
    model = mixedlm("abs_max_time ~ trial_num_centered * trial_type", 
                    data=stepped_trials,
                    groups=stepped_trials["subject"])
    
    result = model.fit(reml=False)
    params = result.params
    pvalues = result.pvalues
    
    interaction_param = 'trial_num_centered:trial_type[T.offset]'
    if interaction_param in pvalues.index:
        interaction_p = pvalues[interaction_param]
        interaction_coef = params[interaction_param]
        sig = "***" if interaction_p < 0.001 else "**" if interaction_p < 0.01 else "*" if interaction_p < 0.05 else "ns"
        
        print(f"Trial order × trial type interaction: β = {interaction_coef:.4f}, p = {interaction_p:.4f} {sig}")
        
        if interaction_p < 0.05:
            print(f"→ Trial order effects on latency differ between offset and inv trials")
        else:
            print(f"→ Trial order effects on latency are similar between offset and inv trials")
    
except Exception as e:
    print(f"Interaction model failed: {e}")

######################################################################################### Compare trial order effects between stepped vs control trials

print("\n" + "="*60)
print("MODEL 3: STEPPED vs CONTROL TRIAL ORDER EFFECTS")
print("="*60)

# Create binary indicator for stepped vs control
lme_data['is_stepped'] = lme_data['trial_type'].isin(['offset', 'inv']).astype(int)

try:
    # Model abs_max_val across all trial types
    model = mixedlm("abs_max_val ~ trial_num_centered * is_stepped", 
                    data=lme_data,
                    groups=lme_data["subject"])
    
    result = model.fit(reml=False)
    
    params = result.params
    pvalues = result.pvalues
    
    print("Testing if trial order effects differ between stepped and control trials:")
    
    interaction_param = 'trial_num_centered:is_stepped'
    if interaction_param in pvalues.index:
        interaction_p = pvalues[interaction_param]
        interaction_coef = params[interaction_param]
        sig = "***" if interaction_p < 0.001 else "**" if interaction_p < 0.01 else "*" if interaction_p < 0.05 else "ns"
        
        print(f"  Trial order × Trial type interaction: β = {interaction_coef:.4f}, p = {interaction_p:.4f} {sig}")
        
        if interaction_p < 0.05:
            if interaction_coef > 0:
                print(f"  → Stepped trials show GREATER trial order effects than control trials")
            else:
                print(f"  → Stepped trials show SMALLER trial order effects than control trials")
        else:
            print(f"  → No difference in trial order effects between stepped and control trials")
    
    # Main effects
    if 'trial_num_centered' in pvalues.index:
        main_p = pvalues['trial_num_centered']
        main_coef = params['trial_num_centered']
        if main_p < 0.05:
            direction = "increases" if main_coef > 0 else "decreases"
            print(f"  → Overall: pain intensity {direction} over trials (p = {main_p:.4f})")

except Exception as e:
    print(f"Model failed: {e}")

# ============================================================================
# Visualization: Trial order effects
# ============================================================================

print("\n" + "="*60)
print("VISUALIZING TRIAL ORDER EFFECTS")
print("="*60)

# Plot 1: Pain intensity over trial number by trial type
plt.figure(figsize=(12, 8))

trial_types = ['offset', 'inv', 't1_hold', 't2_hold']
colors = ['#C0504D', '#9BBB59', '#4F81BD', '#8064A2']

for i, trial_type in enumerate(trial_types):
    subset = lme_data[lme_data['trial_type'] == trial_type]
    
    if len(subset) > 0:
        # Calculate mean and SEM for each trial number
        trial_summary = subset.groupby('trial_num').agg({
            'abs_max_val': ['mean', 'sem', 'count']
        }).round(3)
        
        trial_summary.columns = ['mean', 'sem', 'count']
        trial_summary = trial_summary[trial_summary['count'] >= 3]  # Only plot if ≥3 subjects
        
        if len(trial_summary) > 0:
            x = trial_summary.index
            y = trial_summary['mean']
            yerr = trial_summary['sem']
            
            plt.errorbar(x, y, yerr=yerr, marker='o', linestyle='-', 
                        color=colors[i], label=f'{trial_type} (n={subset["subject"].nunique()})',
                        capsize=3, capthick=1, linewidth=2, markersize=6)

plt.xlabel('Trial Number')
plt.ylabel('Pain Intensity (Max Value)')
plt.title('Pain Intensity Across Trial Order by Trial Type')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Individual subject trajectories for stepped trials
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
# Offset trials
offset_data = stepped_trials[stepped_trials['trial_type'] == 'offset']
for subject in offset_data['subject'].unique()[:10]:  # Show first 10 subjects
    subj_data = offset_data[offset_data['subject'] == subject]
    if len(subj_data) > 1:  # Only if multiple trials
        plt.plot(subj_data['trial_num'], subj_data['abs_max_val'], 
                'o-', alpha=0.6, color='#C0504D', linewidth=1, markersize=4)

plt.xlabel('Trial Number')
plt.ylabel('Pain Intensity (Max Value)')
plt.title('Individual Trajectories: Offset Trials')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Inv trials
inv_data = stepped_trials[stepped_trials['trial_type'] == 'inv']
for subject in inv_data['subject'].unique()[:10]:  # Show first 10 subjects
    subj_data = inv_data[inv_data['subject'] == subject]
    if len(subj_data) > 1:  # Only if multiple trials
        plt.plot(subj_data['trial_num'], subj_data['abs_max_val'], 
                'o-', alpha=0.6, color='#9BBB59', linewidth=1, markersize=4)

plt.xlabel('Trial Number')
plt.ylabel('Pain Intensity (Max Value)')
plt.title('Individual Trajectories: Inv Trials')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# Summary statistics by trial number
# ============================================================================

print("\n" + "="*60)
print("SUMMARY: TRIAL ORDER PATTERNS")
print("="*60)

# Calculate correlations between trial number and pain metrics for each trial type
for trial_type in ['offset', 'inv', 't1_hold', 't2_hold']:
    subset = lme_data[lme_data['trial_type'] == trial_type]
    
    if len(subset) > 10:  # Only if sufficient data
        # Correlation with trial number
        r_max, p_max = stats.pearsonr(subset['trial_num'], subset['abs_max_val'])
        
        print(f"\n{trial_type.upper()}:")
        print(f"  Trial num vs max pain: r = {r_max:.3f}, p = {p_max:.4f}")
        
        # Test for linear trend using simple regression
        from scipy.stats import linregress
        slope, intercept, r_val, p_val, std_err = linregress(subset['trial_num'], subset['abs_max_val'])
        
        if p_val < 0.05:
            direction = "increases" if slope > 0 else "decreases"
            print(f"  → Significant trend: pain {direction} by {abs(slope):.3f} units per trial")
        else:
            print(f"  → No significant linear trend")
        
        # Check if early vs late trials differ
        early_trials = subset[subset['trial_num'] <= subset['trial_num'].median()]
        late_trials = subset[subset['trial_num'] > subset['trial_num'].median()]
        
        if len(early_trials) > 0 and len(late_trials) > 0:
            t_stat, p_early_late = stats.ttest_ind(early_trials['abs_max_val'], 
                                                   late_trials['abs_max_val'])
            if p_early_late < 0.05:
                direction = "higher" if late_trials['abs_max_val'].mean() > early_trials['abs_max_val'].mean() else "lower"
                print(f"  → Late trials have {direction} pain than early trials (p = {p_early_late:.4f})")

# ============================================================================
# Alternative: rmANOVA approach (for comparison)
# ============================================================================

print("\n" + "="*60)
print("ALTERNATIVE: rmANOVA APPROACH")
print("="*60)

try:
    from statsmodels.stats.anova import AnovaRM
    
    # Create trial number bins for rmANOVA (since rmANOVA needs categorical factors)
    lme_data['trial_bin'] = pd.cut(lme_data['trial_num'], 
                                   bins=3, 
                                   labels=['Early', 'Middle', 'Late'])
    
    # Filter to subjects with data in all bins and trial types of interest
    stepped_complete = stepped_trials.dropna(subset=['abs_max_val', 'trial_bin'])
    
    # Check if we have enough data for rmANOVA
    subject_counts = stepped_complete.groupby(['subject', 'trial_type', 'trial_bin']).size().unstack(fill_value=0)
    
    if len(subject_counts) > 0:
        print("rmANOVA Analysis:")
        print("(Note: This treats trial order as categorical bins rather than continuous)")
        
        # Prepare data for rmANOVA - need balanced design
        anova_data = []
        for subject in stepped_complete['subject'].unique():
            subj_data = stepped_complete[stepped_complete['subject'] == subject]
            
            # Check if subject has both trial types and multiple trial bins
            trial_types_present = subj_data['trial_type'].unique()
            trial_bins_present = subj_data['trial_bin'].dropna().unique()
            
            if len(trial_types_present) >= 2 and len(trial_bins_present) >= 2:
                for _, row in subj_data.iterrows():
                    if pd.notna(row['trial_bin']):
                        anova_data.append({
                            'subject': row['subject'],
                            'trial_type': row['trial_type'],
                            'trial_bin': row['trial_bin'],
                            'abs_max_val': row['abs_max_val']
                        })
        
        if len(anova_data) > 20:  # Need sufficient data
            anova_df = pd.DataFrame(anova_data)
            
            try:
                # rmANOVA: trial_bin * trial_type
                aovrm = AnovaRM(anova_df, 'abs_max_val', 'subject', 
                               within=['trial_bin', 'trial_type'])
                res = aovrm.fit()
                
                print("\nrmANOVA Results:")
                print(res.summary())
                
                # Extract key p-values
                anova_table = res.anova_table
                
                if 'trial_bin' in anova_table.index:
                    trial_p = anova_table.loc['trial_bin', 'Pr > F']
                    print(f"\nMain effect of trial order (binned): p = {trial_p:.4f}")
                    if trial_p < 0.05:
                        print("  → Significant trial order effect detected!")
                
                if 'trial_bin:trial_type' in anova_table.index:
                    interaction_p = anova_table.loc['trial_bin:trial_type', 'Pr > F']
                    print(f"Trial order × trial type interaction: p = {interaction_p:.4f}")
                    if interaction_p < 0.05:
                        print("  → Trial order effects differ between offset and inv!")
                
            except Exception as e:
                print(f"rmANOVA failed: {e}")
        else:
            print("Insufficient balanced data for rmANOVA")
    
except ImportError:
    print("rmANOVA requires statsmodels. Install with: pip install statsmodels")
except Exception as e:
    print(f"rmANOVA analysis failed: {e}")

# ============================================================================
# Final Summary and Recommendations
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY: TRIAL ORDER EFFECTS")
print("="*80)

print("\nKey Questions Addressed:")
print("1. Do pain ratings change systematically across trial presentations?")
print("2. Are trial order effects different between stepped (offset/inv) and control trials?")
print("3. Do offset and inv trials show similar patterns of change over time?")
print("4. Are there individual differences in trial order effects?")

print("\nMethodological Notes:")
print("- LME models are preferred over rmANOVA for this analysis because:")
print("  * Can handle unbalanced data (different numbers of trials per subject)")
print("  * Treats trial number as continuous (more powerful)")
print("  * Can model both random intercepts and slopes")
print("  * Better handles missing data")
print("  * More flexible for complex designs")

print("\n- The original paper averaged across trials within subjects, which:")
print("  * Eliminates ability to detect trial order effects")
print("  * May mask important habituation/sensitization patterns")
print("  * Reduces statistical power for detecting individual differences")

print("\nInterpretation Guide:")
print("- Negative trial_num coefficient = habituation (pain decreases over trials)")
print("- Positive trial_num coefficient = sensitization (pain increases over trials)")
print("- Significant interaction = different trial order effects between conditions")
print("- Random slopes variance = individual differences in trial order effects")

print("\nNext Steps:")
print("- If significant trial order effects found, consider:")
print("  * Including trial number as covariate in main analyses")
print("  * Investigating mechanisms (fatigue, learning, etc.)")
print("  * Examining if effects are linear or non-linear")
print("  * Testing if effects interact with individual difference measures")
# %%
