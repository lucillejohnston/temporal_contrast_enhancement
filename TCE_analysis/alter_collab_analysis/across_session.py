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
sys.path.append('/userdata/ljohnston/TCE_analysis/data_from_ben')
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_metrics.csv'
TRIAL_DATA_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data_cleaned_aligned.json'

# Load the metrics data
trial_metrics_df = pd.read_csv(TRIAL_METRICS_PATH)

# Load the raw trial data (for time series plotting)
df = pd.read_json(TRIAL_DATA_PATH, orient='records')
#%%
# %%
# ============================================================================
# Plot Metrics Across Trials within Subjects
# ============================================================================

############################################################################# Plot Across Trials per Subject within a Session
trial_types = trial_metrics_df['trial_type'].unique()
main_metrics = ['auc_A', 'auc_B', 'auc_C', 'auc_total', 'abs_peak_to_peak', 'abs_normalized_pain_change']
for metric in main_metrics:
    for trial_type in trial_types:
        plt.figure(figsize=(10, 5))
        df_type = trial_metrics_df[trial_metrics_df['trial_type'] == trial_type]
        for subject, subj_df in df_type.groupby('subject'):
            subj_df = subj_df.sort_values('trial_num')
            if subj_df.empty:
                continue
            first_val = subj_df[metric].iloc[0] if not subj_df[metric].dropna().empty else np.nan
            last_val = subj_df[metric].iloc[-1] if not subj_df[metric].dropna().empty else np.nan
            if pd.isna(first_val) or pd.isna(last_val):
                color = 'gray'
            else:
                color = 'red' if last_val >= first_val else 'blue'
            plt.plot(subj_df['trial_num'], subj_df[metric], marker='o', linestyle='-', color=color, alpha=0.7)
        plt.title(f"{metric} across trials for {trial_type}")
        plt.xlabel('Trial Order')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()

############################################################################## Combining specific metrics across comparable trial types
# Define your groupings
group1 = ['t1_hold', 'offset']
group2 = ['t2_hold', 'inv', 'stepdown']

# Plot group 1
plt.figure(figsize=(10, 5))
for g in group1:
    df_g = trial_metrics_df[trial_metrics_df['trial_type'] == g]
    for subject, subj_df in df_g.groupby('subject'):
        subj_df = subj_df.sort_values('trial_num')
        if subj_df.empty:
            continue
        first_val = subj_df['auc_A'].iloc[0] if not subj_df['auc_A'].dropna().empty else np.nan
        last_val = subj_df['auc_A'].iloc[-1] if not subj_df['auc_A'].dropna().empty else np.nan
        if pd.isna(first_val) or pd.isna(last_val):
            color = 'gray'
        else:
            color = 'red' if last_val >= first_val else 'blue'
        plt.plot(subj_df['trial_num'], subj_df['auc_A'], marker='o', linestyle='-', color=color, alpha=0.7)
plt.title("auc_A over trial order for t1_hold and offset")
plt.xlabel("Trial Order")
plt.ylabel("auc_A")
plt.tight_layout()
plt.show()

# Plot group 2
plt.figure(figsize=(10, 5))
for g in group2:
    df_g = trial_metrics_df[trial_metrics_df['trial_type'] == g]
    for subject, subj_df in df_g.groupby('subject'):
        subj_df = subj_df.sort_values('trial_num')
        if subj_df.empty:
            continue
        first_val = subj_df['auc_A'].iloc[0] if not subj_df['auc_A'].dropna().empty else np.nan
        last_val = subj_df['auc_A'].iloc[-1] if not subj_df['auc_A'].dropna().empty else np.nan
        if pd.isna(first_val) or pd.isna(last_val):
            color = 'gray'
        else:
            color = 'red' if last_val >= first_val else 'blue'
        plt.plot(subj_df['trial_num'], subj_df['auc_A'], marker='o', linestyle='-', color=color, alpha=0.7)
plt.title("auc_A over trial order for t2_hold, inv, and stepdown")
plt.xlabel("Trial Order")
plt.ylabel("auc_A")
plt.tight_layout()
plt.show()


############################################################################## Compare subject-level color assignments between groups
group1 = ['t1_hold', 'offset']
group2 = ['t2_hold', 'inv', 'stepdown']
# Compute color (red if increasing, blue if decreasing) for each subject in group 1
group1_colors = {}
for subject, subj_df in trial_metrics_df[trial_metrics_df['trial_type'].isin(group1)].groupby('subject'):
    subj_df = subj_df.sort_values('trial_num')
    first_val = subj_df.iloc[0]['auc_A']
    last_val = subj_df.iloc[-1]['auc_A']
    group1_colors[subject] = 'red' if last_val >= first_val else 'blue'

# Compute color for each subject in group 2
group2_colors = {}
for subject, subj_df in trial_metrics_df[trial_metrics_df['trial_type'].isin(group2)].groupby('subject'):
    subj_df = subj_df.sort_values('trial_num')
    first_val = subj_df.iloc[0]['auc_A']
    last_val = subj_df.iloc[-1]['auc_A']
    group2_colors[subject] = 'red' if last_val >= first_val else 'blue'

# Find subjects that appear in both groups
common_subjects = set(group1_colors.keys()) & set(group2_colors.keys())

# Compare colors for shared subjects
for subject in common_subjects:
    color1 = group1_colors[subject]
    color2 = group2_colors[subject]
    match = color1 == color2
    
both_increase = [s for s in common_subjects if group1_colors[s] == 'red' and group2_colors[s] == 'red']
both_decrease = [s for s in common_subjects if group1_colors[s] == 'blue' and group2_colors[s] == 'blue']
discordant = [s for s in common_subjects if group1_colors[s] != group2_colors[s]]
print(f"Number of subjects with both groups increasing: {len(both_increase)}")
print(f"Number of subjects with both groups decreasing: {len(both_decrease)}")
print(f"Number of subjects with discordant colors: {len(discordant)}")
# Increase in group1, decrease in group2
inc1_dec2 = [s for s in discordant if group1_colors[s] == 'red' and group2_colors[s] == 'blue']

# Decrease in group1, increase in group2
dec1_inc2 = [s for s in discordant if group1_colors[s] == 'blue' and group2_colors[s] == 'red']
print(f"Increase in group1, decrease in group2: {len(inc1_dec2)}")
print(f"Decrease in group1, increase in group2: {len(dec1_inc2)}")

############################################################################## Statistical comparisons between trial types for auc_C
from scipy.stats import ttest_rel

# --- OFFSET vs T1_HOLD ---
df_offset = trial_metrics_df[trial_metrics_df['trial_type'] == 'offset']
df_t1_hold = trial_metrics_df[trial_metrics_df['trial_type'] == 't1_hold']

df_offset_avg = df_offset.groupby('subject')['auc_C'].mean().reset_index()
df_t1_hold_avg = df_t1_hold.groupby('subject')['auc_C'].mean().reset_index()

# Merge to align subjects and drop NaNs
df_compare1 = pd.merge(df_offset_avg, df_t1_hold_avg, on='subject', suffixes=('_offset', '_t1_hold')).dropna()
vals_offset = df_compare1['auc_C_offset'].values
vals_t1_hold = df_compare1['auc_C_t1_hold'].values

tstat_1, pval_1 = ttest_rel(vals_offset, vals_t1_hold)
print(f"offset vs t1_hold auc_C: n={len(df_compare1)} subjects, t={tstat_1:.3f}, p={pval_1:.4f}")

means = [vals_offset.mean(), vals_t1_hold.mean()]
errors = [vals_offset.std(ddof=1)/len(vals_offset)**0.5, vals_t1_hold.std(ddof=1)/len(vals_t1_hold)**0.5]
plt.bar([0, 1], means, yerr=errors, capsize=8, color=['#4F81BD', '#C0504D'])
plt.xticks([0, 1], ['offset', 't1_hold'])
plt.ylabel('auc_C')
plt.title(f'offset vs t1_hold auc_C (p={pval_1:.4f})')
plt.tight_layout()
plt.show()

# --- INV vs T2_HOLD ---
df_inv = trial_metrics_df[trial_metrics_df['trial_type'] == 'inv']
df_t2_hold = trial_metrics_df[trial_metrics_df['trial_type'] == 't2_hold']

df_inv_avg = df_inv.groupby('subject')['auc_C'].mean().reset_index()
df_t2_hold_avg = df_t2_hold.groupby('subject')['auc_C'].mean().reset_index()

df_compare2 = pd.merge(df_inv_avg, df_t2_hold_avg, on='subject', suffixes=('_inv', '_t2_hold')).dropna()
vals_inv = df_compare2['auc_C_inv'].values
vals_t2_hold = df_compare2['auc_C_t2_hold'].values

tstat_2, pval_2 = ttest_rel(vals_inv, vals_t2_hold)
print(f"inv vs t2_hold auc_C: n={len(df_compare2)} subjects, t={tstat_2:.3f}, p={pval_2:.4f}")

means = [vals_inv.mean(), vals_t2_hold.mean()]
errors = [vals_inv.std(ddof=1)/len(vals_inv)**0.5, vals_t2_hold.std(ddof=1)/len(vals_t2_hold)**0.5]
plt.bar([0, 1], means, yerr=errors, capsize=8, color=['#4F81BD', '#C0504D'])
plt.xticks([0, 1], ['inv', 't2_hold'])
plt.ylabel('auc_C')
plt.title(f'inv vs t2_hold auc_C (p={pval_2:.4f})')
plt.tight_layout()
plt.show()