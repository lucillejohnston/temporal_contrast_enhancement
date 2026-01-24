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

############################################################################################### Overall distributions of auc_total and abs_max_val
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

########################################################################################## Compare abs_max_val and time_yoked_max_val for control trials
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

#%%
# ============================================================================
# Sanity Check: Replicate figures from the paper
# ============================================================================

from statsmodels.stats.weightstats import DescrStatsW

def mean_ci(data):
    dsw = DescrStatsW(data)
    mean = dsw.mean
    ci_low, ci_upp = dsw.tconfint_mean(alpha=0.05)
    return mean, ci_low, ci_upp

# (trial1, trial2, metric, label, ylims)
comparisons = [
    ('t2_hold', 'inv', 'max_val', 'Max Value: t2_hold vs. inv', (40, 80)),
    ('t1_hold', 'offset', 'min_val', 'Min Value: t1_hold vs. offset', (0, 40)),
    ('t1_hold', 'stepdown', 'min_val', 'Min Value: t1_hold vs. stepdown', (0, 40)),
]
####################################################################################### Compare t1_hold vs offset/stepdown and t2_hold vs inv for max and min values
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


# %%
"""Section 4: Plotting metrics across trials per subject within a session
"""
# Combine the two different t1_hold metrics into one because I can't be bothered to deal with them separately and they are super close together
def combine_t1_hold_metrics(trial_metrics_df):
    t1_hold_mask = trial_metrics_df['trial_type'] == 't1_hold'
    t1_hold_df = trial_metrics_df[t1_hold_mask].copy()
    for idx, row in t1_hold_df.iterrows():
        subject = row['subject']
        trial_num = row['trial_num']
        # Find all offset and stepdown trials for this subject
        offset_trials = trial_metrics_df[(trial_metrics_df['subject'] == subject) & (trial_metrics_df['trial_type'] == 'offset')]
        stepdown_trials = trial_metrics_df[(trial_metrics_df['subject'] == subject) & (trial_metrics_df['trial_type'] == 'stepdown')]
        # Find the closest offset and stepdown trial by trial_num
        offset_trial_num = offset_trials['trial_num'].iloc[(np.abs(offset_trials['trial_num'] - trial_num)).argmin()] if not offset_trials.empty else np.nan
        stepdown_trial_num = stepdown_trials['trial_num'].iloc[(np.abs(stepdown_trials['trial_num'] - trial_num)).argmin()] if not stepdown_trials.empty else np.nan
        # Choose the reference with the smallest distance
        dist_offset = abs(trial_num - offset_trial_num) if not np.isnan(offset_trial_num) else np.inf
        dist_stepdown = abs(trial_num - stepdown_trial_num) if not np.isnan(stepdown_trial_num) else np.inf
        if dist_offset <= dist_stepdown:
            ref = 'offset'
        else:
            ref = 'stepdown'
        # Combine metrics
        min_val = row.get(f'min_val_{ref}', np.nan)
        max_val = row.get(f'max_val_{ref}', np.nan)
        peak_to_peak = row.get(f'peak_to_peak_{ref}', np.nan)
        # Save in new columns
        trial_metrics_df.loc[idx, 'min_val'] = min_val
        trial_metrics_df.loc[idx, 'max_val'] = max_val
        trial_metrics_df.loc[idx, 'peak_to_peak'] = peak_to_peak
        trial_metrics_df.loc[idx, 't1_hold_reference_used'] = ref
    return trial_metrics_df

trial_metrics_df = combine_t1_hold_metrics(trial_metrics_df)

# Plot Across Trials per Subject within a Session
trial_types = trial_metrics_df['trial_type'].unique()
metrics = ['auc_A', 'auc_B', 'auc_C', 'peak_to_peak']
for metric in metrics:
    for trial_type in trial_types:
        plt.figure(figsize=(10, 5))
        df_type = trial_metrics_df[trial_metrics_df['trial_type'] == trial_type]
        for subject, subj_df in df_type.groupby('subject'):
            subj_df = subj_df.sort_values('trial_num')
            if subj_df.empty:
                continue
            first_val = subj_df[metric].iloc[0]
            last_val = subj_df[metric].iloc[-1]
            color = 'red' if last_val >= first_val else 'blue'
            plt.plot(subj_df['trial_num'], subj_df[metric], marker='o', linestyle='-', color=color, alpha=0.7)
        plt.title(f"{metric} across trials for {trial_type}")
        plt.xlabel('Trial Order')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()
#%%
# Combining specific metrics over time
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
        first_val = subj_df['auc_A'].iloc[0]
        last_val = subj_df['auc_A'].iloc[-1]
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
        first_val = subj_df['auc_A'].iloc[0]
        last_val = subj_df['auc_A'].iloc[-1]
        color = 'red' if last_val >= first_val else 'blue'
        plt.plot(subj_df['trial_num'], subj_df['auc_A'], marker='o', linestyle='-', color=color, alpha=0.7)
plt.title("auc_A over trial order for t2_hold, inv, and stepdown")
plt.xlabel("Trial Order")
plt.ylabel("auc_A")
plt.tight_layout()
plt.show()
# %%
# Compare subject-level color assignments between groups
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

#%% 
# Compare 'baseline metrics' 
baseline_metrics = ['threshold','t1_temp','initial_OA']
# Extract T1 temperature for each subject from the raw trial data
t1_temps = []
for subject in df['subject'].unique():
    # Find the first t1_hold or offset trial for this subject
    subj_trials = df[(df['subject'] == subject) & (df['trial_type'].isin(['t1_hold', 'offset']))]
    if subj_trials.empty:
        continue
    # Get the first trial_num for this subject/trial_type
    trial_num = subj_trials['trial_num'].min()
    trial_df = subj_trials[subj_trials['trial_num'] == trial_num]
    # Find the aligned_time closest to zero
    idx_closest = (trial_df['aligned_time'] - 0).abs().idxmin()
    t1_temp_val = trial_df.loc[idx_closest, 'temperature']
    t1_temps.append({'subject': subject, 't1_temp_C': t1_temp_val})

t1_temp_df = pd.DataFrame(t1_temps)
print(t1_temp_df)
# Plot a histogram of T1 temperatures
plt.figure(figsize=(8, 5))
sns.histplot(t1_temp_df['t1_temp_C'], bins=10, kde=False, color='skyblue')
plt.title('Distribution of T1 Temperatures')
plt.xlabel('T1 Temperature (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
# Round T1 temperature to the nearest whole number and save to session_metrics
t1_temp_df['t1_temp'] = t1_temp_df['t1_temp_C'].round().astype(int)
session_metrics = t1_temp_df[['subject', 't1_temp']].copy()
print(session_metrics)

#%% Compare auc_C between OA and t1_hold trials
# Compare auc_C between offset and t1_hold trials per subject
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
# %% NORMALIZED PAIN CHANGE 
# For offset trials: (min_val - max_val) / (max_val) * 100 
# For t1_hold trials: (min_val - max_val) / (max_val) * 100 
# For onset trials: (max_val - min_val) / (100-min_val) * 100
# For t2_hold trials: (max_val - min_val) / (100-min_val) * 100

# Plot normalized pain change across trials for each trial_type
for tt in trial_metrics_df['trial_type'].unique():
    plt.figure(figsize=(10, 6))
    df_tt = trial_metrics_df[trial_metrics_df['trial_type'] == tt]
    for subject, subj_df in df_tt.groupby('subject'):
        subj_df = subj_df.sort_values("trial_num")
        if subj_df.empty:
            continue
        plt.plot(
            subj_df["trial_num"],
            subj_df["normalized_pain_change"],
            marker="o",
            linestyle="-",
            alpha=0.7,
            label=f"Subject {subject}"
        )
    plt.xlabel("Trial Number")
    plt.ylabel("Normalized Pain Change (%)")
    plt.title(f"Normalized Pain Change Across Trials: {tt}")
    plt.tight_layout()
    plt.show()

# Plot according to t1_temp
# Create a color mapping based on t1_temp
color_map = dict(
    zip(
        session_metrics['subject'],
        ['blue' if t <= 45 else 'red' for t in session_metrics['t1_temp']]
    )
)

# Plot normalized pain change across trials for each trial_type, color-coded by t1_temp
for tt in trial_metrics_df['trial_type'].unique():
    plt.figure(figsize=(10, 6))
    df_tt = trial_metrics_df[trial_metrics_df['trial_type'] == tt]
    for subject, subj_df in df_tt.groupby('subject'):
        subj_df = subj_df.sort_values("trial_num")
        if subj_df.empty:
            continue
        color = color_map.get(subject, 'gray')  # default to gray if subject not in session_metrics
        plt.plot(
            subj_df["trial_num"],
            subj_df["normalized_pain_change"],
            marker="o",
            linestyle="-",
            color=color,
            alpha=0.7,
            label=f"Subject {subject}"
        )
    plt.xlabel("Trial Number")
    plt.ylabel("Normalized Pain Change (%)")
    plt.title(f"Normalized Pain Change Across Trials: {tt}")
    plt.tight_layout()
    plt.show()

#%%
# Exclude subjects with t1_temp == 45 and plot again with the same color scheme (maybe forcing a gap between the datasets will force a separation?)
excluded_subjects = session_metrics[session_metrics['t1_temp'] == 45]['subject'].tolist()
# Plot normalized pain change across trials for each trial_type, excluding subjects with t1_temp == 45
for tt in trial_metrics_df['trial_type'].unique():
    plt.figure(figsize=(10, 6))
    df_tt = trial_metrics_df[trial_metrics_df['trial_type'] == tt]
    for subject, subj_df in df_tt.groupby('subject'):
        if subject in excluded_subjects:
            continue  # Skip subjects with t1_temp == 45
        subj_df = subj_df.sort_values("trial_num")
        if subj_df.empty:
            continue
        color = color_map.get(subject, 'gray')  # default to gray if subject not in session_metrics
        plt.plot(
            subj_df["trial_num"],
            subj_df["normalized_pain_change"],
            marker="o",
            linestyle="-",
            color=color,
            alpha=0.7,
            label=f"Subject {subject}"
        )
    plt.xlabel("Trial Number")
    plt.ylabel("Normalized Pain Change (%)")
    plt.title(f"Normalized Pain Change Across Trials: {tt} (Excluding t1_temp == 45)")
    plt.tight_layout()
    plt.show()


# %% Determine habituators vs. sensitizers based on pain trajectory
def get_max_pain_value(row):
    """Get the appropriate max pain value based on trial type"""
    trial_type = row['trial_type']
    
    if trial_type == 't1_hold':
        # For t1_hold, try offset first, then stepdown
        if pd.notna(row.get('max_val_offset')):
            return row['max_val_offset']
        elif pd.notna(row.get('max_val_stepdown')):
            return row['max_val_stepdown']
        else:
            return np.nan
    
    elif trial_type == 't2_hold':
        # For t2_hold, use regular max_val if available
        if pd.notna(row.get('max_val')):
            return row['max_val']
        elif pd.notna(row.get('max_val_inv')):
            return row['max_val_inv']
        else:
            return np.nan
    
    elif trial_type in ['offset', 'inv', 'stepdown']:
        # These should have regular max_val
        return row.get('max_val', np.nan)
    
    else:
        return np.nan

# Add a unified max pain column
trial_metrics_df['max_pain_unified'] = trial_metrics_df.apply(get_max_pain_value, axis=1)

def calculate_pain_trajectory(subject_data):
    # Use the unified max pain column
    clean_data = subject_data.dropna(subset=['max_pain_unified', 'trial_num'])
    
    if len(clean_data) < 3:
        return np.nan, np.nan, np.nan
    
    if clean_data['max_pain_unified'].var() == 0:
        return 0.0, np.nan, np.nan
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            clean_data['trial_num'],
            clean_data['max_pain_unified']
        )
        return slope, r_value, p_value
    except:
        return np.nan, np.nan, np.nan

# Recalculate trajectories
subject_trajectories = []
for subject, subj_df in trial_metrics_df.groupby('subject'):
    slope, r_val, p_val = calculate_pain_trajectory(subj_df)
    subject_trajectories.append({
        'subject': subject,
        'slope': slope,
        'r_value': r_val,
        'p_value': p_val
    })
trajectory_df = pd.DataFrame(subject_trajectories)

# Calculate slopes for all subjects
subject_slopes = []
for subject in trial_metrics_df['subject'].unique():
    subj_data = trial_metrics_df[trial_metrics_df['subject'] == subject]
    if len(subj_data) > 3:  # Need minimum trials
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            subj_data['trial_num'], 
            subj_data['max_pain_unified']
        )
        subject_slopes.append({
            'subject': subject,
            'slope': slope,
            'r_value': r_value,
            'p_value': p_value
        })
slopes_df = pd.DataFrame(subject_slopes)

# Use the corrected histogram
plt.figure(figsize=(8, 6))
plt.hist(slopes_df['slope'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='No change')
plt.xlabel('Max Pain Slope (points per trial)')
plt.ylabel('Number of Subjects')
plt.title('Distribution of Individual Pain Trajectories')
plt.legend()
plt.show()

print(f"Mean slope: {slopes_df['slope'].mean():.2f} points per trial")
print(f"Subjects with positive slopes (sensitizers): {(slopes_df['slope'] > 0).sum()}")
print(f"Subjects with negative slopes (habituators): {(slopes_df['slope'] < 0).sum()}")


# Classify subjects
def classify_subject(row):
    if row['slope'] < 0 and row['p_value'] < 0.05:
        return 'habituator'
    elif row['slope'] > 0 and row['p_value'] < 0.05:
        return 'sensitizer'
    else:
        return 'no_trend'

trajectory_df['classification'] = trajectory_df.apply(classify_subject, axis=1)
# %%
# Pick one subject with a clear trend using max_pain_unified
# Use the trajectory_df (max_pain_unified based) for your example
example_subject = trajectory_df[trajectory_df['slope'] > 3]['subject'].iloc[0]
subj_data = trial_metrics_df[trial_metrics_df['subject'] == example_subject].sort_values('trial_num')

plt.figure(figsize=(8, 6))
plt.scatter(subj_data['trial_num'], subj_data['max_pain_unified'], alpha=0.7, s=60)

# Get the slope from trajectory_df
subject_slope_info = trajectory_df[trajectory_df['subject'] == example_subject].iloc[0]
slope = subject_slope_info['slope']
r_value = subject_slope_info['r_value']
p_value = subject_slope_info['p_value']

# Add regression line
x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
y_vals = subj_data['max_val'].iloc[0] + slope * (x_vals - subj_data['trial_num'].iloc[0])  # Rough approximation
plt.plot(x_vals, y_vals, 'r-', linewidth=2)

plt.xlabel('Trial Number')
plt.ylabel('Maximum Pain Rating')
plt.ylim(0, 100)
plt.title(f'Example: Subject {example_subject}\nSlope = {slope:.2f} points/trial')
# %% Split sensitizers vs habituators and look at across-trial metrics
# Classify subjects
threshold = 1 # for now, set a threshold for slope classification (eventually in some data-driven way)
def classify_trajectory(slope):
    if slope < -threshold:
        return 'habituator'
    elif slope > threshold:
        return 'sensitizer'
    else:
        return 'no_trend'

slopes_df['trajectory_group'] = slopes_df['slope'].apply(classify_trajectory)
print(slopes_df['trajectory_group'].value_counts())

# Merge with trial_metrics_df
trial_metrics_df = trial_metrics_df.merge(
    slopes_df[['subject', 'trajectory_group']], 
    on='subject', 
    how='left'
)

# Check if merge worked
print(f"Trajectory group in trial_metrics_df:")
print(trial_metrics_df['trajectory_group'].value_counts(dropna=False))

# Filter for just habituators and sensitizers, and OA/OH trials
context_analysis_df = trial_metrics_df[
    (trial_metrics_df['trajectory_group'].isin(['habituator', 'sensitizer'])) &
    (trial_metrics_df['trial_type'].isin(['offset', 'inv']))
].copy()

# Add both preceding max and min pain
context_analysis_df['preceding_max_val'] = context_analysis_df.groupby('subject')['max_pain_unified'].shift(1)
context_analysis_df['preceding_min_val'] = context_analysis_df.groupby('subject')['min_val'].shift(1)  # Add this

# Test the correct relationships:
# OH (inv): preceding_max_val → normalized_pain_change
# OA (offset): preceding_min_val → normalized_pain_change

print("=== ONSET HYPERALGESIA (INV) - Preceding MAX Pain ===")
for group in ['habituator', 'sensitizer']:
    subset = context_analysis_df[
        (context_analysis_df['trial_type'] == 'inv') & 
        (context_analysis_df['trajectory_group'] == group)
    ]
    
    clean_subset = subset.dropna(subset=['preceding_max_val', 'normalized_pain_change'])
    
    if len(clean_subset) >= 5:
        corr, p_val = stats.pearsonr(
            clean_subset['preceding_max_val'], 
            clean_subset['normalized_pain_change']
        )
        print(f"{group.capitalize()}s: r={corr:.3f}, p={p_val:.3f} (n={len(clean_subset)})")
    else:
        print(f"{group.capitalize()}s: insufficient data (n={len(clean_subset)})")

print("\n=== OFFSET ANALGESIA (OFFSET) - Preceding MIN Pain ===")
for group in ['habituator', 'sensitizer']:
    subset = context_analysis_df[
        (context_analysis_df['trial_type'] == 'offset') & 
        (context_analysis_df['trajectory_group'] == group)
    ]
    
    clean_subset = subset.dropna(subset=['preceding_min_val', 'normalized_pain_change'])
    
    if len(clean_subset) >= 5:
        corr, p_val = stats.pearsonr(
            clean_subset['preceding_min_val'], 
            clean_subset['normalized_pain_change']
        )
        print(f"{group.capitalize()}s: r={corr:.3f}, p={p_val:.3f} (n={len(clean_subset)})")
    else:
        print(f"{group.capitalize()}s: insufficient data (n={len(clean_subset)})")
# %%
# Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# OH - Habituators (top left)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == 'inv') & 
    (context_analysis_df['trajectory_group'] == 'habituator')
]
clean_subset = subset.dropna(subset=['preceding_max_val', 'normalized_pain_change'])

axes[0,0].scatter(clean_subset['preceding_max_val'], clean_subset['normalized_pain_change'], 
                  alpha=0.6, color='blue', s=50)
axes[0,0].set_xlabel('Preceding Max Pain')
axes[0,0].set_ylabel('OH Magnitude (%)')
axes[0,0].set_title('Onset Hyperalgesia - Habituators')

corr, p_val = stats.pearsonr(clean_subset['preceding_max_val'], clean_subset['normalized_pain_change'])
if p_val < 0.00125: # Bonferroni correction for 4 comparisons
    # Add trendline
    x = clean_subset['preceding_max_val']
    y = clean_subset['normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0,0].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
    # Add stats box
    axes[0,0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                   transform=axes[0,0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# OH - Sensitizers (top right)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == 'inv') & 
    (context_analysis_df['trajectory_group'] == 'sensitizer')
]
clean_subset = subset.dropna(subset=['preceding_max_val', 'normalized_pain_change'])

axes[0,1].scatter(clean_subset['preceding_max_val'], clean_subset['normalized_pain_change'], 
                  alpha=0.6, color='red', s=50)
axes[0,1].set_xlabel('Preceding Max Pain')
axes[0,1].set_ylabel('OH Magnitude (%)')
axes[0,1].set_title('Onset Hyperalgesia - Sensitizers')

corr, p_val = stats.pearsonr(clean_subset['preceding_max_val'], clean_subset['normalized_pain_change'])
if p_val < 0.05:
    # Add trendline
    x = clean_subset['preceding_max_val']
    y = clean_subset['normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0,1].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
    # Add stats box
    axes[0,1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                   transform=axes[0,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# OA - Habituators (bottom left)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == 'offset') & 
    (context_analysis_df['trajectory_group'] == 'habituator')
]
clean_subset = subset.dropna(subset=['preceding_min_val', 'normalized_pain_change'])

axes[1,0].scatter(clean_subset['preceding_min_val'], clean_subset['normalized_pain_change'], 
                  alpha=0.6, color='blue', s=50)
axes[1,0].set_xlabel('Preceding Min Pain')
axes[1,0].set_ylabel('OA Magnitude (%)')
axes[1,0].set_title('Offset Analgesia - Habituators')

corr, p_val = stats.pearsonr(clean_subset['preceding_min_val'], clean_subset['normalized_pain_change'])
if p_val < 0.05:
    # Add trendline
    x = clean_subset['preceding_min_val']
    y = clean_subset['normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1,0].plot(x, p(x), "r-", alpha=0.8, linewidth=2)

# Add stats box (even if not significant, for comparison)
axes[1,0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
               transform=axes[1,0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round", facecolor="lightgray" if p_val >= 0.05 else "wheat", alpha=0.8))

# OA - Sensitizers (bottom right)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == 'offset') & 
    (context_analysis_df['trajectory_group'] == 'sensitizer')
]
clean_subset = subset.dropna(subset=['preceding_min_val', 'normalized_pain_change'])

axes[1,1].scatter(clean_subset['preceding_min_val'], clean_subset['normalized_pain_change'], 
                  alpha=0.6, color='red', s=50)
axes[1,1].set_xlabel('Preceding Min Pain')
axes[1,1].set_ylabel('OA Magnitude (%)')
axes[1,1].set_title('Offset Analgesia - Sensitizers')

corr, p_val = stats.pearsonr(clean_subset['preceding_min_val'], clean_subset['normalized_pain_change'])
if p_val < 0.05:
    # Add trendline
    x = clean_subset['preceding_min_val']
    y = clean_subset['normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1,1].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
    # Add stats box
    axes[1,1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                   transform=axes[1,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.savefig('context_effects_by_trajectory_group.svg', dpi=300, bbox_inches='tight')
plt.show()
# %% Are the habituators driving the lack of first-trial OA/preceding_min correlation? 
# Find first OA trial that has a preceding trial (i.e., not the very first trial of the session)
oa_trials_with_preceding = context_analysis_df[
    (context_analysis_df['trial_type'] == 'offset') & 
    (context_analysis_df['preceding_min_val'].notna())
]

# Get the first OA trial with preceding data for each subject
first_oa_with_context = (
    oa_trials_with_preceding
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

print("First OA trial WITH preceding context by group:")
print(f"Total first trials with context: {len(first_oa_with_context)}")
print(f"Breakdown: {first_oa_with_context['trajectory_group'].value_counts()}")

# Test correlation for each group on first contextual trials
for group in ['habituator', 'sensitizer']:
    subset = first_oa_with_context[first_oa_with_context['trajectory_group'] == group]
    
    if len(subset) >= 5:
        corr, p_val = stats.pearsonr(
            subset['preceding_min_val'], 
            subset['normalized_pain_change']
        )
        print(f"{group.capitalize()}s (first contextual trial): r={corr:.3f}, p={p_val:.3f} (n={len(subset)})")
    else:
        print(f"{group.capitalize()}s (first contextual trial): insufficient data (n={len(subset)})")

# Set Bonferroni-corrected alpha (if you want to use it)
alpha_corrected = 0.05 / 2  # 2 comparisons now
print(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")

# Create a 1x2 subplot figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First OA trial - Habituators (left)
subset = first_oa_with_context[first_oa_with_context['trajectory_group'] == 'habituator']

axes[0].scatter(subset['preceding_min_val'], subset['normalized_pain_change'], 
                alpha=0.6, color='blue', s=60)
axes[0].set_xlabel('Preceding Min Pain')
axes[0].set_ylabel('OA Magnitude (%)')
axes[0].set_title('First OA Trial\nHabituators')

corr, p_val = stats.pearsonr(subset['preceding_min_val'], subset['normalized_pain_change'])
if p_val < alpha_corrected:
    # Add trendline
    x = subset['preceding_min_val']
    y = subset['normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
    # Add stats box (significant)
    axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}*', 
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
else:
    # Add stats box (non-significant)
    axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

# First OA trial - Sensitizers (right)
subset = first_oa_with_context[first_oa_with_context['trajectory_group'] == 'sensitizer']

axes[1].scatter(subset['preceding_min_val'], subset['normalized_pain_change'], 
                alpha=0.6, color='red', s=60)
axes[1].set_xlabel('Preceding Min Pain')
axes[1].set_ylabel('OA Magnitude (%)')
axes[1].set_title('First OA Trial\nSensitizers')

corr, p_val = stats.pearsonr(subset['preceding_min_val'], subset['normalized_pain_change'])
if p_val < alpha_corrected:
    # Add trendline
    x = subset['preceding_min_val']
    y = subset['normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
    # Add stats box (significant)
    axes[1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
else:
    # Add stats box (non-significant)
    axes[1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('first_oa_by_trajectory_group.svg', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSample sizes:")
print(f"Habituators: n={len(first_oa_with_context[first_oa_with_context['trajectory_group'] == 'habituator'])}")
print(f"Sensitizers: n={len(first_oa_with_context[first_oa_with_context['trajectory_group'] == 'sensitizer'])}")

# %% Normalized pain change by trajectory group 

# Create 1x2 subplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# OH Magnitude Histogram (INV trials)
oh_data = context_analysis_df[context_analysis_df['trial_type'] == 'inv']

axes[0].hist(oh_data[oh_data['trajectory_group'] == 'habituator']['normalized_pain_change'], 
             alpha=0.7, color='blue', label='Habituators', bins=20, density=True, edgecolor='white', linewidth=0.5)
axes[0].hist(oh_data[oh_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'], 
             alpha=0.7, color='red', label='Sensitizers', bins=20, density=True, edgecolor='white', linewidth=0.5)

axes[0].set_xlabel('OH Magnitude (%)')
axes[0].set_ylabel('Density (proportion of trials)')
axes[0].set_title('Onset Hyperalgesia Distribution')
axes[0].legend()
axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)

# OA Magnitude Histogram (OFFSET trials)
oa_data = context_analysis_df[context_analysis_df['trial_type'] == 'offset']

axes[1].hist(oa_data[oa_data['trajectory_group'] == 'habituator']['normalized_pain_change'], 
             alpha=0.7, color='blue', label='Habituators', bins=20, density=True, edgecolor='white', linewidth=0.5)
axes[1].hist(oa_data[oa_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'], 
             alpha=0.7, color='red', label='Sensitizers', bins=20, density=True, edgecolor='white', linewidth=0.5)

axes[1].set_xlabel('OA Magnitude (%)')
axes[1].set_ylabel('Density (proportion of trials)')
axes[1].set_title('Offset Analgesia Distribution')
axes[1].legend()
axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Also bar chart of means with error bars
mean_oh_hab = oh_data[oh_data['trajectory_group'] == 'habituator']['normalized_pain_change'].mean()
mean_oh_sens = oh_data[oh_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'].mean()
se_oh_hab = oh_data[oh_data['trajectory_group'] == 'habituator']['normalized_pain_change'].std(ddof=1) / np.sqrt(len(oh_data[oh_data['trajectory_group'] == 'habituator']))
se_oh_sens = oh_data[oh_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'].std(ddof=1) / np.sqrt(len(oh_data[oh_data['trajectory_group'] == 'sensitizer']))
plt.figure(figsize=(6, 6))
plt.bar([0, 1], [mean_oh_hab, mean_oh_sens], yerr=[se_oh_hab, se_oh_sens], capsize=8, color=['blue', 'red'])
plt.xticks([0, 1], ['Habituators', 'Sensitizers'])
plt.ylabel('Mean OH Magnitude (%)')
plt.title('Mean Onset Hyperalgesia by Trajectory Group')
plt.tight_layout()
plt.show()
t, p = stats.ttest_ind(
    oh_data[oh_data['trajectory_group'] == 'habituator']['normalized_pain_change'],
    oh_data[oh_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'],
    equal_var=False
)
print(f"OH Magnitude: t={t:.3f}, p={p:.4f}")



mean_oa_hab = oa_data[oa_data['trajectory_group'] == 'habituator']['normalized_pain_change'].mean()
mean_oa_sens = oa_data[oa_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'].mean()
se_oa_hab = oa_data[oa_data['trajectory_group'] == 'habituator']['normalized_pain_change'].std(ddof=1) / np.sqrt(len(oa_data[oa_data['trajectory_group'] == 'habituator']))
se_oa_sens = oa_data[oa_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'].std(ddof=1) / np.sqrt(len(oa_data[oa_data['trajectory_group'] == 'sensitizer']))
t, p = stats.ttest_ind(
    oa_data[oa_data['trajectory_group'] == 'habituator']['normalized_pain_change'],
    oa_data[oa_data['trajectory_group'] == 'sensitizer']['normalized_pain_change'],
    equal_var=False
)
print(f"OA Magnitude: t={t:.3f}, p={p:.4f}")
plt.figure(figsize=(6, 6))
plt.bar([0, 1], [mean_oa_hab, mean_oa_sens], yerr=[se_oa_hab, se_oa_sens], capsize=8, color=['blue', 'red'])
plt.xticks([0, 1], ['Habituators', 'Sensitizers'])
plt.ylabel('Mean OA Magnitude (%)')
plt.title('Mean Offset Analgesia by Trajectory Group')
plt.tight_layout()
plt.show()



# %%
