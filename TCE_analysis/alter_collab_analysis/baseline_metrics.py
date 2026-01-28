#%%
# ========================================================
# BASELINE METRICS
# Looking at anything at the first trial indicates future effects
# Configuration
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
FIG_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures'
trajectory_path ='/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trajectory_classification.csv'

with open(TRIAL_METRICS_PATH, 'r') as f:
    metrics_data = json.load(f)

with open(TRIAL_DATA_PATH, 'r') as f:
    time_series_trial_data = json.load(f)

trajectory_df = pd.read_csv(trajectory_path)

# Convert to DataFrame
records = []
for subject_id, trials in metrics_data.items():
    for trial_num, trial_data in trials.items():
        record = {
            'subject': int(subject_id),
            'trial_num': int(trial_num),
            **trial_data
        }
        records.append(record)

# Convert to DataFrames
trial_metrics_df = pd.DataFrame(records)
time_series_df = pd.DataFrame(time_series_trial_data)

#%%
# ========================================================
# EXTRACT T1 TEMP AND PLOT DISTRIBUTIONS
# ========================================================
# Extract T1 temperature for each subject from the raw trial data
t1_temps = []
for subject in time_series_df['subject'].unique():
    # Find the first t1_hold trial for this subject
    subj_trials = time_series_df[(time_series_df['subject'] == subject) & 
                                (time_series_df['trial_type'] == 't1_hold')]
    if not subj_trials.empty:
        t1_temp_val = subj_trials['temperature'].max()
        t1_temps.append({'subject': subject, 't1_temp': t1_temp_val})

t1_temp_df = pd.DataFrame(t1_temps)
# Round T1 temperature to the nearest whole number
t1_temp_df['t1_temp'] = t1_temp_df['t1_temp'].round().astype(int)
session_metrics = t1_temp_df[['subject', 't1_temp']].copy()

# Plot histogram of T1 temperatures
plt.figure(figsize=(10, 6))
sns.histplot(session_metrics['t1_temp'], bins=10, kde=False, color='skyblue')
plt.title('Distribution of T1 Temperatures')
plt.xlabel('T1 Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# %% 
# ================================================================================================
# Does the magnitude of the first OH/OA trial correlate with being a sensitizer or a habituator?
# ================================================================================================

# Extract first OH and OA trial metrics for each subject
first_trials = []
for subject in trial_metrics_df['subject'].unique():
    subj_trials = trial_metrics_df[trial_metrics_df['subject'] == subject]
    
    # First OH trial (inv)
    first_oh = subj_trials[subj_trials['trial_type'] == 'inv'].sort_values('trial_num').head(1)
    if not first_oh.empty:
        first_trials.append(first_oh.iloc[0].to_dict())
    
    # First OA trial (offset)
    first_oa = subj_trials[subj_trials['trial_type'] == 'offset'].sort_values('trial_num').head(1)
    if not first_oa.empty:
        first_trials.append(first_oa.iloc[0].to_dict())

first_trials_df = pd.DataFrame(first_trials)

# Merge with trajectory classification
first_trials_with_trajectory = first_trials_df.merge(
    trajectory_df[['subject', 'trajectory_group', 'r_value']], 
    on='subject', 
    how='left'
)

# Remove subjects without trajectory classification
first_trials_with_trajectory = first_trials_with_trajectory.dropna(subset=['trajectory_group'])

print(f"Number of subjects with both first trial data and trajectory classification: {len(first_trials_with_trajectory['subject'].unique())}")
print(f"Trajectory group distribution:\n{first_trials_with_trajectory['trajectory_group'].value_counts()}")

# Separate OH and OA first trials
first_oh_df = first_trials_with_trajectory[first_trials_with_trajectory['trial_type'] == 'inv'].copy()
first_oa_df = first_trials_with_trajectory[first_trials_with_trajectory['trial_type'] == 'offset'].copy()

print(f"\nFirst OH trials: {len(first_oh_df)} subjects")
print(f"First OA trials: {len(first_oa_df)} subjects")

# Analysis 1: First OH trial magnitude vs trajectory slope (r_value)
print("\n" + "="*60)
print("ANALYSIS 1: First OH Trial Magnitude vs Trajectory Slope")
print("="*60)

if not first_oh_df.empty:
    oh_stats = create_correlation_scatter(
        first_oh_df,
        x_col='abs_normalized_pain_change',
        y_col='r_value',
        title='First OH Trial: Pain Change Magnitude vs Trajectory Slope',
        xlabel='First OH Trial Magnitude (% change)',
        ylabel='Trajectory Slope (r_value)',
        figsize=(8, 6)
    )

# Analysis 2: First OA trial magnitude vs trajectory slope (r_value)
print("\n" + "="*60)
print("ANALYSIS 2: First OA Trial Magnitude vs Trajectory Slope")
print("="*60)

if not first_oa_df.empty:
    oa_stats = create_correlation_scatter(
        first_oa_df,
        x_col='abs_normalized_pain_change',
        y_col='r_value',
        title='First OA Trial: Pain Change Magnitude vs Trajectory Slope',
        xlabel='First OA Trial Magnitude (% change)',
        ylabel='Trajectory Slope (r_value)',
        figsize=(8, 6)
    )

# Analysis 3: Compare first trial magnitudes between trajectory groups
print("\n" + "="*60)
print("ANALYSIS 3: First Trial Magnitudes by Trajectory Group")
print("="*60)

# Create box plots comparing trajectory groups
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# First OH trials by group
if not first_oh_df.empty:
    sns.boxplot(data=first_oh_df, x='trajectory_group', y='abs_normalized_pain_change', ax=axes[0])
    sns.stripplot(data=first_oh_df, x='trajectory_group', y='abs_normalized_pain_change', 
                  ax=axes[0], color='black', alpha=0.6, size=4)
    axes[0].set_title('First OH Trial Magnitude\nby Trajectory Group')
    axes[0].set_ylabel('Normalized Pain Change (%)')
    axes[0].set_xlabel('Trajectory Group')

# First OA trials by group
if not first_oa_df.empty:
    sns.boxplot(data=first_oa_df, x='trajectory_group', y='abs_normalized_pain_change', ax=axes[1])
    sns.stripplot(data=first_oa_df, x='trajectory_group', y='abs_normalized_pain_change', 
                  ax=axes[1], color='black', alpha=0.6, size=4)
    axes[1].set_title('First OA Trial Magnitude\nby Trajectory Group')
    axes[1].set_ylabel('Normalized Pain Change (%)')
    axes[1].set_xlabel('Trajectory Group')

plt.tight_layout()
plt.show()

# Statistical tests for group differences
print("\nStatistical Tests for Group Differences:")

if not first_oh_df.empty:
    hab_oh = first_oh_df[first_oh_df['trajectory_group'] == 'habituator']['abs_normalized_pain_change']
    sens_oh = first_oh_df[first_oh_df['trajectory_group'] == 'sensitizer']['abs_normalized_pain_change']
    
    if len(hab_oh) > 0 and len(sens_oh) > 0:
        t_stat, p_val = stats.ttest_ind(hab_oh, sens_oh, equal_var=False)
        print(f"First OH Trial - Habituators vs Sensitizers:")
        print(f"  Habituators: n={len(hab_oh)}, mean={hab_oh.mean():.3f} ± {hab_oh.std():.3f}")
        print(f"  Sensitizers: n={len(sens_oh)}, mean={sens_oh.mean():.3f} ± {sens_oh.std():.3f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")

if not first_oa_df.empty:
    hab_oa = first_oa_df[first_oa_df['trajectory_group'] == 'habituator']['abs_normalized_pain_change']
    sens_oa = first_oa_df[first_oa_df['trajectory_group'] == 'sensitizer']['abs_normalized_pain_change']
    
    if len(hab_oa) > 0 and len(sens_oa) > 0:
        t_stat, p_val = stats.ttest_ind(hab_oa, sens_oa, equal_var=False)
        print(f"First OA Trial - Habituators vs Sensitizers:")
        print(f"  Habituators: n={len(hab_oa)}, mean={hab_oa.mean():.3f} ± {hab_oa.std():.3f}")
        print(f"  Sensitizers: n={len(sens_oa)}, mean={sens_oa.mean():.3f} ± {sens_oa.std():.3f}")
        print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Analysis 4: Separate correlations by trajectory group
print("\n" + "="*60)
print("ANALYSIS 4: Correlations Within Each Trajectory Group")
print("="*60)
# Analysis 4: Separate correlations by trajectory group
print("\n" + "="*60)
print("ANALYSIS 4: Correlations Within Each Trajectory Group")
print("="*60)

# OH trials - Habituators
if not first_oh_df.empty:
    print("\nFirst OH Trials - Habituators:")
    oh_hab_stats = create_correlation_scatter(
        first_oh_df,
        x_col='abs_normalized_pain_change',
        y_col='r_value',
        title='First OH Trial: Habituators Only\nPain Change Magnitude vs Trajectory Slope',
        xlabel='First OH Trial Magnitude (% change)',
        ylabel='Trajectory Slope (r_value)',
        filter_col='trajectory_group',
        filter_val='habituator',
        figsize=(8, 6)
    )

    print("\nFirst OH Trials - Sensitizers:")
    oh_sens_stats = create_correlation_scatter(
        first_oh_df,
        x_col='abs_normalized_pain_change',
        y_col='r_value',
        title='First OH Trial: Sensitizers Only\nPain Change Magnitude vs Trajectory Slope',
        xlabel='First OH Trial Magnitude (% change)',
        ylabel='Trajectory Slope (r_value)',
        filter_col='trajectory_group',
        filter_val='sensitizer',
        figsize=(8, 6)
    )

# OA trials - Habituators and Sensitizers
if not first_oa_df.empty:
    print("\nFirst OA Trials - Habituators:")
    oa_hab_stats = create_correlation_scatter(
        first_oa_df,
        x_col='abs_normalized_pain_change',
        y_col='r_value',
        title='First OA Trial: Habituators Only\nPain Change Magnitude vs Trajectory Slope',
        xlabel='First OA Trial Magnitude (% change)',
        ylabel='Trajectory Slope (r_value)',
        filter_col='trajectory_group',
        filter_val='habituator',
        figsize=(8, 6)
    )

    print("\nFirst OA Trials - Sensitizers:")
    oa_sens_stats = create_correlation_scatter(
        first_oa_df,
        x_col='abs_normalized_pain_change',
        y_col='r_value',
        title='First OA Trial: Sensitizers Only\nPain Change Magnitude vs Trajectory Slope',
        xlabel='First OA Trial Magnitude (% change)',
        ylabel='Trajectory Slope (r_value)',
        filter_col='trajectory_group',
        filter_val='sensitizer',
        figsize=(8, 6)
    )

# Summary of results
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

print("Key Questions:")
print("1. Do first trial magnitudes predict trajectory group membership?")
print("2. Do first trial magnitudes correlate with trajectory slope (r_value)?")
print("3. Are there different patterns within habituators vs sensitizers?")

print(f"\nSample sizes:")
if not first_oh_df.empty:
    print(f"First OH trials: {len(first_oh_df)} total")
    print(f"  - Habituators: {len(first_oh_df[first_oh_df['trajectory_group'] == 'habituator'])}")
    print(f"  - Sensitizers: {len(first_oh_df[first_oh_df['trajectory_group'] == 'sensitizer'])}")

if not first_oa_df.empty:
    print(f"First OA trials: {len(first_oa_df)} total")
    print(f"  - Habituators: {len(first_oa_df[first_oa_df['trajectory_group'] == 'habituator'])}")
    print(f"  - Sensitizers: {len(first_oa_df[first_oa_df['trajectory_group'] == 'sensitizer'])}")

# %%
# Are the habituators driving the lack of first-trial OA/preceding_min correlation? 
# Find first OA trial that has a preceding trial (i.e., not the very first trial of the session)
oa_trials_with_preceding = first_trials_df[
    (first_trials_df['trial_type'] == 'offset') & 
    (first_trials_df['preceding_abs_min_val'].notna())
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
            subset['preceding_abs_min_val'], 
            subset['abs_normalized_pain_change']
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

axes[0].scatter(subset['preceding_abs_min_val'], subset['abs_normalized_pain_change'], 
                alpha=0.6, color='blue', s=60)
axes[0].set_xlabel('Preceding Min Pain')
axes[0].set_ylabel('OA Magnitude (%)')
axes[0].set_title('First OA Trial\nHabituators')

corr, p_val = stats.pearsonr(subset['preceding_abs_min_val'], subset['abs_normalized_pain_change'])
if p_val < alpha_corrected:
    # Add trendline
    x = subset['preceding_abs_min_val']
    y = subset['abs_normalized_pain_change']
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


