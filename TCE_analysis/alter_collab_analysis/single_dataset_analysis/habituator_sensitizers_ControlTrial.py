#%%
# ========================================================
# CONFIGURATION
# ========================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json, sys
from scipy import stats
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis') # add main TCE_analysis folder to path
from alter_collab_analysis.utils.plotting_functions import *  

# dataset -  single dataset currently
dataset = 'plosONE' # options: 'kneeOA', 'plosONE', 'cLBP', 'sEEG'

# File paths
TRIAL_METRICS_PATH = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_metrics.json'
TRIAL_DATA_PATH = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_data_trimmed_downsampled.json'
FIG_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/'
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

if dataset == 'kneeOA':
    stepped_trial_types = ['offset','onset']
elif dataset == 'plosONE':
    stepped_trial_types = ['offset', 'inv']
# %% 
# =====================================================================
# Determine habituators vs. sensitizers 
# =====================================================================
# Primary metric: slope of period C of hold trials (maybe as a whole or in sliding windows)
def calculate_slope(trial_data, C_start=None, C_end=None):
    period_c_data = trial_data[(trial_data['aligned_time'] >= C_start) & (trial_data['aligned_time'] <= C_end)]
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

hold_trials = trial_metrics_df[trial_metrics_df['trial_type'].str.contains('hold')].copy()
results = []
for _, row in hold_trials.iterrows():
    subject_id = row['subject']
    trial_num = row['trial_num']
    c_start = row['C_start']
    c_end = row['C_end']
    a_start = row['A_start']

    trial_ts = time_series_df[
        (time_series_df['subject'] == subject_id) &
        (time_series_df['trial_num'] == trial_num)
    ].copy()

    if len(trial_ts) < 2:
        continue

    slope = calculate_slope(trial_ts, C_start=c_start, C_end=c_end)
    diff_5s = calculate_windowed_difference(trial_ts, window_size=5, A_start=a_start, C_end=c_end)
    diff_10s = calculate_windowed_difference(trial_ts, window_size=10, A_start=a_start, C_end=c_end)
    auc_diff_10s = calculate_auc_difference(trial_ts, window_size=10, A_start=a_start, C_end=c_end)
    norm_change = calculate_normalized_time_aware_change(row, max_floor=5)

    results.append({
        'subject': subject_id,
        'trial_num': trial_num,
        'trial_type': row['trial_type'],
        'slope': slope,
        'late5_minus_early5': diff_5s,
        'late10_minus_early10': diff_10s,
        'auc_diff_10s': auc_diff_10s,
        'time_aware_norm_change': norm_change
    })

hold_metrics_df = pd.DataFrame(results)
# Plot histograms of each metric across all hold trials
for col in hold_metrics_df.columns[3:]:
    vals = hold_metrics_df[col].dropna()
    plt.figure(figsize=(10,6))
    plt.hist(vals, bins='auto', edgecolor='black')
    plt.title(f'Distribution of {col} across hold trials')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# %%
# ==========================================================================
# Split into HABITUATORS vs. SENSITIZERS and PLOT EXAMPLE for each 
# ==========================================================================

# Average across hold trials per subject to get a single classification metric per subject
hold_metrics_avg_per_subject = hold_metrics_df.groupby('subject')[[
    'slope',
    'late5_minus_early5',
    'late10_minus_early10',
    'auc_diff_10s',
    'time_aware_norm_change'
]].mean().reset_index()

# Plot histograms of each subject rather than each trial
for col in hold_metrics_avg_per_subject.columns[1:]:
    vals = hold_metrics_avg_per_subject[col].dropna()

    plt.figure(figsize=(10,6))
    plt.hist(vals, bins='auto', edgecolor='black')
    plt.title(f'Subject-level distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
hold_metrics_avg_per_subject.describe()

###################################################################################### Classify subjects
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
        classification = 'non-responder'
    print(".")
    return classification, observed_mean, lower_bound, upper_bound

classification_results = []
for subject_id, subject_trials in hold_metrics_df.groupby('subject'):
    classification, observed_mean, lower_bound, upper_bound = (
        bootstrap_classification(subject_trials['slope'], 
                                 n_boot=10000, ci=95, random_state=42
                                )
    )
    classification_results.append({
        'subject': subject_id,
        'n_trials': len(subject_trials),
        'observed_mean_slope': observed_mean,
        'classification': classification,
        'ci_lower': lower_bound,
        'ci_upper': upper_bound,
    })

slope_classification_df = pd.DataFrame(classification_results)

#%% Plot results of the classification
# Plot classification results
plot_df = hold_metrics_avg_per_subject.merge(
    slope_classification_df[['subject','classification']],
    on='subject',
    how='left'
)
plt.figure(figsize=(10,6))
for cls in ['habituator','non-responder','sensitizer']:
    subset = plot_df[plot_df['classification'] == cls]

    plt.hist(
        subset['slope'],
        bins=20,
        alpha=0.6,
        label=cls
    )
plt.axvline(0,color='k',linestyle='--')
plt.title('Bootstrap Classification')
plt.xlabel('Mean slope')
plt.ylabel('Count')
plt.legend()
####################################################################################### Plot example subject for each group
def plot_hold_trials_for_subject(subject_id, hold_metrics_df, time_series_df, title=None, time_window=None):
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

    fig, ax1 = plt.subplots(figsize=(10, 6))
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

    plt.tight_layout()
    plt.show()


hab_subject = slope_classification_df[
    slope_classification_df['classification'] == 'habituator'
]['subject'].sample(1).iloc[0]

sens_subject = slope_classification_df[
    slope_classification_df['classification'] == 'sensitizer'
]['subject'].sample(1).iloc[0]

nr_subject = slope_classification_df[
    slope_classification_df['classification'] == 'non-responder'
]['subject'].sample(1).iloc[0]

# Plot examples
plot_hold_trials_for_subject(hab_subject, hold_metrics_df, time_series_df, title=f"Example Habituator (Subject {hab_subject})")
plot_hold_trials_for_subject(sens_subject, hold_metrics_df, time_series_df, title=f"Example Sensitizer (Subject {sens_subject})")
plot_hold_trials_for_subject(nr_subject, hold_metrics_df, time_series_df, title=f"Example Non-responder (Subject {nr_subject})")

# %%
# ===================================================================
# Preceding trial context analysis for HABITUATORS vs. SENSITIZERS
# ===================================================================

# Filter for just habituators and sensitizers, and OA/OH trials
context_analysis_df = trial_metrics_df[
    (trial_metrics_df['classification'].isin(['habituator', 'sensitizer'])) &
    (trial_metrics_df['trial_type'].isin(stepped_trial_types))
].copy()

# Add both preceding max and min pain
context_analysis_df['preceding_abs_max_val'] = context_analysis_df.groupby('subject')['abs_max_val'].shift(1)
context_analysis_df['preceding_abs_min_val'] = context_analysis_df.groupby('subject')['abs_min_val'].shift(1)

# Test the correct relationships:
# OH (inv): preceding_abs_max_val → abs_normalized_pain_change
# OA (offset): preceding_abs_min_val → abs_normalized_pain_change

print("=== ONSET HYPERALGESIA (INV) - Preceding MAX Pain ===")
for group in ['habituator', 'sensitizer']:
    subset = context_analysis_df[
        (context_analysis_df['trial_type'] == stepped_trial_types[1]) & # 'inv' for plosONE, 'onset' for kneeOA
        (context_analysis_df['classification'] == group)
    ]
    
    clean_subset = subset.dropna(subset=['preceding_abs_max_val', 'abs_normalized_pain_change'])
    
    if len(clean_subset) >= 5:
        corr, p_val = stats.pearsonr(
            clean_subset['preceding_abs_max_val'], 
            clean_subset['abs_normalized_pain_change']
        )
        print(f"{group.capitalize()}s: r={corr:.3f}, p={p_val:.3f} (n={len(clean_subset)})")
    else:
        print(f"{group.capitalize()}s: insufficient data (n={len(clean_subset)})")

print("\n=== OFFSET ANALGESIA (OFFSET) - Preceding MIN Pain ===")
for group in ['habituator', 'sensitizer']:
    subset = context_analysis_df[
        (context_analysis_df['trial_type'] == stepped_trial_types[0]) & # 'offset' for both datasets
        (context_analysis_df['classification'] == group)
    ]
    
    clean_subset = subset.dropna(subset=['preceding_abs_min_val', 'abs_normalized_pain_change'])
    
    if len(clean_subset) >= 5:
        corr, p_val = stats.pearsonr(
            clean_subset['preceding_abs_min_val'], 
            clean_subset['abs_normalized_pain_change']
        )
        print(f"{group.capitalize()}s: r={corr:.3f}, p={p_val:.3f} (n={len(clean_subset)})")
    else:
        print(f"{group.capitalize()}s: insufficient data (n={len(clean_subset)})")

############################################################################ Create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

############## OH - Habituators (top left)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == stepped_trial_types[1]) & 
    (context_analysis_df['classification'] == 'habituator')
]
clean_subset = subset.dropna(subset=['preceding_abs_max_val', 'abs_normalized_pain_change'])

axes[0,0].scatter(clean_subset['preceding_abs_max_val'], clean_subset['abs_normalized_pain_change'], 
                  alpha=0.6, color='blue', s=50)
axes[0,0].set_xlabel('Preceding Max Pain')
axes[0,0].set_ylabel('OH Magnitude (%)')
axes[0,0].set_title('Onset Hyperalgesia - Habituators')

corr, p_val = stats.pearsonr(clean_subset['preceding_abs_max_val'], clean_subset['abs_normalized_pain_change'])
if p_val < 0.00125: # Bonferroni correction for 4 comparisons
    # Add trendline
    x = clean_subset['preceding_abs_max_val']
    y = clean_subset['abs_normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0,0].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
# Add stats box
axes[0,0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=axes[0,0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

############## OH - Sensitizers (top right)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == stepped_trial_types[1]) & 
    (context_analysis_df['classification'] == 'sensitizer')
]
clean_subset = subset.dropna(subset=['preceding_abs_max_val', 'abs_normalized_pain_change'])

axes[0,1].scatter(clean_subset['preceding_abs_max_val'], clean_subset['abs_normalized_pain_change'], 
                  alpha=0.6, color='red', s=50)
axes[0,1].set_xlabel('Preceding Max Pain')
axes[0,1].set_ylabel('OH Magnitude (%)')
axes[0,1].set_title('Onset Hyperalgesia - Sensitizers')

corr, p_val = stats.pearsonr(clean_subset['preceding_abs_max_val'], clean_subset['abs_normalized_pain_change'])
if p_val < 0.00125: # Bonferroni correction for 4 comparisons
    # Add trendline
    x = clean_subset['preceding_abs_max_val']
    y = clean_subset['abs_normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[0,1].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
# Add stats box
axes[0,1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=axes[0,1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

############## OA - Habituators (bottom left)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == stepped_trial_types[0]) & 
    (context_analysis_df['classification'] == 'habituator')
]
clean_subset = subset.dropna(subset=['preceding_abs_min_val', 'abs_normalized_pain_change'])

axes[1,0].scatter(clean_subset['preceding_abs_min_val'], clean_subset['abs_normalized_pain_change'], 
                  alpha=0.6, color='blue', s=50)
axes[1,0].set_xlabel('Preceding Min Pain')
axes[1,0].set_ylabel('OA Magnitude (%)')
axes[1,0].set_title('Offset Analgesia - Habituators')

corr, p_val = stats.pearsonr(clean_subset['preceding_abs_min_val'], clean_subset['abs_normalized_pain_change'])
if p_val < 0.05:
    # Add trendline
    x = clean_subset['preceding_abs_min_val']
    y = clean_subset['abs_normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1,0].plot(x, p(x), "r-", alpha=0.8, linewidth=2)

# Add stats box (even if not significant, for comparison)
axes[1,0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
               transform=axes[1,0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round", facecolor="lightgray" if p_val >= 0.05 else "wheat", alpha=0.8))

############### OA - Sensitizers (bottom right)
subset = context_analysis_df[
    (context_analysis_df['trial_type'] == stepped_trial_types[0]) & 
    (context_analysis_df['classification'] == 'sensitizer')
]
clean_subset = subset.dropna(subset=['preceding_abs_min_val', 'abs_normalized_pain_change'])

axes[1,1].scatter(clean_subset['preceding_abs_min_val'], clean_subset['abs_normalized_pain_change'], 
                  alpha=0.6, color='red', s=50)
axes[1,1].set_xlabel('Preceding Min Pain')
axes[1,1].set_ylabel('OA Magnitude (%)')
axes[1,1].set_title('Offset Analgesia - Sensitizers')

corr, p_val = stats.pearsonr(clean_subset['preceding_abs_min_val'], clean_subset['abs_normalized_pain_change'])
if p_val < 0.05:
    # Add trendline
    x = clean_subset['preceding_abs_min_val']
    y = clean_subset['abs_normalized_pain_change']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[1,1].plot(x, p(x), "r-", alpha=0.8, linewidth=2)
    
    # Add stats box
    axes[1,1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                   transform=axes[1,1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

plt.tight_layout()
plt.show()
plt.savefig(f'{FIG_PATH}preceding_min_max_effects_by_classification.svg', dpi=300, bbox_inches='tight')


#%%
# =========================================================================
# Overall difference in OH/OA magnitude between HABITUATORS vs. SENSITIZERS
# =========================================================================

# Filter for OH/OA trials only and merge with classification
oh_oa_data = trial_metrics_df[
    (trial_metrics_df['trial_type'].isin(stepped_trial_types)) &
    (trial_metrics_df['classification'].isin(['habituator', 'sensitizer']))
].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# OH (inv) trials
oh_data = oh_oa_data[oh_oa_data['trial_type'] == stepped_trial_types[1]] # 'inv' for plosONE, 'onset' for kneeOA
if len(oh_data) > 0:
    sns.violinplot(data=oh_data, x='classification', y='abs_normalized_pain_change',
                   palette={'habituator': 'blue', 'sensitizer': 'red'}, inner='box', ax=axes[0])
    axes[0].set_title('Onset Hyperalgesia (OH) Magnitude')
    axes[0].set_xlabel('Classification')
    axes[0].set_ylabel('OH Magnitude (%)')
    
    # Simple t-test
    hab_oh = oh_data[oh_data['classification'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sen_oh = oh_data[oh_data['classification'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(hab_oh) > 0 and len(sen_oh) > 0:
        t_stat, p_val = stats.ttest_ind(hab_oh, sen_oh)
        print(f"OH: Habituators mean={hab_oh.mean():.2f}, Sensitizers mean={sen_oh.mean():.2f}, p={p_val:.3f}")

# OA (offset) trials  
oa_data = oh_oa_data[oh_oa_data['trial_type'] == stepped_trial_types[0]] # 'offset' for both datasets
if len(oa_data) > 0:
    sns.violinplot(data=oa_data, x='classification', y='abs_normalized_pain_change',
                   palette={'habituator': 'blue', 'sensitizer': 'red'}, inner='box', ax=axes[1])
    axes[1].set_title('Offset Analgesia (OA) Magnitude')
    axes[1].set_xlabel('Classification')
    axes[1].set_ylabel('OA Magnitude (%)')
    
    # Simple t-test
    hab_oa = oa_data[oa_data['classification'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sen_oa = oa_data[oa_data['classification'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(hab_oa) > 0 and len(sen_oa) > 0:
        t_stat, p_val = stats.ttest_ind(hab_oa, sen_oa)
        print(f"OA: Habituators mean={hab_oa.mean():.2f}, Sensitizers mean={sen_oa.mean():.2f}, p={p_val:.3f}")

plt.tight_layout()
plt.show()


#%%
# ==================================================================
# Save classification calculations 
# ==================================================================

classification_output_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/classification.csv'
classification_df.to_csv(classification_output_path, index=False)
print(f"Saved subject classifications to {classification_output_path}")


#%%
# =========================================================
# GET GROUP INFO DIRECTLY FROM SQL DATABASE
# =========================================================

import sqlite3

sql_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_data.sqlite'

print(f"=== QUERYING SQL DATABASE FOR GROUP INFO ===")

try:
    conn = sqlite3.connect(sql_path)
    
    # Simple query to get unique subject-group mappings
    query = '''
    SELECT DISTINCT 
        subject,
        COALESCE(NULLIF("group", ""), 'control') AS group_label
    FROM metadata 
    WHERE study = ?
    ORDER BY subject
    '''
    
    subject_groups_df = pd.read_sql_query(query, conn, params=(dataset,))
    conn.close()
    
    print(f"✅ Retrieved group info for {len(subject_groups_df)} subjects")
    print(f"Group distribution:")
    print(subject_groups_df['group_label'].value_counts())
    
    # Show first few rows
    print(f"\nFirst 10 subjects:")
    print(subject_groups_df.head(10))
    
    # Merge with trial_metrics_df
    print(f"\nMerging with trial_metrics_df...")
    print(f"Before merge: trial_metrics_df shape = {trial_metrics_df.shape}")
    
    trial_metrics_df = trial_metrics_df.merge(
        subject_groups_df, 
        on='subject', 
        how='left'
    )
    
    print(f"After merge: trial_metrics_df shape = {trial_metrics_df.shape}")
    
    # Check for missing group labels
    missing_groups = trial_metrics_df['group_label'].isna().sum()
    if missing_groups > 0:
        print(f"⚠️ {missing_groups} trials missing group labels")
        missing_subjects = trial_metrics_df[trial_metrics_df['group_label'].isna()]['subject'].unique()
        print(f"Subjects missing group labels: {missing_subjects}")
    else:
        print(f"✅ All subjects have group labels!")
    
    print(f"\n🎉 SUCCESS! Final group distribution in trial_metrics_df:")
    print(trial_metrics_df['group_label'].value_counts())
    
except Exception as e:
    print(f"❌ Error querying database: {e}")
#%%
# =========================================================
# PLOT SLOPE DISTRIBUTION WITH CLINICAL GROUPS OVERLAID
# =========================================================

# Make sure we have the slopes_df with group_label
# Merge group_label into slopes_df if not already there
if 'group_label' not in slopes_df.columns:
    slopes_df = slopes_df.merge(
        trial_metrics_df[['subject', 'group_label']].drop_duplicates(), 
        on='subject', 
        how='left'
    )

print("=== SLOPE DISTRIBUTION BY CLINICAL GROUP ===")
print("Slopes per group:")
print(slopes_df['group_label'].value_counts())

# Create the plot
plt.figure(figsize=(12, 8))

# Define colors for each group
group_colors = {
    'control': 'blue',
    'low_pain': 'green', 
    'high_pain': 'red'
}

# Plot histogram for each group
for group in slopes_df['group_label'].unique():
    group_data = slopes_df[slopes_df['group_label'] == group]['slope']
    plt.hist(group_data, bins=15, alpha=0.6, 
             color=group_colors.get(group, 'gray'), 
             label=f'{group} (n={len(group_data)})',
             edgecolor='black', linewidth=0.5)

# Add reference line at zero
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No change')

# Add labels and formatting
plt.xlabel('Max Pain Slope (points per trial)', fontsize=12)
plt.ylabel('Number of Subjects', fontsize=12)
plt.title('Distribution of Individual Pain Classifications by Clinical Group', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add some statistics text
stats_text = []
for group in sorted(slopes_df['group_label'].unique()):
    group_data = slopes_df[slopes_df['group_label'] == group]['slope']
    mean_slope = group_data.mean()
    std_slope = group_data.std()
    stats_text.append(f'{group}: μ={mean_slope:.2f}, σ={std_slope:.2f}')

plt.text(0.02, 0.98, '\n'.join(stats_text), 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         fontsize=10)

plt.tight_layout()
plt.savefig(f'{FIG_PATH}slope_distribution_by_clinical_group.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\n=== DETAILED STATISTICS BY GROUP ===")
for group in sorted(slopes_df['group_label'].unique()):
    group_data = slopes_df[slopes_df['group_label'] == group]['slope']
    print(f"\n{group.upper()}:")
    print(f"  N subjects: {len(group_data)}")
    print(f"  Mean slope: {group_data.mean():.3f}")
    print(f"  Std slope: {group_data.std():.3f}")
    print(f"  Sensitizers (slope > 0): {(group_data > 0).sum()} ({(group_data > 0).mean()*100:.1f}%)")
    print(f"  Habituators (slope < 0): {(group_data < 0).sum()} ({(group_data < 0).mean()*100:.1f}%)")

# Statistical comparison between groups
from scipy import stats

print(f"\n=== STATISTICAL COMPARISONS ===")
groups = sorted(slopes_df['group_label'].unique())
for i, group1 in enumerate(groups):
    for group2 in groups[i+1:]:
        data1 = slopes_df[slopes_df['group_label'] == group1]['slope']
        data2 = slopes_df[slopes_df['group_label'] == group2]['slope']
        
        t_stat, p_val = stats.ttest_ind(data1, data2)
        print(f"{group1} vs {group2}: t={t_stat:.3f}, p={p_val:.3f}")