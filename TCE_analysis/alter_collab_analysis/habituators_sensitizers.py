#%%
# ========================================================
# CONFIGURATION
# ========================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from scipy import stats
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_metrics.json'
TRIAL_DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json'
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
# %% 
# =====================================================================
# Determine habituators vs. sensitizers based on pain trajectory
# =====================================================================
def calculate_pain_trajectory(subject_data):
    # Use the max pain column
    clean_data = subject_data.dropna(subset=['abs_max_val', 'trial_num'])
    
    if len(clean_data) < 3:
        return np.nan, np.nan, np.nan
    
    if clean_data['abs_max_val'].var() == 0:
        return 0.0, np.nan, np.nan
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            clean_data['trial_num'],
            clean_data['abs_max_val']
        )
        return slope, r_value, p_value
    except:
        return np.nan, np.nan, np.nan

# Calculate trajectories
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
            subj_data['abs_max_val']
        )
        subject_slopes.append({
            'subject': subject,
            'slope': slope,
            'r_value': r_value,
            'p_value': p_value
        })
slopes_df = pd.DataFrame(subject_slopes)

# Plot distribution of slopes
plt.figure(figsize=(8, 6))
plt.hist(slopes_df['slope'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='No change')
plt.xlabel('Max Pain Slope (points per trial)')
plt.ylabel('Number of Subjects')
plt.title('Distribution of Individual Pain Trajectories')
plt.legend()
plt.savefig(f'{FIG_PATH}distribution_of_individual_pain_trajectories.png', dpi=300, bbox_inches='tight')
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
# ==========================================================================
# Split into HABITUATORS vs. SENSITIZERS and PLOT EXAMPLE for each 
# ==========================================================================

###################################################################################### Classify subjects
threshold = 1 # for now, set a threshold for slope classification (eventually in some data-driven way)
def classify_trajectory(slope):
    if slope < -threshold:
        return 'habituator'
    elif slope > threshold:
        return 'sensitizer'
    else:
        return 'no_trend'
    
trajectory_df['trajectory_group'] = trajectory_df['slope'].apply(classify_trajectory)
print(trajectory_df['trajectory_group'].value_counts())

# Merge with trial_metrics_df
trial_metrics_df = trial_metrics_df.merge(
    trajectory_df[['subject', 'trajectory_group']], 
    on='subject', 
    how='left'
)
####################################################################################### Plot example subject for each group
# Example habituator
example_subject = trajectory_df[trajectory_df['slope'] < -3]['subject'].iloc[0]
subj_data = trial_metrics_df[trial_metrics_df['subject'] == example_subject].sort_values('trial_num')
plt.figure(figsize=(8, 6))
plt.scatter(subj_data['trial_num'], subj_data['abs_max_val'], alpha=0.7, s=60)
subject_slope_info = trajectory_df[trajectory_df['subject'] == example_subject].iloc[0]
slope = subject_slope_info['slope']
r_value = subject_slope_info['r_value']
p_value = subject_slope_info['p_value']
# Add regression line
x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
y_vals = subj_data['abs_max_val'].iloc[0] + slope * (x_vals - subj_data['trial_num'].iloc[0])  # Rough approximation
plt.plot(x_vals, y_vals, 'r-', linewidth=2)
plt.xlabel('Trial Number')
plt.ylabel('Maximum Pain Rating')
plt.ylim(0, 100)
plt.title(f'Example Habituator: Subject {example_subject}\nSlope = {slope:.2f}')
plt.show()

# Example sensitizer
example_subject = trajectory_df[trajectory_df['slope'] > 3]['subject'].iloc[0]
subj_data = trial_metrics_df[trial_metrics_df['subject'] == example_subject].sort_values('trial_num')
plt.figure(figsize=(8, 6))
plt.scatter(subj_data['trial_num'], subj_data['abs_max_val'], alpha=0.7, s=60)
subject_slope_info = trajectory_df[trajectory_df['subject'] == example_subject].iloc[0]
slope = subject_slope_info['slope']
r_value = subject_slope_info['r_value']
p_value = subject_slope_info['p_value']
# Add regression line
x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
y_vals = subj_data['abs_max_val'].iloc[0] + slope * (x_vals - subj_data['trial_num'].iloc[0])  # Rough approximation
plt.plot(x_vals, y_vals, 'r-', linewidth=2)
plt.xlabel('Trial Number')
plt.ylabel('Maximum Pain Rating')
plt.ylim(0, 100)
plt.title(f'Example Sensitizer: Subject {example_subject}\nSlope = {slope:.2f}')
plt.show()

# %%
# ===================================================================
# Preceding trial context analysis for HABITUATORS vs. SENSITIZERS
# ===================================================================

# Filter for just habituators and sensitizers, and OA/OH trials
context_analysis_df = trial_metrics_df[
    (trial_metrics_df['trajectory_group'].isin(['habituator', 'sensitizer'])) &
    (trial_metrics_df['trial_type'].isin(['offset', 'inv']))
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
        (context_analysis_df['trial_type'] == 'inv') & 
        (context_analysis_df['trajectory_group'] == group)
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
        (context_analysis_df['trial_type'] == 'offset') & 
        (context_analysis_df['trajectory_group'] == group)
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
    (context_analysis_df['trial_type'] == 'inv') & 
    (context_analysis_df['trajectory_group'] == 'habituator')
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
    (context_analysis_df['trial_type'] == 'inv') & 
    (context_analysis_df['trajectory_group'] == 'sensitizer')
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
    (context_analysis_df['trial_type'] == 'offset') & 
    (context_analysis_df['trajectory_group'] == 'habituator')
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
    (context_analysis_df['trial_type'] == 'offset') & 
    (context_analysis_df['trajectory_group'] == 'sensitizer')
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
plt.savefig(f'{FIG_PATH}preceding_min_max_effects_by_trajectory_group.svg', dpi=300, bbox_inches='tight')


#%%
# =========================================================================
# Overall difference in OH/OA magnitude between HABITUATORS vs. SENSITIZERS
# =========================================================================

# Filter for OH/OA trials only and merge with trajectory classifications
oh_oa_data = trial_metrics_df[
    (trial_metrics_df['trial_type'].isin(['offset', 'inv'])) &
    (trial_metrics_df['trajectory_group'].isin(['habituator', 'sensitizer']))
].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# OH (inv) trials
oh_data = oh_oa_data[oh_oa_data['trial_type'] == 'inv']
if len(oh_data) > 0:
    sns.violinplot(data=oh_data, x='trajectory_group', y='abs_normalized_pain_change',
                   palette={'habituator': 'blue', 'sensitizer': 'red'}, inner='box', ax=axes[0])
    axes[0].set_title('Onset Hyperalgesia (OH) Magnitude')
    axes[0].set_xlabel('Trajectory Group')
    axes[0].set_ylabel('OH Magnitude (%)')
    
    # Simple t-test
    hab_oh = oh_data[oh_data['trajectory_group'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sen_oh = oh_data[oh_data['trajectory_group'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(hab_oh) > 0 and len(sen_oh) > 0:
        t_stat, p_val = stats.ttest_ind(hab_oh, sen_oh)
        print(f"OH: Habituators mean={hab_oh.mean():.2f}, Sensitizers mean={sen_oh.mean():.2f}, p={p_val:.3f}")

# OA (offset) trials  
oa_data = oh_oa_data[oh_oa_data['trial_type'] == 'offset']
if len(oa_data) > 0:
    sns.violinplot(data=oa_data, x='trajectory_group', y='abs_normalized_pain_change',
                   palette={'habituator': 'blue', 'sensitizer': 'red'}, inner='box', ax=axes[1])
    axes[1].set_title('Offset Analgesia (OA) Magnitude')
    axes[1].set_xlabel('Trajectory Group')
    axes[1].set_ylabel('OA Magnitude (%)')
    
    # Simple t-test
    hab_oa = oa_data[oa_data['trajectory_group'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sen_oa = oa_data[oa_data['trajectory_group'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(hab_oa) > 0 and len(sen_oa) > 0:
        t_stat, p_val = stats.ttest_ind(hab_oa, sen_oa)
        print(f"OA: Habituators mean={hab_oa.mean():.2f}, Sensitizers mean={sen_oa.mean():.2f}, p={p_val:.3f}")

plt.tight_layout()
plt.show()


#%%
# ==================================================================
# Save trajectory calculations 
# ==================================================================

trajectory_output_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trajectory_classification.csv'
trajectory_df.to_csv(trajectory_output_path, index=False)
print(f"Saved subject trajectories to {trajectory_output_path}")

#%%
# ==================================================================================================================
# I noticed some subjects have positive abs_normalized_pain_change values for hold trials and others have negative
# Let's see if this relates to their habituator/sensitizer classification
# ==================================================================================================================





# %%
# ================================================================================================
# LINEAR MIXED EFFECTS ANALYSIS WITH TRAJECTORY GROUPS
# Testing whether habituators vs. sensitizers show different patterns in OH/OA magnitude over trials
# ================================================================================================
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("LME ANALYSIS: OH/OA MAGNITUDE OVER TRIALS BY TRAJECTORY GROUP")
print("=" * 60)

# Prepare data 
oh_oa_data['trial_num_centered'] = oh_oa_data['trial_num'] - oh_oa_data['trial_num'].mean() # center trial num 

model = mixedlm("abs_normalized_pain_change ~ trial_num_centered * trajectory_group",
                data = oh_oa_data,
                groups=oh_oa_data['subject'],
                re_formula="~trial_num_centered")
result = model.fit(reml=False) 
print(result.summary())

# %%
