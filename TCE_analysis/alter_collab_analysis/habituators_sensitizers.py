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
# ==================================================================
# Save trajectory calculations 
# ==================================================================

trajectory_output_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trajectory_classification.csv'
trajectory_df.to_csv(trajectory_output_path, index=False)
print(f"Saved subject trajectories to {trajectory_output_path}")


# %%
"""
EVERYTHING BELOW STRAIGHT FROM CLAUDE PLEASE VERIFY
"""
# ============================================================================
# LME ANALYSIS WITH TRAJECTORY GROUPS - Detecting Signal Within Groups
# ============================================================================

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LME ANALYSIS: TRIAL ORDER EFFECTS BY TRAJECTORY GROUP")
print("=" * 80)

# Prepare data - only include subjects with trajectory classifications
lme_trajectory_data = trial_metrics_df[
    trial_metrics_df['trajectory_group'].isin(['habituator', 'sensitizer'])
].copy()

# Center trial_num for better interpretation
lme_trajectory_data['trial_num_centered'] = (
    lme_trajectory_data['trial_num'] - lme_trajectory_data['trial_num'].mean()
)

print(f"Data summary:")
print(f"- Habituators: {(lme_trajectory_data['trajectory_group'] == 'habituator').sum()} trials, "
      f"{lme_trajectory_data[lme_trajectory_data['trajectory_group'] == 'habituator']['subject'].nunique()} subjects")
print(f"- Sensitizers: {(lme_trajectory_data['trajectory_group'] == 'sensitizer').sum()} trials, "
      f"{lme_trajectory_data[lme_trajectory_data['trajectory_group'] == 'sensitizer']['subject'].nunique()} subjects")

# ============================================================================
# Model 1: Trial order effects within each trajectory group
# ============================================================================

print("\n" + "="*60)
print("MODEL 1: TRIAL ORDER EFFECTS WITHIN TRAJECTORY GROUPS")
print("="*60)

pain_metrics = ['abs_max_val', 'abs_normalized_pain_change', 'auc_total']

for metric in pain_metrics:
    print(f"\n--- {metric.upper()} ---")
    
    # Test each trajectory group separately first
    for group in ['habituator', 'sensitizer']:
        subset = lme_trajectory_data[lme_trajectory_data['trajectory_group'] == group]
        
        if len(subset) > 20:  # Need sufficient data
            try:
                # Simple model: metric ~ trial_num + (1 + trial_num | subject)
                model = mixedlm(f"{metric} ~ trial_num_centered", 
                              data=subset,
                              groups=subset["subject"],
                              re_formula="~trial_num_centered")
                
                result = model.fit(reml=False)
                
                params = result.params
                pvalues = result.pvalues
                
                if 'trial_num_centered' in pvalues.index:
                    trial_effect_p = pvalues['trial_num_centered']
                    trial_effect_coef = params['trial_num_centered']
                    sig = "***" if trial_effect_p < 0.001 else "**" if trial_effect_p < 0.01 else "*" if trial_effect_p < 0.05 else "ns"
                    
                    print(f"  {group.upper()}: β = {trial_effect_coef:.4f}, p = {trial_effect_p:.4f} {sig}")
                    
                    if trial_effect_p < 0.05:
                        direction = "increases" if trial_effect_coef > 0 else "decreases"
                        print(f"    → {metric} {direction} over trials in {group}s")
                
            except Exception as e:
                print(f"  {group.upper()}: Model failed - {e}")

# ============================================================================
# Model 2: Test interaction between trajectory group and trial order
# ============================================================================

print("\n" + "="*60)
print("MODEL 2: TRAJECTORY GROUP × TRIAL ORDER INTERACTIONS")
print("="*60)

for metric in pain_metrics:
    if metric in lme_trajectory_data.columns:
        print(f"\n--- {metric.upper()} ---")
        
        try:
            # Full interaction model
            model = mixedlm(f"{metric} ~ trial_num_centered * trajectory_group", 
                          data=lme_trajectory_data,
                          groups=lme_trajectory_data["subject"],
                          re_formula="~trial_num_centered")
            
            result = model.fit(reml=False)
            
            params = result.params
            pvalues = result.pvalues
            
            print(f"Model: {metric} ~ trial_num_centered * trajectory_group + (1 + trial_num_centered | subject)")
            print(f"AIC: {result.aic:.2f}")
            
            # Main effects and interaction
            key_effects = [
                ('trial_num_centered', 'Main effect of trial order'),
                ('trajectory_group[T.sensitizer]', 'Main effect of trajectory group'),
                ('trial_num_centered:trajectory_group[T.sensitizer]', 'Interaction: trial order × trajectory group')
            ]
            
            for param, description in key_effects:
                if param in params.index:
                    coef = params[param]
                    p = pvalues[param]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"  {description}: β = {coef:.4f}, p = {p:.4f} {sig}")
            
            # Interpretation of interaction
            interaction_param = 'trial_num_centered:trajectory_group[T.sensitizer]'
            if interaction_param in pvalues.index:
                interaction_p = pvalues[interaction_param]
                interaction_coef = params[interaction_param]
                
                if interaction_p < 0.05:
                    print(f"\n  → SIGNIFICANT INTERACTION DETECTED!")
                    print(f"    Habituators and sensitizers show different trial order effects")
                    
                    # Calculate slopes for each group
                    habituator_slope = params['trial_num_centered'] if 'trial_num_centered' in params.index else 0
                    sensitizer_slope = habituator_slope + interaction_coef
                    
                    print(f"    Habituator slope: {habituator_slope:.4f}")
                    print(f"    Sensitizer slope: {sensitizer_slope:.4f}")
                    
                    if habituator_slope < 0 and sensitizer_slope > 0:
                        print(f"    → Classic pattern: habituators decrease, sensitizers increase over trials")
                    elif abs(habituator_slope) > abs(sensitizer_slope):
                        print(f"    → Habituators show stronger trial order effects")
                    else:
                        print(f"    → Sensitizers show stronger trial order effects")
                else:
                    print(f"\n  → No significant interaction (groups show similar trial order patterns)")
                    
        except Exception as e:
            print(f"Model failed for {metric}: {e}")

# ============================================================================
# Model 3: Stepped vs Control trials within trajectory groups
# ============================================================================

print("\n" + "="*60)
print("MODEL 3: STEPPED vs CONTROL BY TRAJECTORY GROUP")
print("="*60)

# Add stepped vs control indicator
lme_trajectory_data['is_stepped'] = lme_trajectory_data['trial_type'].isin(['offset', 'inv']).astype(int)

try:
    # Three-way interaction model
    model = mixedlm("abs_max_val ~ trial_num_centered * trajectory_group * is_stepped", 
                    data=lme_trajectory_data,
                    groups=lme_trajectory_data["subject"])
    
    result = model.fit(reml=False)
    
    params = result.params
    pvalues = result.pvalues
    
    print("Three-way interaction: trial_num × trajectory_group × trial_type")
    print(f"AIC: {result.aic:.2f}")
    
    # Key interactions to examine
    key_interactions = [
        ('trial_num_centered:trajectory_group[T.sensitizer]', 'Trial order × trajectory group'),
        ('trial_num_centered:is_stepped', 'Trial order × trial type (stepped vs control)'),
        ('trial_num_centered:trajectory_group[T.sensitizer]:is_stepped', 'Three-way interaction')
    ]
    
    for param, description in key_interactions:
        if param in params.index:
            coef = params[param]
            p = pvalues[param]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {description}: β = {coef:.4f}, p = {p:.4f} {sig}")
    
    # Three-way interaction interpretation
    threeway_param = 'trial_num_centered:trajectory_group[T.sensitizer]:is_stepped'
    if threeway_param in pvalues.index and pvalues[threeway_param] < 0.05:
        print(f"\n  → THREE-WAY INTERACTION SIGNIFICANT!")
        print(f"    Trial order effects differ between habituators/sensitizers AND")
        print(f"    this difference is different for stepped vs control trials")

except Exception as e:
    print(f"Three-way model failed: {e}")

# ============================================================================
# Visualization: Trial order effects by trajectory group
# ============================================================================

print("\n" + "="*60)
print("VISUALIZING TRAJECTORY GROUP DIFFERENCES")
print("="*60)

# Plot 1: Pain intensity trajectories by group
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colors for trajectory groups
colors = {'habituator': '#4472C4', 'sensitizer': '#E70000', 'no_trend': '#70AD47'}

# Plot 1: All trials
for group in ['habituator', 'sensitizer']:
    subset = lme_trajectory_data[lme_trajectory_data['trajectory_group'] == group]
    
    if len(subset) > 0:
        # Calculate mean and SEM for each trial number
        trial_summary = subset.groupby('trial_num').agg({
            'abs_max_val': ['mean', 'sem', 'count']
        }).round(3)
        
        trial_summary.columns = ['mean', 'sem', 'count']
        trial_summary = trial_summary[trial_summary['count'] >= 3]
        
        if len(trial_summary) > 0:
            x = trial_summary.index
            y = trial_summary['mean']
            yerr = trial_summary['sem']
            
            axes[0].errorbar(x, y, yerr=yerr, marker='o', linestyle='-', 
                           color=colors[group], label=f'{group.title()}s (n={subset["subject"].nunique()})',
                           capsize=3, capthick=1, linewidth=2, markersize=6)

axes[0].set_xlabel('Trial Number')
axes[0].set_ylabel('Pain Intensity (Max Value)')
axes[0].set_title('All Trials: Pain Trajectories by Group')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Stepped trials only
stepped_data = lme_trajectory_data[lme_trajectory_data['is_stepped'] == 1]

for group in ['habituator', 'sensitizer']:
    subset = stepped_data[stepped_data['trajectory_group'] == group]
    
    if len(subset) > 0:
        trial_summary = subset.groupby('trial_num').agg({
            'abs_max_val': ['mean', 'sem', 'count']
        }).round(3)
        
        trial_summary.columns = ['mean', 'sem', 'count']
        trial_summary = trial_summary[trial_summary['count'] >= 2]
        
        if len(trial_summary) > 0:
            x = trial_summary.index
            y = trial_summary['mean']
            yerr = trial_summary['sem']
            
            axes[1].errorbar(x, y, yerr=yerr, marker='o', linestyle='-', 
                           color=colors[group], label=f'{group.title()}s',
                           capsize=3, capthick=1, linewidth=2, markersize=6)

axes[1].set_xlabel('Trial Number')
axes[1].set_ylabel('Pain Intensity (Max Value)')
axes[1].set_title('Stepped Trials Only')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Control trials only
control_data = lme_trajectory_data[lme_trajectory_data['is_stepped'] == 0]

for group in ['habituator', 'sensitizer']:
    subset = control_data[control_data['trajectory_group'] == group]
    
    if len(subset) > 0:
        trial_summary = subset.groupby('trial_num').agg({
            'abs_max_val': ['mean', 'sem', 'count']
        }).round(3)
        
        trial_summary.columns = ['mean', 'sem', 'count']
        trial_summary = trial_summary[trial_summary['count'] >= 2]
        
        if len(trial_summary) > 0:
            x = trial_summary.index
            y = trial_summary['mean']
            yerr = trial_summary['sem']
            
            axes[2].errorbar(x, y, yerr=yerr, marker='o', linestyle='-', 
                           color=colors[group], label=f'{group.title()}s',
                           capsize=3, capthick=1, linewidth=2, markersize=6)

axes[2].set_xlabel('Trial Number')
axes[2].set_ylabel('Pain Intensity (Max Value)')
axes[2].set_title('Control Trials Only')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_PATH}trajectory_groups_trial_order_effects.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# Individual subject examples
# ============================================================================

print("\n" + "="*60)
print("INDIVIDUAL SUBJECT EXAMPLES")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Get example subjects with sufficient data
def get_example_subject(group, min_trials=6):
    candidates = lme_trajectory_data[lme_trajectory_data['trajectory_group'] == group]
    trial_counts = candidates.groupby('subject').size()
    suitable_subjects = trial_counts[trial_counts >= min_trials].index
    
    if len(suitable_subjects) > 0:
        return suitable_subjects[0]  # Return first suitable subject
    return None

# Example habituator - all trials
example_habituator = get_example_subject('habituator')
if example_habituator:
    subj_data = lme_trajectory_data[lme_trajectory_data['subject'] == example_habituator].sort_values('trial_num')
    
    axes[0,0].scatter(subj_data['trial_num'], subj_data['abs_max_val'], 
                     alpha=0.7, s=60, color=colors['habituator'])
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subj_data['trial_num'], subj_data['abs_max_val'])
    
    x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
    y_vals = intercept + slope * x_vals
    axes[0,0].plot(x_vals, y_vals, 'r-', linewidth=2)
    
    axes[0,0].set_xlabel('Trial Number')
    axes[0,0].set_ylabel('Pain Intensity')
    axes[0,0].set_title(f'Example Habituator: Subject {example_habituator}\nSlope = {slope:.3f}, r² = {r_value**2:.3f}')
    axes[0,0].grid(True, alpha=0.3)

# Example sensitizer - all trials
example_sensitizer = get_example_subject('sensitizer')
if example_sensitizer:
    subj_data = lme_trajectory_data[lme_trajectory_data['subject'] == example_sensitizer].sort_values('trial_num')
    
    axes[0,1].scatter(subj_data['trial_num'], subj_data['abs_max_val'], 
                     alpha=0.7, s=60, color=colors['sensitizer'])
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subj_data['trial_num'], subj_data['abs_max_val'])
    
    x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
    y_vals = intercept + slope * x_vals
    axes[0,1].plot(x_vals, y_vals, 'r-', linewidth=2)
    
    axes[0,1].set_xlabel('Trial Number')
    axes[0,1].set_ylabel('Pain Intensity')
    axes[0,1].set_title(f'Example Sensitizer: Subject {example_sensitizer}\nSlope = {slope:.3f}, r² = {r_value**2:.3f}')
    axes[0,1].grid(True, alpha=0.3)

# Habituator - stepped trials only
if example_habituator:
    subj_data = lme_trajectory_data[
        (lme_trajectory_data['subject'] == example_habituator) & 
        (lme_trajectory_data['is_stepped'] == 1)
    ].sort_values('trial_num')
    
    if len(subj_data) > 2:
        axes[1,0].scatter(subj_data['trial_num'], subj_data['abs_max_val'], 
                         alpha=0.7, s=60, color=colors['habituator'])
        
        if len(subj_data) > 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                subj_data['trial_num'], subj_data['abs_max_val'])
            
            x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
            y_vals = intercept + slope * x_vals
            axes[1,0].plot(x_vals, y_vals, 'r-', linewidth=2)
            
            axes[1,0].set_title(f'Habituator - Stepped Trials Only\nSlope = {slope:.3f}')
        else:
            axes[1,0].set_title(f'Habituator - Stepped Trials Only')
    
    axes[1,0].set_xlabel('Trial Number')
    axes[1,0].set_ylabel('Pain Intensity')
    axes[1,0].grid(True, alpha=0.3)

# Sensitizer - stepped trials only
if example_sensitizer:
    subj_data = lme_trajectory_data[
        (lme_trajectory_data['subject'] == example_sensitizer) & 
        (lme_trajectory_data['is_stepped'] == 1)
    ].sort_values('trial_num')
    
    if len(subj_data) > 2:
        axes[1,1].scatter(subj_data['trial_num'], subj_data['abs_max_val'], 
                         alpha=0.7, s=60, color=colors['sensitizer'])
        
        if len(subj_data) > 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                subj_data['trial_num'], subj_data['abs_max_val'])
            
            x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
            y_vals = intercept + slope * x_vals
            axes[1,1].plot(x_vals, y_vals, 'r-', linewidth=2)
            
            axes[1,1].set_title(f'Sensitizer - Stepped Trials Only\nSlope = {slope:.3f}')
        else:
            axes[1,1].set_title(f'Sensitizer - Stepped Trials Only')
    
    axes[1,1].set_xlabel('Trial Number')
    axes[1,1].set_ylabel('Pain Intensity')
    axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_PATH}individual_examples_trajectory_groups.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# Summary statistics by trajectory group
# ============================================================================

print("\n" + "="*60)
print("SUMMARY: TRAJECTORY GROUP TRIAL ORDER EFFECTS")
print("="*60)

# Calculate effect sizes within each group
for group in ['habituator', 'sensitizer']:
    subset = lme_trajectory_data[lme_trajectory_data['trajectory_group'] == group]
    
    print(f"\n{group.upper()}S:")
    print(f"  n = {subset['subject'].nunique()} subjects, {len(subset)} trials")
    
    # Overall correlation between trial number and pain
    if len(subset) > 10:
        r_overall, p_overall = stats.pearsonr(subset['trial_num'], subset['abs_max_val'])
        print(f"  Overall trial-pain correlation: r = {r_overall:.3f}, p = {p_overall:.4f}")
        
        # Mean slope across subjects
        individual_slopes = []
        for subject in subset['subject'].unique():
            subj_data = subset[subset['subject'] == subject]
            if len(subj_data) > 3:
                slope, _, _, p_val, _ = stats.linregress(subj_data['trial_num'], subj_data['abs_max_val'])
                individual_slopes.append(slope)
        
        if individual_slopes:
            mean_slope = np.mean(individual_slopes)
            std_slope = np.std(individual_slopes)
            print(f"  Mean individual slope: {mean_slope:.3f} ± {std_slope:.3f}")
            
            # Test if slopes are significantly different from zero
            t_stat, p_slope = stats.ttest_1samp(individual_slopes, 0)
            sig = "***" if p_slope < 0.001 else "**" if p_slope < 0.01 else "*" if p_slope < 0.05 else "ns"
            print(f"  Slopes vs zero: t = {t_stat:.3f}, p = {p_slope:.4f} {sig}")

# Compare slope distributions between groups
habituator_slopes = []
sensitizer_slopes = []

for group, slope_list in [('habituator', habituator_slopes), ('sensitizer', sensitizer_slopes)]:
    subset = lme_trajectory_data[lme_trajectory_data['trajectory_group'] == group]
    
    for subject in subset['subject'].unique():
        subj_data = subset[subset['subject'] == subject]
        if len(subj_data) > 3:
            slope, _, _, _, _ = stats.linregress(subj_data['trial_num'], subj_data['abs_max_val'])
            slope_list.append(slope)

# Compare slopes between groups
if len(habituator_slopes) > 0 and len(sensitizer_slopes) > 0:
    t_stat, p_between = stats.ttest_ind(habituator_slopes, sensitizer_slopes)
    
    print(f"\nCOMPARISON BETWEEN GROUPS:")
    print(f"  Habituator slopes: {np.mean(habituator_slopes):.3f} ± {np.std(habituator_slopes):.3f} (n={len(habituator_slopes)})")
    print(f"  Sensitizer slopes: {np.mean(sensitizer_slopes):.3f} ± {np.std(sensitizer_slopes):.3f} (n={len(sensitizer_slopes)})")
    print(f"  Between-group difference: t = {t_stat:.3f}, p = {p_between:.4f}")
    
    if p_between < 0.05:
        print(f"  → SIGNIFICANT DIFFERENCE in trial order effects between groups!")
    else:
        print(f"  → No significant difference in trial order effects between groups")

# ============================================================================
# Test if trajectory groups show different contrast enhancement patterns
# ============================================================================

print("\n" + "="*60)
print("CONTRAST ENHANCEMENT BY TRAJECTORY GROUP")
print("="*60)

# Test if habituators vs sensitizers show different offset analgesia / onset hyperalgesia
stepped_trajectory_data = lme_trajectory_data[lme_trajectory_data['is_stepped'] == 1].copy()

for trial_type, effect_name in [('offset', 'Offset Analgesia'), ('inv', 'Onset Hyperalgesia')]:
    print(f"\n{effect_name.upper()}:")
    
    subset = stepped_trajectory_data[stepped_trajectory_data['trial_type'] == trial_type]
    
    # Compare normalized pain change between groups
    habituator_data = subset[subset['trajectory_group'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sensitizer_data = subset[subset['trajectory_group'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(habituator_data) > 0 and len(sensitizer_data) > 0:
        # Independent t-test
        t_stat, p_val = stats.ttest_ind(habituator_data, sensitizer_data)
        
        print(f"  Habituators: {habituator_data.mean():.1f}% ± {habituator_data.std():.1f}% (n={len(habituator_data)})")
        print(f"  Sensitizers: {sensitizer_data.mean():.1f}% ± {sensitizer_data.std():.1f}% (n={len(sensitizer_data)})")
        print(f"  Difference: t = {t_stat:.3f}, p = {p_val:.4f}")
        
        if p_val < 0.05:
            if trial_type == 'offset' and habituator_data.mean() < sensitizer_data.mean():
                print(f"  → Habituators show STRONGER offset analgesia!")
            elif trial_type == 'inv' and sensitizer_data.mean() > habituator_data.mean():
                print(f"  → Sensitizers show STRONGER onset hyperalgesia!")
            else:
                direction = "stronger" if habituator_data.mean() > sensitizer_data.mean() else "weaker"
                print(f"  → Habituators show {direction} {effect_name.lower()}")
