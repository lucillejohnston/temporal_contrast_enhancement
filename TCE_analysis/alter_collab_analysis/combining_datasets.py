"""
This script combines the datasets and looks at clinical population vs. various metrics 
Updated 3/11/26 to include plosONE and kneeOA
Will have to update again when I get the cLBP dataset 
"""
#%%
# ==================================================================================================================
######################################## COMBINED DATASET ANALYSES ########################################
# ==================================================================================================================
import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns
print("=== LOADING AND COMBINING DATASETS ===")

datasets = ['kneeOA', 'plosONE']
combined_trial_metrics = []
combined_trial_data = []
FIGPATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures'
# load in the data and combine it
for dataset in datasets:
    print(f"\n--- Loading {dataset} dataset ---")

    trial_metrics_path = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_metrics.json'
    trial_data_path = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_data_cleaned_aligned.json'
    # Load trial metrics (structured format)
    with open(trial_metrics_path, 'r') as f:
        metrics_data = json.load(f)
    # Convert structured metrics to flat DataFrame
    metrics_records = []
    for subject_id, trials in metrics_data.items():
        for trial_num, trial_data in trials.items():
            record = {
                'dataset': dataset,
                'subject': int(subject_id),
                'trial_num': int(trial_num),
                **trial_data
            }
            metrics_records.append(record)
    
    metrics_df = pd.DataFrame(metrics_records)
    print(f"  Trial metrics: {len(metrics_df)} records")
    # Load trial time series data
    trial_data_df = pd.read_json(trial_data_path, orient='records')
    trial_data_df['dataset'] = dataset
    print(f"  Trial time series: {len(trial_data_df)} records")
    # Add to combined lists
    combined_trial_metrics.append(metrics_df)
    combined_trial_data.append(trial_data_df)

# Combine all datasets
print(f"\n--- Combining datasets ---")
all_trial_metrics = pd.concat(combined_trial_metrics, ignore_index=True)
all_trial_data = pd.concat(combined_trial_data, ignore_index=True)
print(f"Combined trial metrics: {len(all_trial_metrics)} records from {all_trial_metrics['subject'].nunique()} subjects")
print(f"Combined trial data: {len(all_trial_data)} records")
# Fix subject ID overlap by adding offset to kneeOA subjects
print(f"\n--- Fixing subject ID overlap ---")
KNEEOA_SUBJECT_OFFSET = 1000  # Add 1000 to kneeOA subjects to avoid overlap
CLBP_SUBJECT_OFFSET = 2000   # Will add 2000 to cLBP subjects to avoid overlap

# Update subject IDs in trial metrics
kneeoa_mask_metrics = all_trial_metrics['dataset'] == 'kneeOA'
all_trial_metrics.loc[kneeoa_mask_metrics, 'subject'] += KNEEOA_SUBJECT_OFFSET
# Update subject IDs in trial data  
kneeoa_mask_data = all_trial_data['dataset'] == 'kneeOA'
all_trial_data.loc[kneeoa_mask_data, 'subject'] += KNEEOA_SUBJECT_OFFSET
print(f"Updated subject IDs:")
print(f"  plosONE subjects: {all_trial_metrics[all_trial_metrics['dataset'] == 'plosONE']['subject'].min()}-{all_trial_metrics[all_trial_metrics['dataset'] == 'plosONE']['subject'].max()}")
print(f"  kneeOA subjects: {all_trial_metrics[all_trial_metrics['dataset'] == 'kneeOA']['subject'].min()}-{all_trial_metrics[all_trial_metrics['dataset'] == 'kneeOA']['subject'].max()}")
# Get kneeOA group labels from SQL database
import sqlite3
sql_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_data.sqlite'
conn = sqlite3.connect(sql_path)
kneeoa_groups_query = '''
SELECT DISTINCT 
    subject,
    COALESCE(NULLIF("group", ""), 'control') AS group_label
FROM metadata 
WHERE study = 'kneeOA'
ORDER BY subject
'''
kneeoa_groups = pd.read_sql_query(kneeoa_groups_query, conn)
conn.close()

# Apply the same subject ID offset to the group labels
kneeoa_groups['subject'] += KNEEOA_SUBJECT_OFFSET
kneeoa_groups['dataset'] = 'kneeOA'
# Create plosONE group labels (all controls)
plosone_subjects = all_trial_metrics[all_trial_metrics['dataset'] == 'plosONE']['subject'].unique()
plosone_groups = pd.DataFrame({
    'subject': plosone_subjects,
    'group_label': 'Control',
    'dataset': 'plosONE'
})
# Combine group labels
all_groups = pd.concat([kneeoa_groups, plosone_groups], ignore_index=True)
# Merge group labels with trial metrics
all_trial_metrics = all_trial_metrics.merge(
    all_groups[['subject', 'group_label']], 
    on='subject', 
    how='left'
)
# Check final group distribution
print(f"\nFinal group distribution:")
group_dist = all_trial_metrics.groupby(['dataset', 'group_label']).agg({
    'subject': 'nunique',
    'trial_num': 'count'
}).round()
group_dist.columns = ['n_subjects', 'n_trials']
print(group_dist)
# Overall group distribution (ignoring dataset)
print(f"\nOverall group distribution:")
overall_dist = all_trial_metrics.groupby('group_label').agg({
    'subject': 'nunique', 
    'trial_num': 'count'
}).round()
overall_dist.columns = ['n_subjects', 'n_trials']
print(overall_dist)
print(f"\nSubject ID overlap fixed and groups consolidated!")
#%%
# ==================================================================================================================
######################################## 0. COMBINE AND STANDARDIZE THE DATA ########################################
# ==================================================================================================================
# Create a copy for modification
unified_data = all_trial_metrics.copy()

# Standardize trial type names: 'inv' -> 'onset'
print("=== STANDARDIZING TRIAL TYPES ===")
print("Before standardization:")
print(unified_data['trial_type'].value_counts())

# Replace 'inv' with 'onset' for consistency
unified_data['trial_type'] = unified_data['trial_type'].replace('inv', 'onset')

print("\nAfter standardization:")
print(unified_data['trial_type'].value_counts())

# Also need to standardize the time_yoked column names
print("\n=== STANDARDIZING TIME-YOKED COLUMN NAMES ===")

# Find columns that reference 'inv' and rename them to 'onset'
inv_columns = [col for col in unified_data.columns if 'inv' in col]
print(f"Columns with 'inv' to rename: {inv_columns}")

# Create mapping for column renaming
column_mapping = {}
for col in inv_columns:
    new_col = col.replace('inv', 'onset')
    column_mapping[col] = new_col

# Rename columns
unified_data = unified_data.rename(columns=column_mapping)
print(f"Renamed {len(column_mapping)} columns")

# Define unified trial type categories
stepped_trials = ['onset', 'offset', 'stepdown']  # stepdown only in plosONE
control_trials = ['t1_hold', 't2_hold', 'innocuous']  # innocuous only in kneeOA
common_stepped_trials = ['onset', 'offset']  # These exist in both datasets
common_control_trials = ['t1_hold', 't2_hold']  # These exist in both datasets

print(f"\nUnified trial type categories:")
print(f"All stepped trials: {stepped_trials}")
print(f"All control trials: {control_trials}")
print(f"Common stepped trials: {common_stepped_trials}")
print(f"Common control trials: {common_control_trials}")

#%%
# ==================================================================================================================
######################################## 1. BASIC STATS ANALYSIS ########################################
# ==================================================================================================================
# Define consistent colors for clinical groups
GROUP_COLORS = {
    'Control': '#2E8B57',    # Green
    'Low': '#FF8C00',        # Orange  
    'High': '#DC143C'        # Red
}

############################################################################## Raw distributions and group comparisons
# Plot the raw data to get a sense of the distributions and group differences
# Create comprehensive comparison plots with violin plots and sample sizes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# Prepare data for plotting
plot_data = unified_data[unified_data['trial_type'].isin(['onset', 'offset'])].copy()

# Function to add sample sizes to plot
def add_sample_sizes(ax, data, x_col, hue_col):
    """Add sample size annotations to violin plot"""
    # Get unique combinations
    combinations = data.groupby([x_col, hue_col]).size().reset_index(name='n')
    
    # Position for text annotations
    x_positions = {trial: i for i, trial in enumerate(data[x_col].unique())}
    hue_positions = {group: i for i, group in enumerate(data[hue_col].unique())}
    
    # Calculate positions for each group
    n_groups = len(data[hue_col].unique())
    width = 0.8 / n_groups
    
    for _, row in combinations.iterrows():
        x_pos = x_positions[row[x_col]]
        hue_idx = hue_positions[row[hue_col]]
        
        # Adjust x position based on group
        adjusted_x = x_pos + (hue_idx - (n_groups-1)/2) * width * 0.8
        
        # Add text at bottom of plot
        ax.text(adjusted_x, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                f'n={row["n"]}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 1: Max pain by group and trial type
sns.violinplot(data=plot_data, x='trial_type', y='abs_max_val', hue='group_label', palette=GROUP_COLORS, 
               inner='box', ax=axes[0,0])
axes[0,0].set_title('Max Pain by Clinical Group and Trial Type')
axes[0,0].set_ylabel('Max Pain Rating')
add_sample_sizes(axes[0,0], plot_data, 'trial_type', 'group_label')

# Plot 2: Min pain by group and trial type
sns.violinplot(data=plot_data, x='trial_type', y='abs_min_val', hue='group_label', palette=GROUP_COLORS,
               inner='box', ax=axes[0,1])
axes[0,1].set_title('Min Pain by Clinical Group and Trial Type')
axes[0,1].set_ylabel('Min Pain Rating')
add_sample_sizes(axes[0,1], plot_data, 'trial_type', 'group_label')

# Plot 3: AUC by group and trial type
sns.violinplot(data=plot_data, x='trial_type', y='auc_total', hue='group_label', palette=GROUP_COLORS,
               inner='box', ax=axes[1,0])
axes[1,0].set_title('AUC Total by Clinical Group and Trial Type')
axes[1,0].set_ylabel('AUC Total')
add_sample_sizes(axes[1,0], plot_data, 'trial_type', 'group_label')

# Plot 4: Normalized pain change by group and trial type  
sns.violinplot(data=plot_data, x='trial_type', y='abs_normalized_pain_change', hue='group_label', palette=GROUP_COLORS,
               inner='box', ax=axes[1,1])
axes[1,1].set_title('Normalized Pain Change by Clinical Group')
axes[1,1].set_ylabel('Normalized Pain Change (%)')
add_sample_sizes(axes[1,1], plot_data, 'trial_type', 'group_label')

plt.tight_layout()
plt.savefig(f'{FIGPATH}/raw_distributions_by_group.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

#%%
# ========================================================
# Plot OH and OA magnitude by clinical group box plot w/ swarm
fig, ax = plt.subplots(figsize=(10, 6))
oh_oa_data = unified_data[unified_data['trial_type'].isin(['onset', 'offset'])].copy()
for trial_type in ['onset', 'offset']:
    subset = oh_oa_data[oh_oa_data['trial_type'] == trial_type]
    subset = subset[['group_label', 'abs_normalized_pain_change']].copy()
    subset['trial_type'] = trial_type
    if trial_type == 'onset':
        subset = subset.rename(columns={'abs_normalized_pain_change': 'OH_magnitude'})
    else:
        subset = subset.rename(columns={'abs_normalized_pain_change': 'OA_magnitude'})
    if trial_type == 'onset':
        oh_data = subset
    else:
        oa_data = subset

#%%
# ==================================================================================================================
######################################## 2. TRIAL SEQUENCES ########################################
# ==================================================================================================================
from statsmodels.stats.multitest import multipletests

# Calculate preceding trial metrics
def get_preceding_value(row, col, df):
    prev_trial_num = row['trial_num'] - 1
    subject = row['subject']
    prev_row = df[
        (df['subject'] == subject) & 
        (df['trial_num'] == prev_trial_num)
    ]
    if not prev_row.empty:
        return prev_row.iloc[0][col]
    return None

# Define preceding metrics to calculate
preceding_metrics = {
    'preceding_trial_type': 'trial_type',
    'preceding_abs_max_val': 'abs_max_val',
    'preceding_abs_min_val': 'abs_min_val',
    'preceding_abs_peak_to_peak': 'abs_peak_to_peak',
    'preceding_auc_total': 'auc_total',
    'preceding_abs_normalized_pain_change': 'abs_normalized_pain_change'
}

print("Calculating preceding trial metrics...")
for new_col, source_col in preceding_metrics.items():
    unified_data[new_col] = unified_data.apply(
        lambda row: get_preceding_value(row, source_col, unified_data), axis=1
    )

# Filter for onset/offset trials (your main contrast trials)
contrast_trials = unified_data[unified_data['trial_type'].isin(['onset', 'offset'])].copy()

#%%
# Analysis combinations
# NEW 3/17/26: Split up positive and negative values
analyses = [
    ('onset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('onset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
    # ('onset', 'preceding_abs_peak_to_peak', 'negative', 'Negative Peak to Peak'),
    # ('onset', 'preceding_abs_peak_to_peak', 'positive', 'Positive Peak to Peak'),
    ('offset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
#     ('offset', 'preceding_abs_peak_to_peak', 'negative', 'Negative Peak to Peak'),
#     ('offset', 'preceding_abs_peak_to_peak', 'positive', 'Positive Peak to Peak'),
]

print("Creating plots and collecting correlations...")
all_correlations = []
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

for idx, (trial_type, preceding_metric, direction, metric_label) in enumerate(analyses):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    # Y-axis: always 0 to 100 for onset, -100 to 0 for offset
    if trial_type == 'onset':
        ax.set_ylim(0, 101)
    else:
        ax.set_ylim(0, -101)
    # Filter data for this analysis with direction
    base_data = contrast_trials[
        (contrast_trials['trial_type'] == trial_type) & 
        (contrast_trials[preceding_metric].notna()) &
        (contrast_trials['abs_normalized_pain_change'].notna())
    ]
    # Apply direction filter
    if direction == 'positive':
        plot_data = base_data[base_data[preceding_metric] > 0]
        ax.set_xlim(0, 101)  # Focus on positive range
    elif direction == 'negative':
        plot_data = base_data[base_data[preceding_metric] < 0]
        ax.set_xlim(0, -101)  # Focus on negative range
    else:
        plot_data = base_data
    
    # Add total sample size in title
    total_n = len(plot_data)
    ax.set_title(f'{trial_type.title()} - {metric_label}\n(N={total_n})', fontweight='bold', fontsize=10)
    
    if len(plot_data) > 10:
        text_y_positions = [0.95, 0.85, 0.75]
        
        # Create scatter plot by group AND collect correlations
        for group_idx, group in enumerate(['Control', 'Low', 'High']):
            group_data = plot_data[plot_data['group_label'] == group]
            if len(group_data) > 3:
                # PLOT the scatter
                ax.scatter(group_data[preceding_metric], 
                          group_data['abs_normalized_pain_change'],
                          color=GROUP_COLORS[group], 
                          alpha=0.6, 
                          label=f'{group}',
                          s=50, edgecolors='black', linewidth=0.5)
                
                # # CALCULATE correlation
                # r, p = stats.pearsonr(group_data[preceding_metric], 
                #                      group_data['abs_normalized_pain_change'])
                
                # # STORE correlation for FDR correction later
                # all_correlations.append({
                #     'idx': idx,
                #     'trial_type': trial_type,
                #     'metric': preceding_metric,
                #     'direction': direction,
                #     'metric_label': metric_label,
                #     'group': group,
                #     'r': r,
                #     'p_raw': p,
                #     'n': len(group_data),
                #     'ax': ax,  # Store axis reference
                #     'group_idx': group_idx,
                #     'group_data': group_data  # Store data for regression line
                # })
    
    # Formatting
    ax.set_xlabel(f'{preceding_metric.replace("preceding_abs_", "").replace("_", " ").title()}')
    ax.set_ylabel('Current Normalized Pain Change (%)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

# NOW apply FDR correction and add regression lines/stats to existing plots
if all_correlations:
    p_values = [corr['p_raw'] for corr in all_correlations]
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
    
    for i, corr in enumerate(all_correlations):
        corr['p_corrected'] = p_corrected[i]
        corr['significant'] = rejected[i]
        
        # Add regression line and stats to the existing plot
        if corr['significant']:
            group_data = corr['group_data']
            z = np.polyfit(group_data[corr['metric']], 
                          group_data['abs_normalized_pain_change'], 1)
            p_fit = np.poly1d(z)
            x_range = np.linspace(group_data[corr['metric']].min(), 
                                group_data[corr['metric']].max(), 100)
            corr['ax'].plot(x_range, p_fit(x_range), 
                           color=GROUP_COLORS[corr['group']],
                           linestyle='--', linewidth=2, alpha=0.8)
        
        # Add correlation text
        sig_marker = "***" if corr['p_corrected'] < 0.001 else \
                    "**" if corr['p_corrected'] < 0.01 else \
                    "*" if corr['p_corrected'] < 0.05 else "ns"
        
        text_y_positions = [0.95, 0.85, 0.75]
        corr['ax'].text(0.05, text_y_positions[corr['group_idx']], 
                       f'{corr["group"]}: r={corr["r"]:.2f}, p={corr["p_corrected"]:.3f} {sig_marker}',
                       transform=corr['ax'].transAxes, fontsize=8,
                       color=GROUP_COLORS[corr['group']], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    print(f"Multiple comparisons correction applied to {len(all_correlations)} tests")
    significant_count = sum(corr['significant'] for corr in all_correlations)
    print(f"Significant after FDR correction: {significant_count}/{len(all_correlations)}")

plt.suptitle('Trial Sequence Effects by Clinical Group - Split by Direction (FDR Corrected)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

#%%
# ========================================================
# STATISTICAL ANALYSIS: GROUP x SEQUENCE INTERACTIONS
# Testing whether clinical groups differ in how previous contrast experiences affect current responses
# Based on Kahneman et al. (1993) peak-end rule - using normalized pain change as measure of contrast experience
# ========================================================
print("\n=== TESTING GROUP x SEQUENCE INTERACTIONS ===")
print("Research Question: Do clinical groups show different carryover effects from previous pain experiences?")

# Test for group differences in sequence effects using LME
sequence_results = {}

for trial_type in ['onset', 'offset']:
    print(f"\n{'='*60}")
    print(f"{trial_type.upper()} SEQUENCE EFFECTS ANALYSIS")
    print(f"{'='*60}")
    
    # Filter data - include ALL preceding trial types that have normalized pain change
    analysis_data = contrast_trials[
        (contrast_trials['trial_type'] == trial_type) & 
        (contrast_trials['preceding_abs_normalized_pain_change'].notna()) &
        (contrast_trials['abs_normalized_pain_change'].notna())
    ].copy()
    
    print(f"Analysis dataset:")
    print(f"  Total trials: {len(analysis_data)}")
    print(f"  Subjects: {analysis_data['subject'].nunique()}")
    print(f"  Preceding trial types: {analysis_data['preceding_trial_type'].value_counts().to_dict()}")
    
    print(f"\nSample sizes by group:")
    sample_counts = analysis_data['group_label'].value_counts()
    for group, count in sample_counts.items():
        n_subjects = analysis_data[analysis_data['group_label'] == group]['subject'].nunique()
        print(f"  {group}: {count} trials from {n_subjects} subjects")
    
    try:
        print(f"\nFitting statistical models...")
        
        # Model 1: Main effects only
        print("  Model 1: Main effects only")
        model_main = mixedlm(
            "abs_normalized_pain_change ~ C(group_label) + preceding_abs_normalized_pain_change", 
            data=analysis_data,
            groups=analysis_data["subject"]
        )
        result_main = model_main.fit(reml=False)
        
        # Model 2: With interaction
        print("  Model 2: With group x sequence interaction")
        model_interaction = mixedlm(
            "abs_normalized_pain_change ~ C(group_label) * preceding_abs_normalized_pain_change", 
            data=analysis_data,
            groups=analysis_data["subject"]
        )
        result_interaction = model_interaction.fit(reml=False)
        
        # Likelihood ratio test for interaction
        lr_stat = 2 * (result_interaction.llf - result_main.llf)
        df_diff = len(result_interaction.params) - len(result_main.params)
        p_interaction = 1 - stats.chi2.cdf(lr_stat, df=df_diff)
        
        print(f"\nMODEL COMPARISON:")
        print(f"  Main effects model:")
        print(f"    Log-likelihood: {result_main.llf:.2f}")
        print(f"    AIC: {result_main.aic:.2f}")
        print(f"    Parameters: {len(result_main.params)}")
        
        print(f"  Interaction model:")
        print(f"    Log-likelihood: {result_interaction.llf:.2f}")
        print(f"    AIC: {result_interaction.aic:.2f}")
        print(f"    Parameters: {len(result_interaction.params)}")
        
        print(f"\nINTERACTION TEST:")
        print(f"  Likelihood ratio: χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_interaction:.4f}")
        
        # Determine significance and interpret
        if p_interaction < 0.05:
            sig_marker = "***" if p_interaction < 0.001 else "**" if p_interaction < 0.01 else "*"
            print(f"  Result: SIGNIFICANT GROUP x SEQUENCE INTERACTION {sig_marker}")
            print(f"  Interpretation: Clinical groups differ in how previous pain experiences affect current responses")
            
            print(f"\nINTERACTION MODEL DETAILS:")
            print(result_interaction.summary())
            
        else:
            print(f"  Result: NO SIGNIFICANT INTERACTION (p = {p_interaction:.4f})")
            print(f"  Interpretation: All groups show similar carryover effects from previous trials")
            
            print(f"\nMAIN EFFECTS MODEL DETAILS:")
            print(result_main.summary())
            
            # Interpret main effects
            params = result_main.params
            if 'preceding_abs_normalized_pain_change' in params.index:
                sequence_effect = params['preceding_abs_normalized_pain_change']
                print(f"\nMAIN EFFECTS INTERPRETATION:")
                print(f"  Overall sequence effect: {sequence_effect:.4f}")
                print(f"    → 1% increase in previous pain change → {sequence_effect:.4f}% change in current response")
        
        # Store results
        sequence_results[trial_type] = {
            'main_model': result_main,
            'interaction_model': result_interaction,
            'interaction_p': p_interaction,
            'lr_stat': lr_stat,
            'df_diff': df_diff,
            'significant_interaction': p_interaction < 0.05,
            'data': analysis_data,
            'n_trials': len(analysis_data),
            'n_subjects': analysis_data['subject'].nunique()
        }
        
    except Exception as e:
        print(f"  ERROR: Model fitting failed - {e}")
        import traceback
        traceback.print_exc()


#%%
# ================================================================================================================
# ######################################## 3. HABITUATORS VS SENSITIZERS ANALYSIS ###############################
# ================================================================================================================
analysis_data = unified_data.copy()
# ========================================================
# CALCULATE PAIN TRAJECTORIES FOR EACH SUBJECT
# ========================================================

def calculate_pain_trajectory(subject_data):
    """Calculate pain trajectory using max pain over all trials"""
    # Use all trial types, not just contrast trials
    clean_data = subject_data.dropna(subset=['abs_max_val', 'trial_num']).sort_values('trial_num')
    
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

# Calculate trajectories for each subject
print("Calculating pain trajectories for each subject...")
subject_trajectories = []

for subject, subj_df in unified_data.groupby('subject'):
    slope, r_val, p_val = calculate_pain_trajectory(subj_df)
    
    # Get group label for this subject
    group_label = subj_df['group_label'].iloc[0] if 'group_label' in subj_df.columns else 'unknown'
    dataset = subj_df['dataset'].iloc[0] if 'dataset' in subj_df.columns else 'unknown'
    
    subject_trajectories.append({
        'subject': subject,
        'slope': slope,
        'r_value': r_val,
        'p_value': p_val,
        'group_label': group_label,
        'dataset': dataset,
        'n_trials': len(subj_df)
    })

trajectory_df = pd.DataFrame(subject_trajectories)

# Remove subjects with insufficient data
trajectory_df = trajectory_df.dropna(subset=['slope'])
print(f"Analyzed trajectories for {len(trajectory_df)} subjects")

# Plot distribution of slopes first
plt.figure(figsize=(12, 8))

# Plot by clinical group
for group in ['Control', 'Low', 'High']:
    if group in trajectory_df['group_label'].values:
        group_data = trajectory_df[trajectory_df['group_label'] == group]['slope']
        plt.hist(group_data, bins=15, alpha=0.6, 
                color=GROUP_COLORS[group], 
                label=f'{group} (n={len(group_data)})',
                edgecolor='black', linewidth=0.5)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No change')
plt.xlabel('Max Pain Slope (points per trial)', fontsize=12)
plt.ylabel('Number of Subjects', fontsize=12)
plt.title('Distribution of Individual Pain Trajectories by Clinical Group', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add statistics
stats_text = []
for group in ['Control', 'Low', 'High']:
    if group in trajectory_df['group_label'].values:
        group_data = trajectory_df[trajectory_df['group_label'] == group]['slope']
        mean_slope = group_data.mean()
        std_slope = group_data.std()
        stats_text.append(f'{group}: μ={mean_slope:.2f}, σ={std_slope:.2f}')

plt.text(0.02, 0.98, '\n'.join(stats_text), 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# Cutoff based on SD
mean_slope = trajectory_df['slope'].mean()
std_slope = trajectory_df['slope'].std()
lower_thresh = mean_slope - std_slope
upper_thresh = mean_slope + std_slope

# Classify subjects based on slope magnitude and significance
def classify_subject(row):
    if pd.isna(row['slope']):
        return 'insufficient_data'
    elif row['slope'] < lower_thresh:
        return 'habituator'
    elif row['slope'] > upper_thresh:
        return 'sensitizer'
    else:
        return 'no_trend'

trajectory_df['trajectory_classification'] = trajectory_df.apply(classify_subject, axis=1)

print(f"\nTrajectory Classification Results:")
print(trajectory_df['trajectory_classification'].value_counts())

print(f"\nBy Clinical Group:")
trajectory_summary = trajectory_df.groupby(['group_label', 'trajectory_classification']).size().unstack(fill_value=0)
print(trajectory_summary)

#%%
# ========================================================
# EXAMPLE SUBJECTS FOR EACH TRAJECTORY TYPE
# ========================================================
# Plot example habituator
if len(trajectory_df[trajectory_df['trajectory_classification'] == 'habituator']) > 0:
    example_hab = trajectory_df[trajectory_df['trajectory_classification'] == 'habituator'].iloc[0]
    subj_data = analysis_data[analysis_data['subject'] == example_hab['subject']].sort_values('trial_num')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(subj_data['trial_num'], subj_data['abs_max_val'], 
               c=[GROUP_COLORS.get(subj_data['group_label'].iloc[0], 'gray')], 
               alpha=0.7, s=60, edgecolors='black')
    
    # Add regression line
    slope = example_hab['slope']
    intercept = subj_data['abs_max_val'].mean() - slope * subj_data['trial_num'].mean()
    x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-', linewidth=2)
    
    plt.xlabel('Trial Number')
    plt.ylabel('Maximum Pain Rating')
    plt.ylim(0, 100)
    plt.title(f'Example Habituator: Subject {example_hab["subject"]} ({example_hab["group_label"]})\n'
              f'Slope = {slope:.2f}, r = {example_hab["r_value"]:.2f}, p = {example_hab["p_value"]:.3f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGPATH}/example_habituator.svg")
    plt.show()

# Plot example sensitizer
if len(trajectory_df[trajectory_df['trajectory_classification'] == 'sensitizer']) > 0:
    example_sen = trajectory_df[trajectory_df['trajectory_classification'] == 'sensitizer'].iloc[0]
    subj_data = analysis_data[analysis_data['subject'] == example_sen['subject']].sort_values('trial_num')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(subj_data['trial_num'], subj_data['abs_max_val'],
               c=[GROUP_COLORS.get(subj_data['group_label'].iloc[0], 'gray')], 
               alpha=0.7, s=60, edgecolors='black')
    
    # Add regression line
    slope = example_sen['slope']
    intercept = subj_data['abs_max_val'].mean() - slope * subj_data['trial_num'].mean()
    x_vals = np.array([subj_data['trial_num'].min(), subj_data['trial_num'].max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-', linewidth=2)
    
    plt.xlabel('Trial Number')
    plt.ylabel('Maximum Pain Rating')
    plt.ylim(0, 100)
    plt.title(f'Example Sensitizer: Subject {example_sen["subject"]} ({example_sen["group_label"]})\n'
              f'Slope = {slope:.2f}, r = {example_sen["r_value"]:.2f}, p = {example_sen["p_value"]:.3f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGPATH}/example_sensitizer.svg")
    plt.show()

#%%
# ==========================================================================
# TEMPORAL CONTRAST BY TRAJECTORY - CORRECTED FOR PSEUDOREPLICATION
# ==========================================================================

print("=== CORRECTING FOR PSEUDOREPLICATION ===")
print("Averaging trials within subjects first...")
contrast_analysis = unified_data.merge(trajectory_df[['subject', 'trajectory_classification']], 
                                         on='subject', how='left')
# Calculate subject-level averages for each trial type
subject_averages = []

for subject, subj_data in contrast_analysis.groupby('subject'):
    traj_class = subj_data['trajectory_classification'].iloc[0]
    # Average onset trials for this subject
    onset_trials = subj_data[subj_data['trial_type'] == 'onset']['abs_normalized_pain_change']
    offset_trials = subj_data[subj_data['trial_type'] == 'offset']['abs_normalized_pain_change']
    
    if len(onset_trials) > 0:
        subject_averages.append({
            'subject': subject,
            'trajectory_classification': traj_class,
            'trial_type': 'onset',
            'avg_normalized_pain_change': onset_trials.mean(),
            'n_trials': len(onset_trials)
        })
    
    if len(offset_trials) > 0:
        subject_averages.append({
            'subject': subject,
            'trajectory_classification': traj_class,
            'trial_type': 'offset', 
            'avg_normalized_pain_change': offset_trials.mean(),
            'n_trials': len(offset_trials)
        })

subject_avg_df = pd.DataFrame(subject_averages)

print(f"Subject-level data: {len(subject_avg_df)} subject-trial_type combinations")
print(f"From {subject_avg_df['subject'].nunique()} unique subjects")

# Now plot using subject averages (proper n)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Define colors and order
traj_colors = {'habituator': 'blue', 'no_trend': 'gray', 'sensitizer': 'red'}
traj_order = ['habituator', 'no_trend', 'sensitizer']

def add_significance_brackets(ax, x1, x2, y, h, text, fontsize=12):
    """Add significance brackets between bars"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', 
            fontweight='bold', fontsize=fontsize)

# ONSET TRIALS
onset_subj_data = subject_avg_df[subject_avg_df['trial_type'] == 'onset']
if len(onset_subj_data) > 0:
    sns.violinplot(data=onset_subj_data, x='trajectory_classification', y='avg_normalized_pain_change',
                   palette=traj_colors, inner='box', ax=axes[0], order=traj_order)
    axes[0].set_title('Onset Hyperalgesia by Trajectory Group\n(Subject Averages)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Trajectory Group', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Average Normalized Pain Change (%)', fontweight='bold', fontsize=12)
    
    # Add sample sizes (now subjects, not trials!)
    for i, traj_group in enumerate(traj_order):
        n_subjects = len(onset_subj_data[onset_subj_data['trajectory_classification'] == traj_group])
        if n_subjects > 0:
            axes[0].text(i, axes[0].get_ylim()[0] + 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), 
                        f'n={n_subjects} subjects', ha='center', fontweight='bold', fontsize=11)

# OFFSET TRIALS  
offset_subj_data = subject_avg_df[subject_avg_df['trial_type'] == 'offset']
if len(offset_subj_data) > 0:
    sns.violinplot(data=offset_subj_data, x='trajectory_classification', y='avg_normalized_pain_change',
                   palette=traj_colors, inner='box', ax=axes[1], order=traj_order)
    axes[1].set_title('Offset Analgesia by Trajectory Group\n(Subject Averages)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Trajectory Group', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Average Normalized Pain Change (%)', fontweight='bold', fontsize=12)
    
    # Add sample sizes
    for i, traj_group in enumerate(traj_order):
        n_subjects = len(offset_subj_data[offset_subj_data['trajectory_classification'] == traj_group])
        if n_subjects > 0:
            axes[1].text(i, axes[1].get_ylim()[0] + 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]), 
                        f'n={n_subjects} subjects', ha='center', fontweight='bold', fontsize=11)

# Re-run statistics on subject averages
print("\n=== STATISTICS ON SUBJECT AVERAGES ===")
# Onset comparisons
onset_groups = {}
for traj_group in traj_order:
    group_data = onset_subj_data[onset_subj_data['trajectory_classification'] == traj_group]['avg_normalized_pain_change']
    if len(group_data) > 0:
        onset_groups[traj_group] = group_data
        print(f"Onset {traj_group}: n={len(group_data)} subjects, mean={group_data.mean():.2f}")

# Offset comparisons  
offset_groups = {}
for traj_group in traj_order:
    group_data = offset_subj_data[offset_subj_data['trajectory_classification'] == traj_group]['avg_normalized_pain_change']
    if len(group_data) > 0:
        offset_groups[traj_group] = group_data
        print(f"Offset {traj_group}: n={len(group_data)} subjects, mean={group_data.mean():.2f}")

plt.tight_layout()
plt.show()


#%%
# QUESTION: Is there a significant difference trajectory value across clinical groups?
# Statistical test: Do trajectory slopes differ across clinical groups?
print(f"\n{'='*60}")
print("TRAJECTORY SLOPES BY CLINICAL GROUP")
print(f"{'='*60}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Box plot of slopes by group
sns.boxplot(data=trajectory_df, x='group_label', y='slope', palette=GROUP_COLORS, ax=axes[0])
sns.stripplot(data=trajectory_df, x='group_label', y='slope', color='black', alpha=0.4, size=6, ax=axes[0])
axes[0].set_title('Distribution of Trajectory Slopes by Clinical Group', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Clinical Group', fontweight='bold')
axes[0].set_ylabel('Slope (pain points per trial)', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Violin plot for better distribution visualization
sns.violinplot(data=trajectory_df, x='group_label', y='slope', palette=GROUP_COLORS, ax=axes[1])
axes[1].set_title('Density Distribution of Slopes by Clinical Group', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Clinical Group', fontweight='bold')
axes[1].set_ylabel('Slope (pain points per trial)', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{FIGPATH}/trajectory_slopes_by_group.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# # Statistical analysis: ANOVA comparing slopes across groups
# print("\nStatistical Comparison of Slopes Across Groups:")
# groups_list = []
# for group in ['Control', 'Low', 'High']:
#     group_data = trajectory_df[trajectory_df['group_label'] == group]['slope']
#     if len(group_data) > 0:
#         groups_list.append(group_data.values)
#         print(f"\n{group}:")
#         print(f"  n = {len(group_data)}")
#         print(f"  Mean slope = {group_data.mean():.3f}")
#         print(f"  Std = {group_data.std():.3f}")
#         print(f"  Range = [{group_data.min():.3f}, {group_data.max():.3f}]")

# # One-way ANOVA
# if len(groups_list) >= 2:
#     f_stat, p_anova = stats.f_oneway(*groups_list)
#     sig = "***" if p_anova < 0.001 else "**" if p_anova < 0.01 else "*" if p_anova < 0.05 else "ns"
#     print(f"\nOne-way ANOVA: F = {f_stat:.3f}, p = {p_anova:.4f} {sig}")
    
#     if p_anova < 0.05:
#         print("Result: Significant difference in trajectory slopes across clinical groups")
#     else:
#         print("Result: No significant difference in trajectory slopes across clinical groups")

# %% 
# Break down trial_sequence effects by trajectory classification - 2x2 OVERLAY PLOTS
print(f"\n{'='*60}")
print("TRIAL SEQUENCE EFFECTS BY TRAJECTORY CLASSIFICATION - 2x2 OVERLAY PLOTS")
print(f"{'='*60}")
# Prepare plot_data
plot_data = unified_data[unified_data['trial_type'].isin(['onset', 'offset'])].copy()
plot_data = plot_data.merge(trajectory_df[['subject', 'trajectory_classification']],
                            on='subject', how='left')
plot_data = plot_data[plot_data['trajectory_classification'].isin(['habituator', 'sensitizer', 'no_trend'])].copy()

metric = 'preceding_abs_normalized_pain_change'
metric_label = 'Preceding Normalized Pain Change'
traj_groups = ['habituator', 'no_trend', 'sensitizer']
traj_colors = {'habituator': 'blue', 'sensitizer': 'red', 'no_trend': 'grey'}
trial_types = ['onset', 'offset']
trial_labels = {'onset': 'Onset Hyperalgesia', 'offset': 'Offset Analgesia'}

# Create 2x2 plot: rows = [positive, negative], cols = [onset, offset]
fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharey=False)
fig.suptitle('Trial Sequence Effects by Trajectory Group\n(Split by Preceding Pain Change Direction)',
             fontsize=16, fontweight='bold', y=0.98)

for col_idx, trial_type in enumerate(trial_types):
    # Filter by trial type first
    trial_data = plot_data[plot_data['trial_type'] == trial_type].copy()
    
    for row_idx, (sign_label, sign_filter) in enumerate([('Positive', lambda x: x > 0), 
                                                         ('Negative', lambda x: x < 0)]):
        ax = axes[row_idx, col_idx]
        
        # Filter by sign of preceding pain change
        subset = trial_data[
            trial_data[metric].notnull() & 
            trial_data['abs_normalized_pain_change'].notnull() &
            trial_data[metric].apply(sign_filter)
        ].copy()
        
        if len(subset) > 0:
            # Plot each trajectory group
            for traj_group in traj_groups:
                group_data = subset[subset['trajectory_classification'] == traj_group]
                
                if len(group_data) > 0:
                    # Scatter plot for this group
                    ax.scatter(group_data[metric], 
                              group_data['abs_normalized_pain_change'],
                              color=traj_colors[traj_group], 
                              alpha=0.6, 
                              label=f'{traj_group.replace("_", " ").title()} (n={len(group_data)})',
                              s=50, edgecolors='black', linewidth=0.5)
                    
                    # Add regression line if enough points and significant
                    if len(group_data) > 5:
                        x = group_data[metric]
                        y = group_data['abs_normalized_pain_change']
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        # Only plot line if significant
                        if p_value < 0.05:
                            x_vals = np.array([x.min(), x.max()])
                            y_vals = intercept + slope * x_vals
                            ax.plot(x_vals, y_vals, color=traj_colors[traj_group],
                                   linestyle='--', linewidth=2, alpha=0.8)
                        
                        # Add correlation info in corner - position by group
                        text_y_pos = 0.95 - (traj_groups.index(traj_group) * 0.08)
                        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        
                        ax.text(0.05, text_y_pos,
                               f'{traj_group.replace("_", " ").title()}: r={r_value:.2f}, p={p_value:.3f} {sig_marker}',
                               transform=ax.transAxes,
                               fontsize=9,
                               color=traj_colors[traj_group],
                               fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add OVERALL trend line (all trajectory groups combined)
            if len(subset) > 10:
                x_all = subset[metric]
                y_all = subset['abs_normalized_pain_change']
                r_all, p_all = stats.pearsonr(x_all, y_all)
                
                # Plot overall regression line if significant
                if p_all < 0.05:
                    slope_all, intercept_all, _, _, _ = stats.linregress(x_all, y_all)
                    x_range = np.linspace(x_all.min(), x_all.max(), 100)
                    y_range = intercept_all + slope_all * x_range
                    ax.plot(x_range, y_range, color='black', linestyle='-', linewidth=3, alpha=0.8)
                
                sig_all = "***" if p_all < 0.001 else "**" if p_all < 0.01 else "*" if p_all < 0.05 else "ns"
                
                # Add overall statistics
                ax.text(0.65, 0.95,
                        f"Overall: r = {r_all:.2f}, p = {p_all:.3f} {sig_all}\n"
                        f"n = {len(subset)} trials\n"
                        f"n = {subset['subject'].nunique()} subjects",
                        transform=ax.transAxes,
                        verticalalignment='top',
                        fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8, edgecolor='black'))
        
        # Set y-axis limits based on trial type
        if trial_type == 'onset':  # Hyperalgesia trials
            ax.set_ylim(0, 100)
        else:  # Offset/Analgesia trials
            ax.set_ylim(0, -100)
        
        # Set x-axis limits based on direction
        if sign_label == 'Positive':
            ax.set_xlim(0, 100)
        else:  # Negative
            ax.set_xlim(0, -100)
        
        # Labels and title
        if row_idx == 1:  # Only add x-label to bottom row
            ax.set_xlabel(metric_label, fontsize=11, fontweight='bold')
        
        if col_idx == 0:  # Only add y-label to leftmost column
            ax.set_ylabel(f'{trial_labels[trial_type]} Magnitude (%)',
                          fontsize=11, fontweight='bold')

        title = f"{sign_label} Preceding Change\n{trial_labels[trial_type]}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add legend only to top right plot
        if row_idx == 0 and col_idx == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(f'{FIGPATH}/sequence_effects_by_trajectory_2x2_overlay.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# Print summary statistics
print(f"\n=== SUMMARY STATISTICS BY DIRECTION AND TRIAL TYPE ===")
for ttype in trial_types:
    print(f"\n{ttype.upper()} TRIALS:")
    trial_data = plot_data[plot_data['trial_type'] == ttype].copy()
    
    for sign_label, sign_filter in [('Positive', lambda x: x > 0), ('Negative', lambda x: x < 0)]:
        subset = trial_data[
            trial_data[metric].notnull() & 
            trial_data['abs_normalized_pain_change'].notnull() &
            trial_data[metric].apply(sign_filter)
        ]
        
        print(f"  {sign_label} preceding change:")
        print(f"    Total trials: {len(subset)}")
        print(f"    Total subjects: {subset['subject'].nunique()}")
        
        for group in ['control', 'low_pain', 'high_pain']:
            group_data = subset[subset['group_label'] == group]
            if len(group_data) > 0:
                print(f"    {group}: {len(group_data)} trials from {group_data['subject'].nunique()} subjects")

# %%
# Chi-squared analysis: trajectory classification vs clinical group
print("\n=== CHI-SQUARED ANALYSIS: TRAJECTORY CLASSIFICATION vs CLINICAL GROUP ===")


# Create contingency table
trajectory_df['group_label'] = pd.Categorical(
    trajectory_df['group_label'],
    categories=['Control', 'Low', 'High'],
    ordered=True
)

trajectory_df['trajectory_classification'] = pd.Categorical(
    trajectory_df['trajectory_classification'],
    categories=['sensitizer', 'no_trend', 'habituator'],
    ordered=True
)

contingency_table = pd.crosstab(
    trajectory_df['trajectory_classification'],
    trajectory_df['group_label']
)

print("Contingency Table:")
print(contingency_table)

# Run chi-squared test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-squared test results:")
print(f"  chi2 = {chi2:.2f}")
print(f"  p-value = {p:.4f}")
print(f"  degrees of freedom = {dof}")
print("Expected counts:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

if p < 0.05:
    print("Result: Significant association between trajectory classification and clinical group")
else:
    print("Result: No significant association between trajectory classification and clinical group")

# Plot it out too
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Contingency Table: Trajectory Classification vs Clinical Group', fontweight='bold')
plt.xlabel('Clinical Group', fontweight='bold')
plt.ylabel('Trajectory Classification', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/contingency_table_trajectory_vs_group.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# %%
full_patient_info_path = '/Users/ljohnston1/UCSF DBS for Pain Dropbox/PainNeuromodulationLab/DATA ANALYSIS/Lucy/BenAlter_Collab_Data/KneeNIRS data/kneeNIRS dataset from MMtrimmedcsv250107.xlsx'
full_patient_info = pd.read_excel(full_patient_info_path)

columns_of_interest = ['PCS_rumination','PCS_magnification','SPCS_QST','pain_now','HPT_forearm_avg']




# %%
# FIGURES FOR THE POSTER
analyses = [
    ('onset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('onset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
]

print("Creating plots and collecting correlations...")
all_correlations = []
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

for idx, (trial_type, preceding_metric, direction, metric_label) in enumerate(analyses):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Y-axis: always 0 to 100 for onset, -100 to 0 for offset
    if trial_type == 'onset':
        ax.set_ylim(0, 101)
    else:
        ax.set_ylim(0, -101)
    
    # Filter data for this analysis with direction
    base_data = contrast_trials[
        (contrast_trials['trial_type'] == trial_type) & 
        (contrast_trials[preceding_metric].notna()) &
        (contrast_trials['abs_normalized_pain_change'].notna())
    ]
    
    # Apply direction filter
    if direction == 'positive':
        plot_data = base_data[base_data[preceding_metric] > 0]
        ax.set_xlim(0, 101)  # Focus on positive range
    elif direction == 'negative':
        plot_data = base_data[base_data[preceding_metric] < 0]
        ax.set_xlim(0, -101)  # Focus on negative range
    else:
        plot_data = base_data
    
    # Add total sample size in title
    total_n = len(plot_data)
    ax.set_title(f'{trial_type.title()} - {metric_label}\n(N={total_n})', fontweight='bold', fontsize=10)
    
    if len(plot_data) > 10:
        # Create scatter plot by group (for visualization)
        for group in ['Control', 'Low', 'High']:
            group_data = plot_data[plot_data['group_label'] == group]
            if len(group_data) > 0:
                ax.scatter(group_data[preceding_metric], 
                          group_data['abs_normalized_pain_change'],
                          color=GROUP_COLORS[group], 
                          alpha=0.6, 
                          label=f'{group} (n={len(group_data)})',
                          s=50, edgecolors='black', linewidth=0.5)
        
        # CALCULATE correlation for ENTIRE dataset (all groups combined)
        r, p = stats.pearsonr(plot_data[preceding_metric], 
                             plot_data['abs_normalized_pain_change'])
        
        # STORE correlation for FDR correction later
        all_correlations.append({
            'idx': idx,
            'trial_type': trial_type,
            'metric': preceding_metric,
            'direction': direction,
            'metric_label': metric_label,
            'r': r,
            'p_raw': p,
            'n': len(plot_data),
            'ax': ax,  # Store axis reference
            'plot_data': plot_data  # Store data for regression line
        })
    
    # Formatting
    ax.set_xlabel('Preceding Trial Normalized Pain Change (%)')
    ax.set_ylabel('Current Normalized Pain Change (%)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

# NOW apply FDR correction and add regression lines/stats to existing plots
if all_correlations:
    p_values = [corr['p_raw'] for corr in all_correlations]
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
    
    for i, corr in enumerate(all_correlations):
        corr['p_corrected'] = p_corrected[i]
        corr['significant'] = rejected[i]
        
        # Add regression line for the ENTIRE dataset if significant
        if corr['significant']:
            plot_data = corr['plot_data']
            z = np.polyfit(plot_data[corr['metric']], 
                          plot_data['abs_normalized_pain_change'], 1)
            p_fit = np.poly1d(z)
            x_range = np.linspace(plot_data[corr['metric']].min(), 
                                plot_data[corr['metric']].max(), 100)
            corr['ax'].plot(x_range, p_fit(x_range), 
                           color='black',  # Use black for overall correlation
                           linestyle='-', linewidth=3, alpha=0.8,
                           label=f'Overall trend')
        
        # Add correlation text for overall correlation
        sig_marker = "***" if corr['p_corrected'] < 0.001 else \
                    "**" if corr['p_corrected'] < 0.01 else \
                    "*" if corr['p_corrected'] < 0.05 else "ns"
        
        corr['ax'].text(0.05, 0.95, 
                       f'Overall: r={corr["r"]:.3f}, p={corr["p_corrected"]:.3f} {sig_marker}',
                       transform=corr['ax'].transAxes, fontsize=10,
                       color='black', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    print(f"Multiple comparisons correction applied to {len(all_correlations)} plots")
    significant_count = sum(corr['significant'] for corr in all_correlations)
    print(f"Significant after FDR correction: {significant_count}/{len(all_correlations)}")

plt.suptitle('Trial Sequence Effects - Overall Correlations by Direction (FDR Corrected)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/OH_OA_NormPainChange_PrecedingNormPainChange_PosNeg_Full.svg")
plt.show()

# Print summary of results
print("\nSummary of correlations:")
for corr in all_correlations:
    print(f"{corr['trial_type'].title()} - {corr['metric_label']}: "
          f"r={corr['r']:.3f}, p_raw={corr['p_raw']:.3f}, "
          f"p_corrected={corr['p_corrected']:.3f}, "
          f"significant={'Yes' if corr['significant'] else 'No'}")


