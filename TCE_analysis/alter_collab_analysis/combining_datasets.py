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
    'group_label': 'control',
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
    'control': '#2E8B57',    # Green
    'low_pain': '#FF8C00',   # Orange  
    'high_pain': '#DC143C'   # Red
}

############################################################################## Raw distributions and group comparisons
# Plot the raw data to get a sense of the distributions and group differences
# Create comprehensive comparison plots with violin plots and sample sizes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
main_analysis_data = unified_data[unified_data['trial_type'].isin(common_stepped_trials)].copy()
# Prepare data for plotting
plot_data = main_analysis_data[main_analysis_data['trial_type'].isin(['onset', 'offset'])]

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

############################################################################## LME analysis with cleaned data and diagnostics
# Check for data issues
print("Checking for data issues...")
for trial_type in ['onset', 'offset']:
    trial_data = main_analysis_data[main_analysis_data['trial_type'] == trial_type].copy()
    print(f"\n{trial_type.upper()} data check:")
    print(f"  Total rows: {len(trial_data)}")
    print(f"  Missing abs_normalized_pain_change: {trial_data['abs_normalized_pain_change'].isna().sum()}")
    print(f"  Infinite values: {np.isinf(trial_data['abs_normalized_pain_change']).sum()}")
    # Check for extreme outliers that might cause issues
    if not trial_data['abs_normalized_pain_change'].empty:
        q99 = trial_data['abs_normalized_pain_change'].quantile(0.99)
        q01 = trial_data['abs_normalized_pain_change'].quantile(0.01)
        print(f"  Range: {q01:.1f} to {q99:.1f}")
# Clean the data before analysis
def clean_data_for_analysis(data):
    """Clean data by removing infinite values and extreme outliers"""
    # Remove infinite values
    data = data[np.isfinite(data['abs_normalized_pain_change'])]
    # Remove extreme outliers (beyond 4 standard deviations)
    mean_val = data['abs_normalized_pain_change'].mean()
    std_val = data['abs_normalized_pain_change'].std()
    data = data[np.abs(data['abs_normalized_pain_change'] - mean_val) <= 4 * std_val]
    return data

# Perform LME analysis for abs_normalized_pain_change and trial type
lme_results = {}
for trial_type in ['onset', 'offset']:
    trial_data = main_analysis_data[main_analysis_data['trial_type'] == trial_type].copy()
    lme_results[trial_type] = {}
    # Clean data
    clean_data = trial_data.dropna(subset=['abs_normalized_pain_change', 'group_label', 'subject'])
    clean_data = clean_data_for_analysis(clean_data)   
    try:
        # Mixed effects model: abs_normalized_pain_change ~ group_label + (1|subject)
        model = mixedlm(f"abs_normalized_pain_change ~ C(group_label)", 
                        data=clean_data,
                        groups=clean_data["subject"])
        result = model.fit(reml=False)
        # Extract all parameters and standard errors
        params = result.params
        std_errors = result.bse
        pvalues = result.pvalues
        conf_int = result.conf_int()
        print(result.summary())
        # Store control group results 
        baseline_mean = params['Intercept']
        baseline_se = std_errors['Intercept']  
        control_n = len(clean_data[clean_data['group_label'] == 'control'])
        lme_results[trial_type]['control'] = {
            'mean': baseline_mean,
            'se': baseline_se,
            'n': control_n,
            'ci_lower': conf_int.loc['Intercept', 0],
            'ci_upper': conf_int.loc['Intercept', 1]
        }
        for param in params.index:
            if 'group_label' in param and '[T.' in param: 
                coef = params[param]
                se = std_errors[param]
                p_val = pvalues[param]
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                # Interpret the coefficient
                # Extract group name and calculate actual mean
                group_name = param.split('[T.')[-1].replace(']', '')
                actual_mean = baseline_mean + coef
                group_n = len(clean_data[clean_data['group_label'] == group_name])
                
                # Store results
                lme_results[trial_type][group_name] = {
                    'mean': actual_mean,
                    'se': se,  # Fixed: now using the correct se variable
                    'n': group_n,
                    'p_vs_control': p_val,
                    'coef': coef,
                    'ci_lower': conf_int.loc[param, 0],
                    'ci_upper': conf_int.loc[param, 1]
                }
                # Print results
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                direction = "higher" if coef > 0 else "lower"
                print(f"  {group_name}: {direction} by {abs(coef):.2f} ± {se:.2f}, p = {p_val:.4f} {sig}")
                print(f"    Estimated mean: {actual_mean:.2f}")
        # Overall F-test for group effect
        null_model = mixedlm(f"abs_normalized_pain_change ~ 1", 
                            data=clean_data,
                            groups=clean_data["subject"])
        null_result = null_model.fit(reml=False)
        # Likelihood ratio test
        lr_stat = 2 * (result.llf - null_result.llf)
        df = len(params) - len(null_result.params)  # Fixed: calculate df properly
        p_lr = 1 - scipy_stats.chi2.cdf(lr_stat, df=df)  # Fixed: use calculated df
        overall_sig = "***" if p_lr < 0.001 else "**" if p_lr < 0.01 else "*" if p_lr < 0.05 else "ns"
        
        print(f"  Likelihood Ratio Test: χ² = {lr_stat:.2f}, df = {df}, p = {p_lr:.4f} {overall_sig}")
        
        # Store overall test results
        lme_results[trial_type]['overall_test'] = {
            'lr_stat': lr_stat,
            'df': df,
            'p_value': p_lr
        }
        
    except Exception as e:
        print(f"  Model failed: {e}")
        import traceback
        traceback.print_exc()

############################################################################## Plot results from LME analysis
# Get values and errors for onset
# Extract data for plotting from lme_results
groups = ['control', 'low_pain', 'high_pain']
group_labels = ['Control', 'Low Pain', 'High Pain']
onset_means = [lme_results['onset'][group]['mean'] for group in groups]
onset_ses = [lme_results['onset'][group]['se'] for group in groups]
onset_ns = [lme_results['onset'][group]['n'] for group in groups]

# Get values and errors for offset  
offset_means = [lme_results['offset'][group]['mean'] for group in groups]
offset_ses = [lme_results['offset'][group]['se'] for group in groups]
offset_ns = [lme_results['offset'][group]['n'] for group in groups]

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

# ONSET HYPERALGESIA (Left panel)
x_pos = np.arange(len(groups))
bars1 = ax1.bar(x_pos, onset_means, 
               color=[GROUP_COLORS[group] for group in groups],
               alpha=0.8, capsize=5, edgecolor='black', linewidth=1)

# Add error bars
ax1.errorbar(x_pos, onset_means, yerr=onset_ses, fmt='none', 
             color='black', capsize=5, linewidth=1.5)

# Add significance brackets and stars
def add_significance_bracket(ax, x1, x2, y, h, text):
    """Add significance bracket between two bars"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', 
            fontweight='bold', fontsize=14)

# Add significance based on your LME results
y_max = max(onset_means) + max(onset_ses)
if lme_results['onset']['low_pain']['p_vs_control'] < 0.05:
    sig_star = '***' if lme_results['onset']['low_pain']['p_vs_control'] < 0.001 else '**' if lme_results['onset']['low_pain']['p_vs_control'] < 0.01 else '*'
    add_significance_bracket(ax1, 0, 1, y_max + 3, 2, sig_star)

if lme_results['onset']['high_pain']['p_vs_control'] < 0.05:
    sig_star = '***' if lme_results['onset']['high_pain']['p_vs_control'] < 0.001 else '**' if lme_results['onset']['high_pain']['p_vs_control'] < 0.01 else '*'
    add_significance_bracket(ax1, 0, 2, y_max + 8, 2, sig_star)

# Formatting for onset plot
ax1.set_ylabel('Normalized Pain Change (%)', fontweight='bold')
ax1.set_title('Onset Hyperalgesia', fontweight='bold', pad=20)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{label}\n(n={n})' for label, n in zip(group_labels, onset_ns)])
ax1.set_ylim(0, max(onset_means) + max(onset_ses) + 15)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars1, onset_means)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3, 
             f'{value:.1f}%', ha='center', va='top', fontweight='bold', 
             color='white', fontsize=11)

# OFFSET ANALGESIA (Right panel)
bars2 = ax2.bar(x_pos, offset_means,
               color=[GROUP_COLORS[group] for group in groups],
               alpha=0.8, capsize=5, edgecolor='black', linewidth=1)

# Add error bars
ax2.errorbar(x_pos, offset_means, yerr=offset_ses, fmt='none',
             color='black', capsize=5, linewidth=1.5)

# Add significance brackets for offset
y_min = min(offset_means) - max(offset_ses)
if lme_results['offset']['high_pain']['p_vs_control'] < 0.05:
    sig_star = '***' if lme_results['offset']['high_pain']['p_vs_control'] < 0.001 else '**' if lme_results['offset']['high_pain']['p_vs_control'] < 0.01 else '*'
    add_significance_bracket(ax2, 0, 2, y_min - 3, -2, sig_star)

if lme_results['offset']['low_pain']['p_vs_control'] < 0.05:
    sig_star = '***' if lme_results['offset']['low_pain']['p_vs_control'] < 0.001 else '**' if lme_results['offset']['low_pain']['p_vs_control'] < 0.01 else '*'
    add_significance_bracket(ax2, 0, 1, y_min - 8, -2, sig_star)

# Formatting for offset plot
ax2.set_ylabel('Normalized Pain Change (%)', fontweight='bold')
ax2.set_title('Offset Analgesia', fontweight='bold', pad=20)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{label}\n(n={n})' for label, n in zip(group_labels, offset_ns)])
ax2.set_ylim(min(offset_means) - max(offset_ses) - 15, -50)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars2, offset_means)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', 
             color='black', fontsize=11)

# Overall title
fig.suptitle('Temporal Contrast Effects by Clinical Group', 
             fontsize=16, fontweight='bold', y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.15)
# Save the plot
plt.savefig(f'{FIGPATH}/temporal_contrast_LME_results.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

#%%
# ==================================================================================================================
######################################## 2. TRIAL SEQUENCES ########################################
# ==================================================================================================================
from statsmodels.stats.multitest import multipletests

sequence_data = unified_data.copy()

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
    sequence_data[new_col] = sequence_data.apply(
        lambda row: get_preceding_value(row, source_col, sequence_data), axis=1
    )

# Filter for onset/offset trials (your main contrast trials)
contrast_trials = sequence_data[sequence_data['trial_type'].isin(['onset', 'offset'])].copy()

# Analysis combinations
analyses = [
    ('onset', 'preceding_auc_total', 'AUC Total'),
    ('onset', 'preceding_abs_max_val', 'Max Pain'),
    ('onset', 'preceding_abs_normalized_pain_change', 'Normalized Change'),
    ('offset', 'preceding_auc_total', 'AUC Total'),
    ('offset', 'preceding_abs_max_val', 'Max Pain'),
    ('offset', 'preceding_abs_normalized_pain_change', 'Normalized Change')
]

# First pass: collect all correlations for multiple comparisons correction
print("Collecting correlations for multiple comparisons correction...")
all_correlations = []

for idx, (trial_type, preceding_metric, metric_label) in enumerate(analyses):
    # Filter data for this analysis
    plot_data = contrast_trials[
        (contrast_trials['trial_type'] == trial_type) & 
        (contrast_trials[preceding_metric].notna()) &
        (contrast_trials['abs_normalized_pain_change'].notna())
    ]
    
    if len(plot_data) > 10:
        for group in ['control', 'low_pain', 'high_pain']:
            group_data = plot_data[plot_data['group_label'] == group]
            if len(group_data) > 5:
                r, p = stats.pearsonr(group_data[preceding_metric], 
                                     group_data['abs_normalized_pain_change'])
                all_correlations.append({
                    'idx': idx,
                    'trial_type': trial_type,
                    'metric': preceding_metric,
                    'metric_label': metric_label,
                    'group': group,
                    'r': r,
                    'p_raw': p,
                    'n': len(group_data),
                    'data': group_data
                })

# Apply FDR correction
if all_correlations:
    p_values = [corr['p_raw'] for corr in all_correlations]
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
    
    for i, corr in enumerate(all_correlations):
        corr['p_corrected'] = p_corrected[i]
        corr['significant'] = rejected[i]
    
    print(f"Multiple comparisons correction applied to {len(all_correlations)} tests")
    significant_count = sum(corr['significant'] for corr in all_correlations)
    print(f"Significant after FDR correction: {significant_count}/{len(all_correlations)}")

# Create correlation lookup for plotting
correlation_lookup = {}
for corr in all_correlations:
    key = (corr['idx'], corr['group'])
    correlation_lookup[key] = corr

# Now create the plot with corrected significance
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (trial_type, preceding_metric, metric_label) in enumerate(analyses):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Filter data for this analysis
    plot_data = contrast_trials[
        (contrast_trials['trial_type'] == trial_type) & 
        (contrast_trials[preceding_metric].notna()) &
        (contrast_trials['abs_normalized_pain_change'].notna())
    ]
    
    # Add total sample size in title
    total_n = len(plot_data)
    ax.set_title(f'{trial_type.title()} Trials (N={total_n})', fontweight='bold')
    
    if len(plot_data) > 10:
        text_y_positions = [0.95, 0.85, 0.75]  # Y positions for text annotations
        
        # Create scatter plot by group
        for group_idx, group in enumerate(['control', 'low_pain', 'high_pain']):
            group_data = plot_data[plot_data['group_label'] == group]
            if len(group_data) > 3:
                ax.scatter(group_data[preceding_metric], 
                          group_data['abs_normalized_pain_change'],
                          color=GROUP_COLORS[group], 
                          alpha=0.6, 
                          label=f'{group}',
                          s=50, edgecolors='black', linewidth=0.5)
                
                # Check if correlation is significant after correction
                key = (idx, group)
                if key in correlation_lookup:
                    corr_info = correlation_lookup[key]
                    
                    # Add regression line only if significant after correction
                    if corr_info['significant']:
                        z = np.polyfit(group_data[preceding_metric], 
                                     group_data['abs_normalized_pain_change'], 1)
                        p_fit = np.poly1d(z)
                        x_range = np.linspace(group_data[preceding_metric].min(), 
                                            group_data[preceding_metric].max(), 100)
                        ax.plot(x_range, p_fit(x_range), 
                               color=GROUP_COLORS[group],
                               linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Add correlation info (show both raw and corrected p-values)
                    sig_marker = "***" if corr_info['p_corrected'] < 0.001 else \
                                "**" if corr_info['p_corrected'] < 0.01 else \
                                "*" if corr_info['p_corrected'] < 0.05 else "ns"
                    
                    ax.text(0.05, text_y_positions[group_idx], 
                           f'{group}: r={corr_info["r"]:.2f}, p={corr_info["p_corrected"]:.3f} {sig_marker}',
                           transform=ax.transAxes, fontsize=8,
                           color=GROUP_COLORS[group], fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Formatting
    ax.set_xlabel(f'Preceding {metric_label}')
    ax.set_ylabel('Current Normalized Pain Change (%)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Trial Sequence Effects by Clinical Group (FDR Corrected)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print summary table of results
if all_correlations:
    print(f"\n=== SEQUENCE EFFECTS CORRELATION SUMMARY ===")
    print(f"{'Trial':<8} {'Metric':<15} {'Group':<12} {'r':<7} {'p_raw':<7} {'p_FDR':<7} {'n':<4} {'Sig'}")
    print("-" * 70)
    
    for corr in all_correlations:
        sig = "***" if corr['p_corrected'] < 0.001 else "**" if corr['p_corrected'] < 0.01 else "*" if corr['p_corrected'] < 0.05 else "ns"
        metric_short = corr['metric'].replace('preceding_abs_', '').replace('preceding_', '')[:12]
        
        print(f"{corr['trial_type']:<8} {metric_short:<15} {corr['group']:<12} "
              f"{corr['r']:<7.3f} {corr['p_raw']:<7.4f} {corr['p_corrected']:<7.4f} "
              f"{corr['n']:<4} {sig}")
    
    print(f"\nSignificant correlations after FDR correction: {significant_count}/{len(all_correlations)}")
# BUT THIS IS VERY SUS LET ME DO THIS MORE RIGOROUSLY IN THE NEXT CELL



















"""
EVERYTHING BELOW HERE WAS STRAIGHT FROM CLAUDE AND WAS UNCHECKED MY HUMANS
"""
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
    
    if len(analysis_data) < 20:
        print(f"Insufficient data for {trial_type} (n={len(analysis_data)})")
        continue
    
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

# Use your existing unified_data
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

for subject, subj_df in analysis_data.groupby('subject'):
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
for group in ['control', 'low_pain', 'high_pain']:
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
for group in ['control', 'low_pain', 'high_pain']:
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

# Classify subjects based on slope magnitude and significance
def classify_subject(row):
    """Classify based on slope and statistical significance"""
    if pd.isna(row['slope']) or pd.isna(row['p_value']):
        return 'insufficient_data'
    elif row['slope'] < -1 and row['p_value'] < 0.05:  # Significant decrease
        return 'habituator'
    elif row['slope'] > 1 and row['p_value'] < 0.05:   # Significant increase
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
    plt.show()

#%%
# ========================================================
# TEMPORAL CONTRAST RESPONSES BY TRAJECTORY GROUP
# ========================================================

print(f"\n{'='*60}")
print("TEMPORAL CONTRAST RESPONSES BY TRAJECTORY GROUP")
print(f"{'='*60}")

# Merge trajectory classifications with trial data
contrast_with_trajectories = unified_data.merge(
    trajectory_df[['subject', 'trajectory_classification']], 
    on='subject', 
    how='left'
)

# Filter for contrast trials and subjects with clear trajectories
contrast_analysis = contrast_with_trajectories[
    (contrast_with_trajectories['trial_type'].isin(['onset', 'offset'])) &
    (contrast_with_trajectories['trajectory_classification'].isin(['habituator', 'sensitizer']))
].copy()

print(f"Analysis data: {len(contrast_analysis)} trials from {contrast_analysis['subject'].nunique()} subjects")
print(f"Trajectory distribution:")
print(contrast_analysis['trajectory_classification'].value_counts())

# Compare onset hyperalgesia between habituators and sensitizers
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ONSET TRIALS
onset_data = contrast_analysis[contrast_analysis['trial_type'] == 'onset']
if len(onset_data) > 0:
    sns.violinplot(data=onset_data, x='trajectory_classification', y='abs_normalized_pain_change',
                   palette={'habituator': 'blue', 'sensitizer': 'red'}, 
                   inner='box', ax=axes[0])
    axes[0].set_title('Onset Hyperalgesia by Trajectory Group')
    axes[0].set_xlabel('Trajectory Group')
    axes[0].set_ylabel('Normalized Pain Change (%)')
    
    # Add sample sizes
    for i, traj_group in enumerate(['habituator', 'sensitizer']):
        n = len(onset_data[onset_data['trajectory_classification'] == traj_group])
        axes[0].text(i, axes[0].get_ylim()[0] + 0.05 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), 
                    f'n={n}', ha='center', fontweight='bold')
    
    # Statistical test
    hab_onset = onset_data[onset_data['trajectory_classification'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sen_onset = onset_data[onset_data['trajectory_classification'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(hab_onset) > 0 and len(sen_onset) > 0:
        t_stat, p_val = stats.ttest_ind(hab_onset, sen_onset)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"ONSET: Habituators mean={hab_onset.mean():.2f}, Sensitizers mean={sen_onset.mean():.2f}, p={p_val:.4f} {sig}")
        
        # Add significance to plot
        axes[0].text(0.5, 0.95, f'p = {p_val:.4f} {sig}', 
                    transform=axes[0].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# OFFSET TRIALS
offset_data = contrast_analysis[contrast_analysis['trial_type'] == 'offset']
if len(offset_data) > 0:
    sns.violinplot(data=offset_data, x='trajectory_classification', y='abs_normalized_pain_change',
                   palette={'habituator': 'blue', 'sensitizer': 'red'}, 
                   inner='box', ax=axes[1])
    axes[1].set_title('Offset Analgesia by Trajectory Group')
    axes[1].set_xlabel('Trajectory Group')
    axes[1].set_ylabel('Normalized Pain Change (%)')
    
    # Add sample sizes
    for i, traj_group in enumerate(['habituator', 'sensitizer']):
        n = len(offset_data[offset_data['trajectory_classification'] == traj_group])
        axes[1].text(i, axes[1].get_ylim()[0] + 0.05 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]), 
                    f'n={n}', ha='center', fontweight='bold')
    
    # Statistical test
    hab_offset = offset_data[offset_data['trajectory_classification'] == 'habituator']['abs_normalized_pain_change'].dropna()
    sen_offset = offset_data[offset_data['trajectory_classification'] == 'sensitizer']['abs_normalized_pain_change'].dropna()
    
    if len(hab_offset) > 0 and len(sen_offset) > 0:
        t_stat, p_val = stats.ttest_ind(hab_offset, sen_offset)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"OFFSET: Habituators mean={hab_offset.mean():.2f}, Sensitizers mean={sen_offset.mean():.2f}, p={p_val:.4f} {sig}")
        
        # Add significance to plot
        axes[1].text(0.5, 0.95, f'p = {p_val:.4f} {sig}', 
                    transform=axes[1].transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()
