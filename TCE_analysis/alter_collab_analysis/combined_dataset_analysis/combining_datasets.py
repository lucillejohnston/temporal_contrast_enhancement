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

datasets = ['kneeOA', 'plosONE', 'cLBP'] # 'sEEG' 
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
CLBP_SUBJECT_OFFSET = 2000   # Add 2000 to cLBP subjects to avoid overlap

# Update subject IDs in trial metrics
kneeoa_mask_metrics = all_trial_metrics['dataset'] == 'kneeOA'
all_trial_metrics.loc[kneeoa_mask_metrics, 'subject'] += KNEEOA_SUBJECT_OFFSET
clbp_mask_metrics = all_trial_metrics['dataset'] == 'cLBP'
all_trial_metrics.loc[clbp_mask_metrics, 'subject'] += CLBP_SUBJECT_OFFSET
# Update subject IDs in trial data  
kneeoa_mask_data = all_trial_data['dataset'] == 'kneeOA'
all_trial_data.loc[kneeoa_mask_data, 'subject'] += KNEEOA_SUBJECT_OFFSET
clbp_mask_data = all_trial_data['dataset'] == 'cLBP'
all_trial_data.loc[clbp_mask_data, 'subject'] += CLBP_SUBJECT_OFFSET

print(f"Updated subject IDs:")
print(f"  plosONE subjects: {all_trial_metrics[all_trial_metrics['dataset'] == 'plosONE']['subject'].min()}-{all_trial_metrics[all_trial_metrics['dataset'] == 'plosONE']['subject'].max()}")
print(f"  kneeOA subjects: {all_trial_metrics[all_trial_metrics['dataset'] == 'kneeOA']['subject'].min()}-{all_trial_metrics[all_trial_metrics['dataset'] == 'kneeOA']['subject'].max()}")
print(f"  cLBP subjects: {all_trial_metrics[all_trial_metrics['dataset'] == 'cLBP']['subject'].min()}-{all_trial_metrics[all_trial_metrics['dataset'] == 'cLBP']['subject'].max()}")
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

# Get cLBP group labels from SQL database
conn = sqlite3.connect(sql_path)
clbp_groups_query = '''
SELECT DISTINCT
    subject,
    COALESCE(NULLIF("group", ''), 'High') AS group_label
FROM metadata
WHERE study LIKE 'cLBP%'
ORDER BY subject
'''
clbp_groups = pd.read_sql_query(clbp_groups_query, conn)
conn.close()

# Apply the same subject ID offset to the group labels
kneeoa_groups['subject'] += KNEEOA_SUBJECT_OFFSET
kneeoa_groups['dataset'] = 'kneeOA'
clbp_groups['subject'] += CLBP_SUBJECT_OFFSET
clbp_groups['dataset'] = 'cLBP'
# Create plosONE group labels (all controls)
plosone_subjects = all_trial_metrics[all_trial_metrics['dataset'] == 'plosONE']['subject'].unique()
plosone_groups = pd.DataFrame({
    'subject': plosone_subjects,
    'group_label': 'Control',
    'dataset': 'plosONE'
})
# Combine group labels
all_groups = pd.concat([kneeoa_groups, clbp_groups, plosone_groups], ignore_index=True)
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

# ========================================================
# Plot OH and OA magnitude by clinical group (subject-averaged)
# + pairwise Welch t-tests + FDR correction + significance bars + n subjects
# ========================================================
from itertools import combinations
from statsmodels.stats.multitest import multipletests

subj_avg = (
    unified_data[unified_data['trial_type'].isin(['onset', 'offset'])]
    .groupby(['subject', 'group_label', 'trial_type'], as_index=False)['abs_normalized_pain_change']
    .mean()
    .rename(columns={'abs_normalized_pain_change': 'mean_abs_normalized_pain_change'})
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
order = ['Control', 'Low', 'High']

panel_data = {}
tests_by_panel = {0: [], 1: []}

def p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

def add_sig_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, c='black')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')

# 1) Build per-panel datasets and run pairwise tests
for ax_idx, (trial_type, title) in enumerate([
    ('onset', 'Onset Trials'),
    ('offset', 'Offset Trials')
]):
    subset = subj_avg[subj_avg['trial_type'] == trial_type].copy()
    panel_data[ax_idx] = subset

    group_series = {
        g: subset.loc[subset['group_label'] == g, 'mean_abs_normalized_pain_change'].dropna()
        for g in order
    }
    present_groups = [g for g in order if len(group_series[g]) > 1]

    # Pairwise independent t-tests (Welch)
    raw_tests = []
    for g1, g2 in combinations(present_groups, 2):
        t_stat, p_raw = stats.ttest_ind(group_series[g1], group_series[g2], equal_var=False, nan_policy='omit')
        raw_tests.append({'g1': g1, 'g2': g2, 't_stat': t_stat, 'p_raw': p_raw})

    # FDR correction within this panel
    if raw_tests:
        pvals = [t['p_raw'] for t in raw_tests]
        reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, t in enumerate(raw_tests):
            t['p_fdr'] = p_fdr[i]
            t['sig'] = bool(reject[i])
            tests_by_panel[ax_idx].append(t)

# 2) Plot each panel and annotate n + significance bars
for ax_idx, ax in enumerate(axes):
    subset = panel_data[ax_idx]
    trial_type = 'onset' if ax_idx == 0 else 'offset'
    title = 'Onset Trials' if ax_idx == 0 else 'Offset Trials'

    sns.violinplot(
        data=subset,
        x='group_label',
        y='mean_abs_normalized_pain_change',
        order=order,
        palette=GROUP_COLORS,
        inner='box',
        ax=ax
    )

    sns.stripplot(
        data=subset,
        x='group_label',
        y='mean_abs_normalized_pain_change',
        order=order,
        color='black',
        alpha=0.35,
        size=4,
        ax=ax
    )

    ax.set_title(f'{title}: Subject-Averaged Normalized Pain Change')
    ax.set_xlabel('Clinical Group')
    ax.set_ylabel('Normalized Pain Change (%)')
    ax.grid(True, alpha=0.3)

    # n subjects per group
    counts = (
        subset.groupby('group_label')['subject']
        .nunique()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )

    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-6)
    y_n = y_min + 0.03 * y_span

    for xi, g in enumerate(order):
        ax.text(
            xi, y_n, f'n={counts[g]}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
        )

    # Significant pairs for this panel
    sig_tests = [t for t in tests_by_panel[ax_idx] if t['sig']]
    if sig_tests:
        x_map = {g: i for i, g in enumerate(order)}
        base_y = y_max + 0.04 * y_span
        step = 0.08 * y_span
        h = 0.02 * y_span

        for k, t in enumerate(sig_tests):
            x1, x2 = x_map[t['g1']], x_map[t['g2']]
            y = base_y + k * step
            add_sig_bar(ax, x1, x2, y, h, p_to_stars(t['p_fdr']))

        top = base_y + (len(sig_tests) - 1) * step + h + 0.06 * y_span
        ax.set_ylim(y_min, top)

plt.tight_layout()
plt.savefig(
    f'{FIGPATH}/normalized_pain_change_by_group_split_by_trial_type_subject_avg.png',
    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none'
)
plt.show()


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
                
                # CALCULATE correlation
                r, p = stats.pearsonr(group_data[preceding_metric], 
                                     group_data['abs_normalized_pain_change'])
                
                # STORE correlation for FDR correction later
                all_correlations.append({
                    'idx': idx,
                    'trial_type': trial_type,
                    'metric': preceding_metric,
                    'direction': direction,
                    'metric_label': metric_label,
                    'group': group,
                    'r': r,
                    'p_raw': p,
                    'n': len(group_data),
                    'ax': ax,  # Store axis reference
                    'group_idx': group_idx,
                    'group_data': group_data  # Store data for regression line
                })
    
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
# ================================================================================================================
# ######################################## 3. HABITUATORS VS SENSITIZERS ANALYSIS ###############################
# ================================================================================================================
import sys
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/alter_collab_analysis/single_dataset_analysis/')
from habituator_sensitizers_ControlTrial import calculate_slope, calculate_windowed_difference, calculate_auc_difference, calculate_normalized_time_aware_change, bootstrap_classification, plot_hold_trials_for_subject

hold_trials = unified_data[unified_data['trial_type'].astype(str).str.contains('hold', case=False, na=False)].copy()
hold_results = []
for _, row in hold_trials.iterrows():
    subject_id = row['subject']
    dataset_name = row['dataset']
    trial_num = row['trial_num']
    c_start = row['C_start']
    c_end = row['C_end']
    a_start = row['A_start']

    trial_ts = all_trial_data[
        (all_trial_data['dataset'] == dataset_name) &
        (all_trial_data['subject'] == subject_id) &
        (all_trial_data['trial_num'] == trial_num)
    ].copy()

    if len(trial_ts) < 2:
        continue

    slope = calculate_slope(trial_ts, C_start=c_start, C_end=c_end)
    diff_5s = calculate_windowed_difference(trial_ts, window_size=5, A_start=a_start, C_end=c_end)
    diff_10s = calculate_windowed_difference(trial_ts, window_size=10, A_start=a_start, C_end=c_end)
    auc_diff_10s = calculate_auc_difference(trial_ts, window_size=10, A_start=a_start, C_end=c_end)
    norm_change = calculate_normalized_time_aware_change(row, max_floor=5)

    hold_results.append({
        'dataset': dataset_name,
        'subject': subject_id,
        'trial_num': trial_num,
        'trial_type': row['trial_type'],
        'slope': slope,
        'late5_minus_early5': diff_5s,
        'late10_minus_early10': diff_10s,
        'auc_diff_10s': auc_diff_10s,
        'time_aware_norm_change': norm_change
    })

hold_metrics_df = pd.DataFrame(hold_results)

classification_results = []

for (dataset_name, subject_id), subject_trials in hold_metrics_df.groupby(['dataset', 'subject']):
    classification, observed_mean, lower_bound, upper_bound = bootstrap_classification(
        subject_trials['slope'],
        n_boot=10000,
        ci=95,
        random_state=42
    )

    classification_results.append({
        'dataset': dataset_name,
        'subject': subject_id,
        'n_trials': len(subject_trials),
        'observed_mean_slope': observed_mean,
        'classification': classification,
        'ci_lower': lower_bound,
        'ci_upper': upper_bound
    })
slope_classification_df = pd.DataFrame(classification_results)

print(f"Analyzed hold trial slopes for {len(slope_classification_df)} subjects")
unified_data = unified_data.merge(
    slope_classification_df[['dataset', 'subject', 'classification', 'observed_mean_slope']],
    on=['dataset', 'subject'],
    how='left'
)
subject_level_slopes = slope_classification_df.merge(
    all_groups[['subject','group_label']],
    on='subject',
    how='left'
)


# Plot distribution of slopes first
plt.figure(figsize=(12, 8))
# Plot by clinical group
for group in ['Control', 'Low', 'High']:
    if group in subject_level_slopes['group_label'].values:
        group_data = subject_level_slopes[subject_level_slopes['group_label'] == group]['observed_mean_slope']
        plt.hist(group_data, bins=15, alpha=0.6, 
                color=GROUP_COLORS[group], 
                label=f'{group} (n={len(group_data)})',
                edgecolor='black', linewidth=0.5)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No change')
plt.xlabel('Mean Hold-Trial Slope in Period C', fontsize=12)
plt.ylabel('Number of Subjects', fontsize=12)
plt.title('Distribution of Individual Pain Slopes by Clinical Group', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add statistics
stats_text = []
for group in ['Control', 'Low', 'High']:
    if group in subject_level_slopes['group_label'].values:
        group_data = subject_level_slopes[subject_level_slopes['group_label'] == group]['observed_mean_slope']
        mean_slope = group_data.mean()
        std_slope = group_data.std()
        stats_text.append(f'{group}: μ={mean_slope:.2f}, σ={std_slope:.2f}')

plt.text(0.02, 0.98, '\n'.join(stats_text), 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

slope_classification_df.to_csv(
    f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/holdslope_classifications.csv',
    index=False
)

# ========================================================
# EXAMPLE SUBJECTS FOR EACH TRAJECTORY TYPE
# ========================================================
hab_subject = slope_classification_df[
    slope_classification_df['classification'] == 'habituator'
]['subject'].sample(1).iloc[0]

sens_subject = slope_classification_df[
    slope_classification_df['classification'] == 'sensitizer'
]['subject'].sample(1).iloc[0]

nr_subject = slope_classification_df[
    slope_classification_df['classification'] == 'no trend'
]['subject'].sample(1).iloc[0]

# Plot examples
plot_hold_trials_for_subject(hab_subject, hold_metrics_df, all_trial_data, title=f"Example Habituator (Subject {hab_subject})")
plot_hold_trials_for_subject(sens_subject, hold_metrics_df, all_trial_data, title=f"Example Sensitizer (Subject {sens_subject})")
plot_hold_trials_for_subject(nr_subject, hold_metrics_df, all_trial_data, title=f"Example No Trend (Subject {nr_subject})")



#%%
# ==========================================================================
# TEMPORAL CONTRAST BY TRAJECTORY - CORRECTED FOR PSEUDOREPLICATION
# ==========================================================================
print("=== CORRECTING FOR PSEUDOREPLICATION ===")
print("Averaging trials within subjects first...")
contrast_analysis = unified_data.copy()
# Calculate subject-level averages for each trial type
subject_averages = []

for subject, subj_data in contrast_analysis.groupby('subject'):
    classification = subj_data['classification'].iloc[0]
    # Average onset trials for this subject
    onset_trials = subj_data[subj_data['trial_type'] == 'onset']['abs_normalized_pain_change']
    offset_trials = subj_data[subj_data['trial_type'] == 'offset']['abs_normalized_pain_change']
    
    if len(onset_trials) > 0:
        subject_averages.append({
            'subject': subject,
            'classification': classification,
            'trial_type': 'onset',
            'avg_normalized_pain_change': onset_trials.mean(),
            'n_trials': len(onset_trials)
        })
    
    if len(offset_trials) > 0:
        subject_averages.append({
            'subject': subject,
            'classification': classification,
            'trial_type': 'offset', 
            'avg_normalized_pain_change': offset_trials.mean(),
            'n_trials': len(offset_trials)
        })

subject_avg_df = pd.DataFrame(subject_averages)

print(f"Subject-level data: {len(subject_avg_df)} subject-trial_type combinations")
print(f"From {subject_avg_df['subject'].nunique()} unique subjects")

# Now plot using subject averages (proper n)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# Define colors and order
traj_colors = {'habituator': 'blue', 'no trend': 'gray', 'sensitizer': 'red'}
traj_order = ['habituator', 'no trend', 'sensitizer']

def add_significance_brackets(ax, x1, x2, y, h, text, fontsize=12):
    """Add significance brackets between bars"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', 
            fontweight='bold', fontsize=fontsize)

# ONSET TRIALS
onset_subj_data = subject_avg_df[subject_avg_df['trial_type'] == 'onset']
if len(onset_subj_data) > 0:
    sns.violinplot(data=onset_subj_data, x='classification', y='avg_normalized_pain_change',
                   palette=traj_colors, inner='box', ax=axes[0], order=traj_order)
    axes[0].set_title('Onset Hyperalgesia by Classification\n(Subject Averages)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Classification', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Average Normalized Pain Change (%)', fontweight='bold', fontsize=12)
    
    # Add sample sizes (now subjects, not trials!)
    for i, traj_group in enumerate(traj_order):
        n_subjects = len(onset_subj_data[onset_subj_data['classification'] == traj_group])
        if n_subjects > 0:
            axes[0].text(i, axes[0].get_ylim()[0] + 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), 
                        f'n={n_subjects} subjects', ha='center', fontweight='bold', fontsize=11)

# OFFSET TRIALS  
offset_subj_data = subject_avg_df[subject_avg_df['trial_type'] == 'offset']
if len(offset_subj_data) > 0:
    sns.violinplot(data=offset_subj_data, x='classification', y='avg_normalized_pain_change',
                   palette=traj_colors, inner='box', ax=axes[1], order=traj_order)
    axes[1].set_title('Offset Analgesia by Classification\n(Subject Averages)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Classification', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Average Normalized Pain Change (%)', fontweight='bold', fontsize=12)
    
    # Add sample sizes
    for i, traj_group in enumerate(traj_order):
        n_subjects = len(offset_subj_data[offset_subj_data['classification'] == traj_group])
        if n_subjects > 0:
            axes[1].text(i, axes[1].get_ylim()[0] + 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]), 
                        f'n={n_subjects} subjects', ha='center', fontweight='bold', fontsize=11)

# Statistics on subject averages
print("\n=== STATISTICS ON SUBJECT AVERAGES ===")
# Onset comparisons
onset_groups = {}
for traj_group in traj_order:
    group_data = onset_subj_data[onset_subj_data['classification'] == traj_group]['avg_normalized_pain_change']
    if len(group_data) > 0:
        onset_groups[traj_group] = group_data
        print(f"Onset {traj_group}: n={len(group_data)} subjects, mean={group_data.mean():.2f}")

# Offset comparisons  
offset_groups = {}
for traj_group in traj_order:
    group_data = offset_subj_data[offset_subj_data['classification'] == traj_group]['avg_normalized_pain_change']
    if len(group_data) > 0:
        offset_groups[traj_group] = group_data
        print(f"Offset {traj_group}: n={len(group_data)} subjects, mean={group_data.mean():.2f}")

# Statistical tests with FDR correction
print("\n--- PAIRWISE T-TESTS (with FDR correction) ---")
from itertools import combinations

# Onset pairwise comparisons
onset_sig_pairs = []
if len(onset_groups) > 1:
    print("\nONSET ONSET:")
    onset_pvalues = []
    onset_comparisons = []
    for group1, group2 in combinations(traj_order, 2):
        if group1 in onset_groups and group2 in onset_groups:
            t_stat, p_val = scipy_stats.ttest_ind(onset_groups[group1], onset_groups[group2])
            onset_pvalues.append(p_val)
            onset_comparisons.append((group1, group2, t_stat, p_val))
    
    # Apply FDR correction
    if onset_pvalues:
        from statsmodels.stats.multitest import multipletests
        reject, pvals_corrected, _, _ = multipletests(onset_pvalues, alpha=0.05, method='fdr_bh')
        for (group1, group2, t_stat, p_val), p_corr, is_sig in zip(onset_comparisons, pvals_corrected, reject):
            sig_marker = "***" if is_sig else "ns"
            print(f"  {group1} vs {group2}: t={t_stat:.3f}, p_orig={p_val:.4f}, p_FDR={p_corr:.4f} {sig_marker}")
            if is_sig:
                onset_sig_pairs.append((traj_order.index(group1), traj_order.index(group2)))

# Offset pairwise comparisons
offset_sig_pairs = []
if len(offset_groups) > 1:
    print("\nOFFSET:")
    offset_pvalues = []
    offset_comparisons = []
    for group1, group2 in combinations(traj_order, 2):
        if group1 in offset_groups and group2 in offset_groups:
            t_stat, p_val = scipy_stats.ttest_ind(offset_groups[group1], offset_groups[group2])
            offset_pvalues.append(p_val)
            offset_comparisons.append((group1, group2, t_stat, p_val))
    
    # Apply FDR correction
    if offset_pvalues:
        from statsmodels.stats.multitest import multipletests
        reject, pvals_corrected, _, _ = multipletests(offset_pvalues, alpha=0.05, method='fdr_bh')
        for (group1, group2, t_stat, p_val), p_corr, is_sig in zip(offset_comparisons, pvals_corrected, reject):
            sig_marker = "***" if is_sig else "ns"
            print(f"  {group1} vs {group2}: t={t_stat:.3f}, p_orig={p_val:.4f}, p_FDR={p_corr:.4f} {sig_marker}")
            if is_sig:
                offset_sig_pairs.append((traj_order.index(group1), traj_order.index(group2)))

# Add significance markers to plots
y_max_onset = axes[0].get_ylim()[1]
line_height = y_max_onset * 0.02
for idx, (i1, i2) in enumerate(onset_sig_pairs):
    y_pos = y_max_onset * (0.95 + idx * 0.12)
    axes[0].plot([i1, i2], [y_pos, y_pos], 'k-', linewidth=1.5)
    x_pos = (i1 + i2) / 2
    axes[0].text(x_pos, y_pos + line_height, '***', ha='center', fontsize=12, fontweight='bold', color='black')

y_max_offset = axes[1].get_ylim()[1]
line_height = y_max_offset * 0.02
for idx, (i1, i2) in enumerate(offset_sig_pairs):
    y_pos = y_max_offset * (0.95 + idx * 0.12)
    axes[1].plot([i1, i2], [y_pos, y_pos], 'k-', linewidth=1.5)
    x_pos = (i1 + i2) / 2
    axes[1].text(x_pos, y_pos + line_height, '***', ha='center', fontsize=12, fontweight='bold', color='black')

plt.tight_layout()
plt.show()


#%%
# QUESTION: Is there a significant difference in HOLD slopes across clinical groups?
# Statistical test: Do trajectory slopes differ across clinical groups?
print(f"\n{'='*60}")
print("HOLD SLOPES BY CLINICAL GROUP")
print(f"{'='*60}")
# add group labels to slope_classification_df
if 'group_label' not in slope_classification_df.columns:
    slope_classification_df = slope_classification_df.merge(
        all_groups[['subject', 'group_label']],
        on='subject',
        how='left'
    )

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Box plot of slopes by group
sns.boxplot(data=slope_classification_df, x='group_label', y='observed_mean_slope', palette=GROUP_COLORS, ax=axes[0])
sns.stripplot(data=slope_classification_df, x='group_label', y='observed_mean_slope', color='black', alpha=0.4, size=6, ax=axes[0])
axes[0].set_title('Distribution of HOLD Slopes by Clinical Group', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Clinical Group', fontweight='bold')
axes[0].set_ylabel('Mean HOLD Slope', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Violin plot for better distribution visualization
sns.violinplot(data=slope_classification_df, x='group_label', y='observed_mean_slope', palette=GROUP_COLORS, ax=axes[1])
axes[1].set_title('Density Distribution of HOLD Slopes by Clinical Group', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Clinical Group', fontweight='bold')
axes[1].set_ylabel('Mean HOLD Slope', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Statistical analysis: ANOVA + pairwise tests with FDR, then plot sig bars
print("\nStatistical Comparison of Slopes Across Groups:")
group_order = ['Control', 'Low', 'High']
present_groups = []
groups_list = []

for group in group_order:
    group_data = slope_classification_df[
        slope_classification_df['group_label'] == group
    ]['observed_mean_slope'].dropna()
    if len(group_data) > 0:
        present_groups.append(group)
        groups_list.append(group_data.values)
        print(f"\n{group}:")
        print(f"  n = {len(group_data)}")
        print(f"  Mean slope = {group_data.mean():.3f}")
        print(f"  Std = {group_data.std():.3f}")
        print(f"  Range = [{group_data.min():.3f}, {group_data.max():.3f}]")

def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def add_sig_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')

sig_pairs = []  # tuples: (group1, group2, p_fdr)

# One-way ANOVA
if len(groups_list) >= 2:
    f_stat, p_anova = stats.f_oneway(*groups_list)
    sig = p_to_stars(p_anova)
    print(f"\nOne-way ANOVA: F = {f_stat:.3f}, p = {p_anova:.4f} {sig}")

    if p_anova < 0.05:
        print("Result: Significant difference in HOLD slopes across clinical groups")
    else:
        print("Result: No significant difference in HOLD slopes across clinical groups")

    # Pairwise post-hoc only if omnibus ANOVA is significant
    if p_anova < 0.05 and len(present_groups) >= 2:
        from itertools import combinations
        from statsmodels.stats.multitest import multipletests

        comparisons = []
        raw_p = []

        for g1, g2 in combinations(present_groups, 2):
            d1 = slope_classification_df[
                slope_classification_df['group_label'] == g1
            ]['observed_mean_slope'].dropna()
            d2 = slope_classification_df[
                slope_classification_df['group_label'] == g2
            ]['observed_mean_slope'].dropna()

            # Welch t-test is safer when variances/sample sizes differ
            t_stat, p_val = scipy_stats.ttest_ind(d1, d2, equal_var=False, nan_policy='omit')
            comparisons.append((g1, g2, t_stat, p_val))
            raw_p.append(p_val)

        reject, p_fdr, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')

        print("\nPairwise post-hoc tests (Welch t-test, FDR corrected):")
        for (g1, g2, t_stat, p_val), keep, p_corr in zip(comparisons, reject, p_fdr):
            mark = p_to_stars(p_corr)
            print(f"  {g1} vs {g2}: t={t_stat:.3f}, p_raw={p_val:.4f}, p_FDR={p_corr:.4f} {mark}")
            if keep:
                sig_pairs.append((g1, g2, p_corr))

# Add significance bars to both axes only for significant pairwise differences
if sig_pairs:
    x_pos = {g: i for i, g in enumerate(group_order)}
    y = slope_classification_df['observed_mean_slope'].dropna()
    y_min, y_max = y.min(), y.max()
    y_span = max(y_max - y_min, 1.0)

    y_start = y_max + 0.06 * y_span
    y_step = 0.10 * y_span
    bar_h = 0.03 * y_span

    for i, (g1, g2, p_corr) in enumerate(sig_pairs):
        x1, x2 = x_pos[g1], x_pos[g2]
        y_i = y_start + i * y_step
        label = p_to_stars(p_corr)
        add_sig_bar(axes[0], x1, x2, y_i, bar_h, label)
        add_sig_bar(axes[1], x1, x2, y_i, bar_h, label)

    # Expand y-limits so bars are visible
    top = y_start + (len(sig_pairs) - 1) * y_step + bar_h + 0.08 * y_span
    axes[0].set_ylim(y_min - 0.05 * y_span, top)
    axes[1].set_ylim(y_min - 0.05 * y_span, top)

plt.tight_layout()
plt.savefig(f'{FIGPATH}/hold_slopes_by_group.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
# %% 
# Break down trial_sequence effects by trajectory classification - 2x2 OVERLAY PLOTS
print(f"\n{'='*60}")
print("TRIAL SEQUENCE EFFECTS BY TRAJECTORY CLASSIFICATION - 2x2 OVERLAY PLOTS")
print(f"{'='*60}")
# Prepare plot_data
plot_data = unified_data[unified_data['trial_type'].isin(['onset', 'offset'])].copy()
plot_data = plot_data[plot_data['classification'].isin(['habituator', 'sensitizer', 'no trend'])].copy()

metric = 'preceding_abs_normalized_pain_change'
metric_label = 'Preceding Normalized Pain Change'
traj_groups = ['habituator', 'no trend', 'sensitizer']
traj_colors = {'habituator': 'blue', 'sensitizer': 'red', 'no trend': 'grey'}
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
                group_data = subset[subset['classification'] == traj_group]
                
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
# Chi-squared analysis: classification vs clinical group
print("\n=== CHI-SQUARED ANALYSIS: CLASSIFICATION vs CLINICAL GROUP ===")

# Create contingency table
subject_table = (
subject_level_slopes[['subject', 'group_label', 'classification']]
.dropna(subset=['group_label', 'classification'])
.drop_duplicates(subset=['subject'])
)

contingency_table = (
subject_table
.groupby(['classification', 'group_label'])
.size()
.unstack(fill_value=0)
.reindex(index=['sensitizer', 'no trend', 'habituator'],
columns=['Control', 'Low', 'High'],
fill_value=0)
)

print(contingency_table)

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
    print("Result: Significant association between classification and clinical group")
else:
    print("Result: No significant association between classification and clinical group")

# Plot it out too
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Oranges', cbar=False)
plt.title('Contingency Table: Classification vs Clinical Group', fontweight='bold')
plt.xlabel('Clinical Group', fontweight='bold')
plt.ylabel('Classification', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/contingency_table_classification_vs_group.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# %% Look at correlations between hold slopes and clinical measures (PCS, SPCS, pain ratings, HPT)
full_patient_info_path = '/Users/ljohnston1/UCSF DBS for Pain Dropbox/PainNeuromodulationLab/DATA ANALYSIS/Lucy/BenAlter_Collab_Data/KneeNIRS data/kneeNIRS dataset from MMtrimmedcsv250107.xlsx'
full_patient_info = pd.read_excel(full_patient_info_path)

columns_of_interest = ['PCS_rumination','PCS_magnification','SPCS_QST','pain_now','HPT_forearm_avg']



# %%
