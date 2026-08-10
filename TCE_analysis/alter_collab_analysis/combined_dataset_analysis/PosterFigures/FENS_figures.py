"""
This script combines the datasets and looks at clinical population vs. various metrics 
Updated 3/11/26 to include plosONE and kneeOA
Will have to update again when I get the cLBP dataset 
"""
#%%
# ==================================================================================================================
######################################## LOAD THE DATA ########################################
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
FIGPATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/UCSF/0_PainLab/FIGURES/2026/FENS2026/'
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
######################################## OA/OH MAGNITUDE BY CLINICAL GROUP ########################################
# ==================================================================================================================
# Define consistent colors for clinical groups
GROUP_COLORS = {
    'Control': '#2E8B57',    # Green
    'Low': '#FF8C00',        # Orange  
    'High': '#DC143C'        # Red
}

# ========================================================
# Plot OH and OA magnitude by clinical group (subject-averaged)
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

# 1) Build per-panel datasets and run LME pairwise tests
for ax_idx, (trial_type, title) in enumerate([
    ('onset', 'Onset Trials'),
    ('offset', 'Offset Trials')
]):
    tests_by_panel[ax_idx] = []
    panel_data[ax_idx] = subj_avg[subj_avg['trial_type'] == trial_type].copy()

    # Use raw trial-level data (all replicates) for LME — random intercept handles within-subject clustering
    lme_data = (
        unified_data[
            (unified_data['trial_type'] == trial_type) &
            unified_data['group_label'].isin(order) &
            unified_data['abs_normalized_pain_change'].notna()
        ]
        .copy()
    )

    try:
        # Fit LME: pain_change ~ group + (1 | subject), Control as reference
        result = mixedlm(
            "abs_normalized_pain_change ~ C(group_label, Treatment('Control'))",
            data=lme_data,
            groups=lme_data['subject']
        ).fit(reml=True, disp=False)

        print(f"\n{'='*60}")
        print(f"Panel: {title} — LME (random intercept per subject, REML)")
        print(f"{'='*60}")
        print(result.summary())

        # Pairwise contrasts from the fitted model, FDR corrected
        # Contrast vector indexes into result.params (fixed effects + Group Var at end)
        param_names = list(result.params.index)
        raw_tests = []
        for g1, g2 in combinations(order, 2):
            contrast = np.zeros(len(param_names))
            for i, name in enumerate(param_names):
                if f'T.{g2}' in str(name):
                    contrast[i] = 1.0
                if f'T.{g1}' in str(name):
                    contrast[i] = -1.0
            t_res = result.t_test(contrast)
            raw_tests.append({
                'g1': g1, 'g2': g2,
                't_stat': float(np.squeeze(t_res.tvalue)),
                'p_raw': float(np.squeeze(t_res.pvalue)),
            })

        pvals = [t['p_raw'] for t in raw_tests]
        reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

        print(f"\n{'Comparison':<35} {'t-stat':>8} {'p-raw':>10} {'p-FDR':>10} {'Sig?':>6}")
        print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
        for i, t in enumerate(raw_tests):
            t['p_fdr'] = p_fdr[i]
            t['sig'] = bool(reject[i])
            tests_by_panel[ax_idx].append(t)
            comp = f"{t['g1']} vs {t['g2']}"
            sig_marker = p_to_stars(t['p_fdr'])
            print(f"{comp:<35} {t['t_stat']:>8.3f} {t['p_raw']:>10.4f} {t['p_fdr']:>10.4f} {sig_marker:>6}")

    except Exception as e:
        print(f"LME failed for {title}: {e}")


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
    f'{FIGPATH}/normalized_pain_change_by_group_split_by_trial_type_subject_avg.svg',
    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none'
)
plt.show()


#%%
# ==================================================================================================================
########################################### PREVIOUS TRIAL X CURRENT TRIAL ########################################
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
        text_y_positions = [0.95, 0.85, 0.75]
        
        # Create scatter plot by group and collect correlations
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

        # Add overall regression line (all groups combined) in black
        x_all = plot_data[preceding_metric].to_numpy()
        y_all = plot_data['abs_normalized_pain_change'].to_numpy()
        n_all = len(x_all)

        r_all, p_all = stats.pearsonr(x_all, y_all)
        z_all = np.polyfit(x_all, y_all, 1)
        p_fit_all = np.poly1d(z_all)
        x_range_all = np.linspace(x_all.min(), x_all.max(), 100)
        y_fit_all = p_fit_all(x_range_all)

        # 95% CI band for the overall regression line
        dof_all = n_all - 2
        resid_std_err_all = np.sqrt(np.sum((y_all - p_fit_all(x_all)) ** 2) / dof_all)
        x_mean_all = x_all.mean()
        sxx_all = np.sum((x_all - x_mean_all) ** 2)
        se_fit_all = resid_std_err_all * np.sqrt(1 / n_all + (x_range_all - x_mean_all) ** 2 / sxx_all)
        ci_all = stats.t.ppf(0.975, dof_all) * se_fit_all

        ax.plot(x_range_all, y_fit_all,
               color='black', linewidth=2.5, alpha=0.9, label='All', zorder=5)
        ax.fill_between(x_range_all, y_fit_all - ci_all, y_fit_all + ci_all,
                        color='black', alpha=0.12, linewidth=0, zorder=4)
        ax.text(0.05, 0.65, f'All: r={r_all:.2f}, p={p_all:.3f}',
               transform=ax.transAxes, fontsize=8, color='black', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

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
            x = group_data[corr['metric']].to_numpy()
            y = group_data['abs_normalized_pain_change'].to_numpy()
            n = len(x)

            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_fit = p_fit(x_range)

            # 95% CI band for the regression line (CI on the mean response, not the raw data spread)
            dof = n - 2
            resid_std_err = np.sqrt(np.sum((y - p_fit(x)) ** 2) / dof)
            x_mean = x.mean()
            sxx = np.sum((x - x_mean) ** 2)
            se_fit = resid_std_err * np.sqrt(1 / n + (x_range - x_mean) ** 2 / sxx)
            ci = stats.t.ppf(0.975, dof) * se_fit

            corr['ax'].plot(x_range, y_fit,
                           color=GROUP_COLORS[corr['group']],
                           linestyle='--', linewidth=2, alpha=0.8)
            corr['ax'].fill_between(x_range, y_fit - ci, y_fit + ci,
                                   color=GROUP_COLORS[corr['group']], alpha=0.15, linewidth=0)
        
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
plt.savefig(f"{FIGPATH}/sequences_by_clinical_group.svg")
plt.tight_layout()
plt.show()

#%%
# ================================================================================================================
# ######################################## HABITUATORS VS SENSITIZERS ###############################
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
    hold_results.append({
        'dataset': dataset_name,
        'subject': subject_id,
        'trial_num': trial_num,
        'trial_type': row['trial_type'],
        'slope': slope
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
plt.figure(figsize=(10, 8))
# Plot by classification (habituator, sensitizer, no trend)
class_colors = {'habituator': 'blue', 'no trend': 'gray', 'sensitizer': 'red'}
for group in ['habituator', 'no trend', 'sensitizer']:
    if group in subject_level_slopes['classification'].values:
        group_data = subject_level_slopes[subject_level_slopes['classification'] == group]['observed_mean_slope']
        plt.hist(group_data, bins=50, alpha=0.6,
                color=class_colors[group],
                label=f'{group} (n={len(group_data)})',
                edgecolor='black', linewidth=0.3)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No change')
plt.xlabel('Mean Hold-Trial Slope in Period C', fontsize=12)
plt.ylabel('Number of Subjects', fontsize=12)
plt.title('Distribution of Individual Pain Slopes by Classification', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add statistics
stats_text = []
for group in ['habituator', 'no trend', 'sensitizer']:
    if group in subject_level_slopes['classification'].values:
        group_data = subject_level_slopes[subject_level_slopes['classification'] == group]['observed_mean_slope']
        mean_slope = group_data.mean()
        std_slope = group_data.std()
        stats_text.append(f'{group}: μ={mean_slope:.2f}, σ={std_slope:.2f}')

plt.text(0.02, 0.98, '\n'.join(stats_text), 
         transform=plt.gca().transAxes, 
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(f"{FIGPATH}/slope_distribution_classification.svg")
plt.show()

# ========================================================
# EXAMPLE SUBJECTS FOR EACH CLASSIFICATION TYPE
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

# Plot examples - stacked subplots sharing an x-axis
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
plot_hold_trials_for_subject(hab_subject, hold_metrics_df, all_trial_data, title=f"Example Habituator (Subject {hab_subject})", ax1=axes[0])
plot_hold_trials_for_subject(sens_subject, hold_metrics_df, all_trial_data, title=f"Example Sensitizer (Subject {sens_subject})", ax1=axes[1])
plot_hold_trials_for_subject(nr_subject, hold_metrics_df, all_trial_data, title=f"Example No Trend (Subject {nr_subject})", ax1=axes[2])

# Only label the x-axis on the bottom subplot since they share one
axes[0].set_xlabel('')
axes[1].set_xlabel('')
for ax in axes:
    ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{FIGPATH}example_trajectories.svg", format='svg', bbox_inches='tight')
plt.show()


#%%
# ==========================================================================
# TEMPORAL CONTRAST BY CLASSIFICATION
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
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Define colors and order
class_colors = {'habituator': 'blue', 'no trend': 'gray', 'sensitizer': 'red'}
class_order = ['habituator', 'no trend', 'sensitizer']

def add_significance_brackets(ax, x1, x2, y, h, text, fontsize=12):
    """Add significance brackets between bars"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', 
            fontweight='bold', fontsize=fontsize)

# OFFSET TRIALS  
offset_subj_data = subject_avg_df[subject_avg_df['trial_type'] == 'offset']
if len(offset_subj_data) > 0:
    sns.violinplot(data=offset_subj_data, x='classification', y='avg_normalized_pain_change',
                   palette=class_colors, inner='box', ax=axes[0], order=class_order)
    axes[0].set_title('Offset Analgesia by Classification\n(Subject Averages)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Classification', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Average Normalized Pain Change (%)', fontweight='bold', fontsize=12)
    
    # Add sample sizes
    for i, class_group in enumerate(class_order):
        n_subjects = len(offset_subj_data[offset_subj_data['classification'] == class_group])
        if n_subjects > 0:
            axes[0].text(i, axes[0].get_ylim()[0] + 0.02 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]), 
                        f'n={n_subjects} subjects', ha='center', fontweight='bold', fontsize=11)


# ONSET TRIALS
onset_subj_data = subject_avg_df[subject_avg_df['trial_type'] == 'onset']
if len(onset_subj_data) > 0:
    sns.violinplot(data=onset_subj_data, x='classification', y='avg_normalized_pain_change',
                   palette=class_colors, inner='box', ax=axes[1], order=class_order)
    axes[1].set_title('Onset Hyperalgesia by Classification\n(Subject Averages)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Classification', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Average Normalized Pain Change (%)', fontweight='bold', fontsize=12)
    
    # Add sample sizes (now subjects, not trials!)
    for i, class_group in enumerate(class_order):
        n_subjects = len(onset_subj_data[onset_subj_data['classification'] == class_group])
        if n_subjects > 0:
            axes[1].text(i, axes[1].get_ylim()[0] + 0.02 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]), 
                        f'n={n_subjects} subjects', ha='center', fontweight='bold', fontsize=11)


# Statistics on subject averages
print("\n=== STATISTICS ON SUBJECT AVERAGES ===")
# Onset comparisons
onset_groups = {}
for class_group in class_order:
    group_data = onset_subj_data[onset_subj_data['classification'] == class_group]['avg_normalized_pain_change']
    if len(group_data) > 0:
        onset_groups[class_group] = group_data
        print(f"Onset {class_group}: n={len(group_data)} subjects, mean={group_data.mean():.2f}")

# Offset comparisons  
offset_groups = {}
for class_group in class_order:
    group_data = offset_subj_data[offset_subj_data['classification'] == class_group]['avg_normalized_pain_change']
    if len(group_data) > 0:
        offset_groups[class_group] = group_data
        print(f"Offset {class_group}: n={len(group_data)} subjects, mean={group_data.mean():.2f}")

# Statistical tests with FDR correction
print("\n--- PAIRWISE T-TESTS (with FDR correction) ---")
from itertools import combinations

# Onset pairwise comparisons
onset_sig_pairs = []
if len(onset_groups) > 1:
    print("\nONSET ONSET:")
    onset_pvalues = []
    onset_comparisons = []
    for group1, group2 in combinations(class_order, 2):
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
                onset_sig_pairs.append((class_order.index(group1), class_order.index(group2)))

# Offset pairwise comparisons
offset_sig_pairs = []
if len(offset_groups) > 1:
    print("\nOFFSET:")
    offset_pvalues = []
    offset_comparisons = []
    for group1, group2 in combinations(class_order, 2):
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
                offset_sig_pairs.append((class_order.index(group1), class_order.index(group2)))

# Add significance markers to plots
y_max_onset = axes[1].get_ylim()[1]
line_height = y_max_onset * 0.02
for idx, (i1, i2) in enumerate(onset_sig_pairs):
    y_pos = y_max_onset * (0.95 + idx * 0.12)
    axes[1].plot([i1, i2], [y_pos, y_pos], 'k-', linewidth=1.5)
    x_pos = (i1 + i2) / 2
    axes[1].text(x_pos, y_pos + line_height, '***', ha='center', fontsize=12, fontweight='bold', color='black')

y_max_offset = axes[0].get_ylim()[1]
line_height = y_max_offset * 0.02
for idx, (i1, i2) in enumerate(offset_sig_pairs):
    y_pos = y_max_offset * (0.95 + idx * 0.12)
    axes[0].plot([i1, i2], [y_pos, y_pos], 'k-', linewidth=1.5)
    x_pos = (i1 + i2) / 2
    axes[0].text(x_pos, y_pos + line_height, '***', ha='center', fontsize=12, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig(f"{FIGPATH}/OA_OH_by_classification.svg")
plt.show()


#%%
# QUESTION: Is there a significant difference in HOLD slopes across clinical groups?
# Statistical test: Do classification slopes differ across clinical groups?
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

# Box plot of slopes by group
fig, ax = plt.subplots(figsize=(8, 8))
sns.boxplot(data=slope_classification_df, x='group_label', y='observed_mean_slope', palette=GROUP_COLORS, order=['Control', 'Low', 'High'], ax=ax)
sns.stripplot(data=slope_classification_df, x='group_label', y='observed_mean_slope', color='black', alpha=0.4, size=6, order=['Control', 'Low', 'High'], ax=ax)
ax.set_title('Distribution of HOLD Slopes by Clinical Group', fontweight='bold', fontsize=13)
ax.set_xlabel('Clinical Group', fontweight='bold')
ax.set_ylabel('Mean HOLD Slope', fontweight='bold')

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

        print("\nPairwise post-hoc tests (T-test, FDR corrected):")
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
        add_sig_bar(ax, x1, x2, y_i, bar_h, label)

    # Expand y-limits so bars are visible
    top = y_start + (len(sig_pairs) - 1) * y_step + bar_h + 0.08 * y_span
    ax.set_ylim(y_min - 0.05 * y_span, top)

plt.tight_layout()
plt.savefig(f'{FIGPATH}/hold_slopes_by_group.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()


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

# Plot it out too - one pie chart per clinical group showing classification breakdown
traj_colors = {'habituator': 'blue', 'no trend': 'gray', 'sensitizer': 'red'}
group_order = ['Control', 'Low', 'High']
traj_order = ['sensitizer', 'no trend', 'habituator']

counts = contingency_table.loc[traj_order, group_order]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, group in zip(axes, group_order):
    group_counts = counts[group]
    nonzero = group_counts[group_counts > 0]
    ax.pie(nonzero, labels=nonzero.index, colors=[traj_colors[t] for t in nonzero.index],
          autopct=lambda pct: f'{pct:.0f}%\n(n={int(round(pct / 100 * nonzero.sum()))})',
          wedgeprops=dict(edgecolor='black', linewidth=0.5), startangle=90)
    ax.set_title(f'{group} (n={int(group_counts.sum())})', fontweight='bold')

fig.suptitle(f'Classification Distribution by Clinical Group\n(chi2={chi2:.2f}, p={p:.3f})', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/contingency_table_classification_vs_group.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# Plot it out too - stacked bar chart showing classification breakdown per clinical group
fig, ax = plt.subplots(figsize=(7, 6))
bottom = np.zeros(len(group_order))
for traj in traj_order:
    values = counts.loc[traj, group_order].to_numpy()
    bars = ax.bar(group_order, values, bottom=bottom, label=traj,
                   color=traj_colors[traj], edgecolor='black', linewidth=0.5)
    for x, (v, b) in enumerate(zip(values, bottom)):
        if v > 0:
            ax.text(x, b + v / 2, f'n={int(v)}',
                    ha='center', va='center', fontsize=9)
    bottom += values

ax.set_xticks(range(len(group_order)))
ax.set_xticklabels([f'{g}\n(n={int(counts[g].sum())})' for g in group_order])
ax.set_ylabel('Number of Subjects')
ax.legend(title='Classification', bbox_to_anchor=(1.02, 1), loc='upper left')
fig.suptitle(f'Classification Distribution by Clinical Group\n(chi2={chi2:.2f}, p={p:.3f})', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/contingency_table_classification_vs_group_stacked_bar.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# Post-hoc: which clinical group pairs are driving the omnibus association?
print(f"\n{'='*60}")
print("POST-HOC PAIRWISE COMPARISONS (classification vs clinical group)")
print(f"{'='*60}")

# Plot the observed contingency table itself as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table.loc[traj_order, group_order], annot=True, fmt='d',
           cmap='Blues', linewidths=0.5, linecolor='black',
           cbar_kws={'label': 'Number of subjects'})
plt.title(f'Observed Contingency Table:\nClassification x Clinical Group\n(chi2={chi2:.2f}, p={p:.3f})',
         fontweight='bold')
plt.xlabel('Clinical Group', fontweight='bold')
plt.ylabel('Classification', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/chi2_contingency_table_heatmap.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# Standardized residuals from the omnibus table - cells with |resid| > ~2 are the main contributors
residuals = (contingency_table - expected) / np.sqrt(expected)
print("\nStandardized residuals (omnibus table):")
print(residuals.round(2))
print("(|residual| > ~2 flags cells that deviate more than expected by chance)")

# Plot the residuals - diverging colormap makes over/under-represented cells jump out
plt.figure(figsize=(8, 6))
sns.heatmap(residuals.loc[traj_order, group_order], annot=True, fmt='.2f',
           cmap='RdBu_r', center=0, vmin=-4, vmax=4,
           linewidths=0.5, linecolor='black', cbar_kws={'label': 'Standardized residual'})
plt.title('Where the Association Comes From:\nStandardized Residuals (Classification x Clinical Group)',
         fontweight='bold')
plt.xlabel('Clinical Group', fontweight='bold')
plt.ylabel('Classification', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/chi2_standardized_residuals.svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

from itertools import combinations
from statsmodels.stats.multitest import multipletests

pairwise_results = []
for g1, g2 in combinations(group_order, 2):
    sub_table = contingency_table[[g1, g2]]
    chi2_pair, p_pair, dof_pair, expected_pair = stats.chi2_contingency(sub_table)
    pairwise_results.append({'pair': f'{g1} vs {g2}', 'chi2': chi2_pair, 'p_raw': p_pair, 'dof': dof_pair})

raw_p = [r['p_raw'] for r in pairwise_results]
reject, p_fdr, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')

print("\nPairwise chi-squared tests (FDR corrected):")
for r, keep, p_corr in zip(pairwise_results, reject, p_fdr):
    r['p_fdr'] = p_corr
    r['significant'] = keep
    mark = p_to_stars(p_corr)
    sig_note = " <-- driving the association" if keep else ""
    print(f"  {r['pair']}: chi2={r['chi2']:.2f}, p_raw={r['p_raw']:.4f}, p_FDR={p_corr:.4f} {mark}{sig_note}")

"""
Phenotype fingerprint panel
----------------------------
Three columns (habituator / no-trend / sensitizer) each showing:
  - top:    hold-trial slope, last 20s of the 30s hold-temperature trial
            (plotted on the same 0-30s axis as the row below, so the
            analysis window lines up visually with the OA/OH trial)
  - bottom: offset analgesia trial curve (5s step-up, 20s step-down),
            with a dashed baseline so trough depth is easy to compare
            across columns.
  - bottom row (spanning all three columns): phenotype distribution by
    clinical group, tying the hold-trial phenotype back to pain severity.

All curves are defined by a handful of (time, pain_rating) control points
and smoothed with a monotonic cubic spline (scipy) purely for visual
polish -- swap the control points for your real group-averaged data
whenever you have it, or replace the whole smoothing step with your
actual per-timepoint traces.

Requires: matplotlib, scipy, numpy
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Colors (swap for your own palette if you like)
# ---------------------------------------------------------------------------
COLOR_HABITUATOR = "#1D9E75"   # teal
COLOR_NO_TREND   = "#888780"   # gray
COLOR_SENSITIZER = "#D85A30"   # coral

# ---------------------------------------------------------------------------
# Control points -- REPLACE THESE WITH YOUR REAL DATA
# ---------------------------------------------------------------------------
# Hold-trial slope: x = time (s) over the last 20s of the 30s hold,
# y = normalized pain rating. Only a few control points are given;
# PchipInterpolator produces a smooth monotonic-ish curve through them
# without overshooting (unlike a plain cubic spline).
HOLD_WINDOW_START = 10  # seconds -- start of the "last 20s" classification window

hold_trials = {
    "Habituator": {
        "color": COLOR_HABITUATOR,
        "x": [10, 15, 20, 25, 30],
        "y": [0.88, 0.74, 0.56, 0.34, 0.12],   # decelerating decline -> habituation
    },
    "No trend": {
        "color": COLOR_NO_TREND,
        "x": [10, 15, 20, 25, 30],
        "y": [0.5, 0.53, 0.48, 0.52, 0.5],     # flat, small wiggle
    },
    "Sensitizer": {
        "color": COLOR_SENSITIZER,
        "x": [10, 15, 20, 25, 30],
        "y": [0.12, 0.32, 0.54, 0.74, 0.9],    # accelerating rise -> sensitization
    },
}

# Offset analgesia trial: x = time (s) across the 30s trial
# (0-5s step up, 5-25s step down/hold, baseline pain = 0.5).
# Trough depth encodes OA magnitude; adjust per column.
oa_trials = {
    "Habituator": {
        "color": COLOR_HABITUATOR,
        "x": [0, 5, 8, 14, 20, 30],
        "y": [0.5, 0.75, 0.55, 0.45, 0.47, 0.5],   # shallow trough
    },
    "No trend": {
        "color": COLOR_NO_TREND,
        "x": [0, 5, 8, 14, 20, 30],
        "y": [0.5, 0.75, 0.5, 0.3, 0.4, 0.48],     # medium trough
    },
    "Sensitizer": {
        "color": COLOR_SENSITIZER,
        "x": [0, 5, 8, 14, 20, 30],
        "y": [0.5, 0.72, 0.4, 0.15, 0.3, 0.45],    # deep trough
    },
}

BASELINE = 0.5  # dashed reference line in the OA trial panels

# ---------------------------------------------------------------------------
# Clinical group phenotype distribution -- REPLACE WITH YOUR CHI-SQUARE DATA
# ---------------------------------------------------------------------------
# Fraction of each phenotype within each clinical group. Each column of
# fractions should sum to 1. This is the panel that ties the hold-trial
# phenotype back to clinical pain severity.
clinical_groups = ["Pain-free", "Low chronic pain", "High chronic pain"]
phenotype_fractions = {
    "Habituator": [0.45, 0.35, 0.20],
    "No trend":   [0.40, 0.40, 0.35],
    "Sensitizer": [0.15, 0.25, 0.45],
}
GROUP_STAT_LABEL = "\u03c7\u00b2(4) = --, p = --"  # replace with your actual test statistic

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def smooth(x, y, n=300):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pchip = PchipInterpolator(x, y)
    x_smooth = np.linspace(x.min(), x.max(), n)
    return x_smooth, pchip(x_smooth)


columns = ["Habituator", "No trend", "Sensitizer"]

fig = plt.figure(figsize=(10, 9))
gs = gridspec.GridSpec(
    3, 3, figure=fig,
    height_ratios=[1, 1, 1.1],
    hspace=0.55, wspace=0.15,
)

top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
bottom_axes = [fig.add_subplot(gs[1, i], sharex=top_axes[i]) for i in range(3)]
group_ax = fig.add_subplot(gs[2, :])

for col_idx, name in enumerate(columns):
    # --- row 1: hold-trial slope, last 20s of the 30s hold, shares the
    #     same 0-30s x-axis as the OA/OH row below it ---
    ax = top_axes[col_idx]
    data = hold_trials[name]
    xs, ys = smooth(data["x"], data["y"])
    ax.axvspan(0, HOLD_WINDOW_START, color="0.93", zorder=0)
    ax.plot(xs, ys, color=data["color"], linewidth=2.5, solid_capstyle="round", zorder=2)
    ax.set_title(name, fontsize=13, fontweight="medium", color=data["color"], pad=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 30)
    if col_idx == 0:
        ax.set_ylabel("Pain rating\n(hold-trial slope)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, labelbottom=False)

    # --- row 2: offset analgesia / onset hyperalgesia trial, same x-axis ---
    ax2 = bottom_axes[col_idx]
    data2 = oa_trials[name]
    xs2, ys2 = smooth(data2["x"], data2["y"])
    ax2.axhline(BASELINE, color="0.6", linestyle="--", linewidth=1, zorder=1)
    ax2.plot(xs2, ys2, color=data2["color"], linewidth=2.5, solid_capstyle="round", zorder=2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(0, 30)
    if col_idx == 0:
        ax2.set_ylabel("Pain rating\n(OA/OH trial)", fontsize=9)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)

top_axes[0].text(
    HOLD_WINDOW_START + 1, 1.0, "classification window",
    fontsize=7.5, color="0.45", va="top",
)

# --- row 3: phenotype distribution by clinical group (spans all columns) ---
x = np.arange(len(clinical_groups))
bottom = np.zeros(len(clinical_groups))
for name in columns:
    fracs = np.array(phenotype_fractions[name])
    group_ax.bar(
        x, fracs, bottom=bottom, width=0.55,
        color={"Habituator": COLOR_HABITUATOR, "No trend": COLOR_NO_TREND,
               "Sensitizer": COLOR_SENSITIZER}[name],
        label=name, edgecolor="white", linewidth=1.5,
    )
    bottom += fracs

group_ax.set_xticks(x)
group_ax.set_xticklabels(clinical_groups, fontsize=10)
group_ax.set_ylabel("Proportion of group", fontsize=9)
group_ax.set_ylim(0, 1.08)
group_ax.spines[["top", "right"]].set_visible(False)
group_ax.tick_params(labelsize=8)
group_ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
    frameon=False, fontsize=9,
)
group_ax.text(
    0.98, 1.02, GROUP_STAT_LABEL, transform=group_ax.transAxes,
    ha="right", va="bottom", fontsize=8, color="0.4",
)
group_ax.set_title(
    "Sensitizer phenotype becomes more common with higher clinical pain",
    fontsize=11, fontweight="medium", pad=10,
)

fig.suptitle(
    "Hold-trial phenotype predicts offset analgesia magnitude",
    fontsize=14, fontweight="medium", y=0.995,
)
fig.text(
    0.5, 0.655,
    "Deeper trough relative to baseline (dashed) = greater offset analgesia",
    ha="center", fontsize=9, color="0.4",
)

fig.savefig(f"{FIGPATH}/graphical_takeaway_habituator_sensitizer.svg", dpi=300, bbox_inches="tight")
plt.show()
# %%
