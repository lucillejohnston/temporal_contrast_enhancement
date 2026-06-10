#%% 
import json
import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

FIGPATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures'
base_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data'
sql_path = f'{base_path}/combined_data.sqlite'

datasets = ['kneeOA', 'cLBP']
combined = []

for dataset in datasets:
    trial_metrics_path = f'{base_path}/{dataset}_trial_metrics.json'
    with open(trial_metrics_path, 'r') as f:
        metrics_data = json.load(f)

    records = []
    for subject_id, trials in metrics_data.items():
        for trial_num, trial_data in trials.items():
            records.append({
                'dataset': dataset,
                'subject': int(subject_id),
                'trial_num': int(trial_num),
                **trial_data
            })

    df = pd.DataFrame(records)
    combined.append(df)

all_trial_metrics = pd.concat(combined, ignore_index=True)

# Fix ID overlap
all_trial_metrics.loc[all_trial_metrics['dataset'] == 'kneeOA', 'subject'] += 1000
all_trial_metrics.loc[all_trial_metrics['dataset'] == 'cLBP', 'subject'] += 2000

# Get kneeOA group labels
conn = sqlite3.connect(sql_path)
kneeoa_groups = pd.read_sql_query("""
SELECT DISTINCT
    subject,
    COALESCE(NULLIF("group", ""), 'control') AS group_label
FROM metadata
WHERE study = 'kneeOA'
ORDER BY subject
""", conn)
conn.close()

kneeoa_groups['subject'] += 1000
kneeoa_groups['dataset'] = 'kneeOA'

# Get cLBP group labels (all High)
conn = sqlite3.connect(sql_path)
clbp_groups = pd.read_sql_query("""
SELECT DISTINCT
    subject,
    COALESCE(NULLIF("group", ''), 'High') AS group_label
FROM metadata
WHERE study LIKE 'cLBP%'
ORDER BY subject
""", conn)
conn.close()

clbp_groups['subject'] += 2000
clbp_groups['dataset'] = 'cLBP'

# Combine group labels
all_groups = pd.concat([kneeoa_groups, clbp_groups], ignore_index=True)

# Merge with trial metrics
all_trial_metrics = all_trial_metrics.merge(
    all_groups[['subject', 'group_label', 'dataset']],
    on=['subject', 'dataset'],
    how='left'
)

# Standardize group labels
all_trial_metrics['group_label'] = all_trial_metrics['group_label'].replace({
    'control': 'Control',
    'low_pain': 'Low',
    'high_pain': 'High'
})

# Keep only High and Low pain groups (exclude controls)
df = all_trial_metrics[all_trial_metrics['group_label'].isin(['High', 'Low'])].copy()

# Create pain_group_source column: combines dataset + group_label
df['pain_group_source'] = df['dataset'] + '_' + df['group_label']

print("\n=== PAIN GROUP SOURCES ===")
print(df['pain_group_source'].value_counts())
print(f"\nSubjects per group:")
print(df.groupby('pain_group_source')['subject'].nunique())

# Standardize trial labels
df['trial_type'] = df['trial_type'].replace('inv', 'onset')
df = df[df['trial_type'].isin(['onset', 'offset', 't1_hold', 't2_hold'])].copy()

print("\n=== TRIAL TYPE DISTRIBUTION ===")
print(df['trial_type'].value_counts())

# Add preceding variables
def get_preceding_value(row, col, data):
    prev = data[(data['subject'] == row['subject']) & (data['trial_num'] == row['trial_num'] - 1)]
    return prev.iloc[0][col] if not prev.empty else np.nan

print("\nCalculating preceding trial metrics...")
for new_col, source_col in {
    'preceding_trial_type': 'trial_type',
    'preceding_abs_max_val': 'abs_max_val',
    'preceding_abs_min_val': 'abs_min_val',
    'preceding_abs_peak_to_peak': 'abs_peak_to_peak',
    'preceding_auc_total': 'auc_total',
    'preceding_abs_normalized_pain_change': 'abs_normalized_pain_change'
}.items():
    df[new_col] = df.apply(lambda row: get_preceding_value(row, source_col, df), axis=1)

contrast_df = df[df['trial_type'].isin(['onset', 'offset'])].copy()

# Load trajectory classifications
trajectory_df = pd.read_csv(f'{base_path}/holdslope_classifications.csv')
df = df.merge(
    trajectory_df[['subject', 'classification']],
    on='subject',
    how='left'
)

print("\nData preparation complete!")
print(f"Total trials: {len(df)}")
print(f"Total subjects: {df['subject'].nunique()}")

#%% Are there differences in OA and OH across pain group sources?

# Define colors for each pain group source
PAIN_GROUP_COLORS = {
    'kneeOA_High': '#8B0000',    # Dark red
    'kneeOA_Low': '#FFA500',     # Orange
    'cLBP_High': '#DC143C',      # Crimson
    'cLBP_Low': '#FF8C00'        # Dark orange
}

# LME of onset trials 
print("\n" + "="*60)
print("LME: ONSET TRIALS BY PAIN GROUP SOURCE")
print("="*60)

onset_df = contrast_df[contrast_df['trial_type'] == 'onset'].copy()

model_onset = smf.mixedlm(
    "abs_normalized_pain_change ~ C(pain_group_source)",
    data=onset_df,
    groups=onset_df["subject"]
)

result_onset = model_onset.fit(reml=False)
print(result_onset.summary())

# LME of offset trials
print("\n" + "="*60)
print("LME: OFFSET TRIALS BY PAIN GROUP SOURCE")
print("="*60)

offset_df = contrast_df[contrast_df['trial_type'] == 'offset'].copy()

model_offset = smf.mixedlm(
    "abs_normalized_pain_change ~ C(pain_group_source)",
    data=offset_df,
    groups=offset_df["subject"]
)

result_offset = model_offset.fit(reml=False)
print(result_offset.summary())

# Plot with stat annotations
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

trial_order = ['onset', 'offset']
group_order = ['kneeOA_Low', 'kneeOA_High', 'cLBP_Low', 'cLBP_High']

for ax_idx, trial_type in enumerate(trial_order):
    ax = axes[ax_idx]
    
    trial_data = contrast_df[contrast_df['trial_type'] == trial_type]
    
    sns.violinplot(
        data=trial_data,
        x='pain_group_source',
        y='abs_normalized_pain_change',
        palette=PAIN_GROUP_COLORS,
        inner='box',
        order=group_order,
        ax=ax
    )
    
    # Add sample sizes
    summary_stats = trial_data.groupby('pain_group_source').agg(
        n_trials=('abs_normalized_pain_change', 'count'),
        n_subjects=('subject', 'nunique'),
        mean=('abs_normalized_pain_change', 'mean')
    ).reindex(group_order)
    
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    
    for i, group in enumerate(group_order):
        if group in summary_stats.index:
            stats_row = summary_stats.loc[group]
            ax.text(
                x=i,
                y=y_min + 0.03 * y_span,
                s=f"n subj = {int(stats_row['n_subjects'])}\n"
                  f"n trials = {int(stats_row['n_trials'])}",
                ha='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    ax.set_title(f'{trial_type.title()} Trials', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pain Group Source', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Pain Change (%)' if ax_idx == 0 else '', 
                  fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    'Temporal Contrast Effects by Pain Group Source',
    fontsize=15,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{FIGPATH}/pain_group_source_temporal_contrast_comparison.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

#%% Pairwise comparisons with FDR correction

print("\n" + "="*60)
print("PAIRWISE COMPARISONS (Welch t-tests + FDR correction)")
print("="*60)

from itertools import combinations

for trial_type in ['onset', 'offset']:
    print(f"\n{trial_type.upper()} TRIALS:")
    trial_data = contrast_df[contrast_df['trial_type'] == trial_type]
    
    # Get data for each group
    group_data = {}
    for group in group_order:
        group_data[group] = trial_data[
            trial_data['pain_group_source'] == group
        ]['abs_normalized_pain_change'].dropna()
    
    # Pairwise tests
    comparisons = []
    raw_p = []
    
    for g1, g2 in combinations(group_order, 2):
        if len(group_data[g1]) > 0 and len(group_data[g2]) > 0:
            t_stat, p_val = stats.ttest_ind(
                group_data[g1], 
                group_data[g2], 
                equal_var=False, 
                nan_policy='omit'
            )
            comparisons.append((g1, g2, t_stat, p_val))
            raw_p.append(p_val)
    
    # FDR correction
    if raw_p:
        reject, p_fdr, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')
        
        for (g1, g2, t_stat, p_val), keep, p_corr in zip(comparisons, reject, p_fdr):
            sig = '***' if keep else 'ns'
            print(f"  {g1} vs {g2}: t={t_stat:.3f}, p_raw={p_val:.4f}, p_FDR={p_corr:.4f} {sig}")

#%% Are there differences in preceding trial effects across pain group sources?

analyses = [
    ('onset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('onset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
]

print("\n" + "="*60)
print("TRIAL SEQUENCE EFFECTS BY PAIN GROUP SOURCE")
print("="*60)

all_correlations = []
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

for idx, (trial_type, preceding_metric, direction, metric_label) in enumerate(analyses):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Y-axis formatting
    if trial_type == 'onset':
        ax.set_ylim(0, 101)
    else:
        ax.set_ylim(0, -101)

    # Base filtering
    base_data = contrast_df[
        (contrast_df['trial_type'] == trial_type) &
        (contrast_df[preceding_metric].notna()) &
        (contrast_df['abs_normalized_pain_change'].notna())
    ]

    # Direction filtering
    if direction == 'positive':
        plot_data = base_data[base_data[preceding_metric] > 0]
        ax.set_xlim(0, 101)
    elif direction == 'negative':
        plot_data = base_data[base_data[preceding_metric] < 0]
        ax.set_xlim(0, -101)
    else:
        plot_data = base_data

    total_n = len(plot_data)
    ax.set_title(
        f'{trial_type.title()} - {metric_label}\n(N={total_n})',
        fontweight='bold',
        fontsize=10
    )

    if len(plot_data) > 10:
        text_y_positions = [0.95, 0.88, 0.81, 0.74]

        for group_idx, source in enumerate(group_order):
            group_data = plot_data[plot_data['pain_group_source'] == source]

            if len(group_data) > 3:
                # Scatter
                ax.scatter(
                    group_data[preceding_metric],
                    group_data['abs_normalized_pain_change'],
                    color=PAIN_GROUP_COLORS[source],
                    alpha=0.6,
                    label=f'{source} (n={len(group_data)})',
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )

                # Correlation
                r, p = stats.pearsonr(
                    group_data[preceding_metric],
                    group_data['abs_normalized_pain_change']
                )

                all_correlations.append({
                    'idx': idx,
                    'trial_type': trial_type,
                    'metric': preceding_metric,
                    'direction': direction,
                    'metric_label': metric_label,
                    'group': source,
                    'r': r,
                    'p_raw': p,
                    'n': len(group_data),
                    'ax': ax,
                    'group_idx': group_idx,
                    'group_data': group_data
                })

    ax.set_xlabel(
        preceding_metric.replace("preceding_abs_", "").replace("_", " ").title()
    )
    ax.set_ylabel('Current Normalized Pain Change (%)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

# FDR correction
if all_correlations:
    p_values = [corr['p_raw'] for corr in all_correlations]
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)

    for i, corr in enumerate(all_correlations):
        corr['p_corrected'] = p_corrected[i]
        corr['significant'] = rejected[i]

        # Plot regression line if significant
        if corr['significant']:
            group_data = corr['group_data']
            z = np.polyfit(
                group_data[corr['metric']],
                group_data['abs_normalized_pain_change'],
                1
            )
            p_fit = np.poly1d(z)
            x_range = np.linspace(
                group_data[corr['metric']].min(),
                group_data[corr['metric']].max(),
                100
            )
            corr['ax'].plot(
                x_range,
                p_fit(x_range),
                color=PAIN_GROUP_COLORS[corr['group']],
                linestyle='--',
                linewidth=2,
                alpha=0.8
            )

        # Significance marker
        sig_marker = (
            "***" if corr['p_corrected'] < 0.001 else
            "**" if corr['p_corrected'] < 0.01 else
            "*" if corr['p_corrected'] < 0.05 else
            "ns"
        )

        text_y_positions = [0.95, 0.88, 0.81, 0.74]
        corr['ax'].text(
            0.05,
            text_y_positions[corr['group_idx']],
            f'{corr["group"]}: r={corr["r"]:.2f}, p={corr["p_corrected"]:.3f} {sig_marker}',
            transform=corr['ax'].transAxes,
            fontsize=7,
            color=PAIN_GROUP_COLORS[corr['group']],
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    print(f"\nMultiple comparisons correction: {len(all_correlations)} tests")
    significant_count = sum(corr['significant'] for corr in all_correlations)
    print(f"Significant after FDR: {significant_count}/{len(all_correlations)}")

plt.suptitle(
    'Trial Sequence Effects by Pain Group Source (FDR Corrected)',
    fontsize=16,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    f'{FIGPATH}/pain_group_sequence_effects.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

#%% Are there significant differences in trajectory classification across pain groups?

# Subject-level dataframe only
subject_df = df.drop_duplicates('subject').copy()

# Create contingency table
contingency_table = pd.crosstab(
    subject_df['classification'],
    subject_df['pain_group_source']
)

print("\n" + "="*60)
print("TRAJECTORY CLASSIFICATION BY PAIN GROUP SOURCE")
print("="*60)
print("\nContingency Table:")
print(contingency_table)

# Run chi-squared test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\nChi-Squared Results:")
print(f"  chi2 = {chi2:.3f}")
print(f"  p-value = {p:.4f}")
print(f"  dof = {dof}")

if p < 0.05:
    print("  Result: Significant association between trajectory and pain group source")
else:
    print("  Result: No significant association")

print("\nExpected Counts:")
expected_df = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)
print(expected_df)

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(
    contingency_table,
    annot=True,
    fmt='d',
    cmap='Reds',
    cbar=False
)
plt.title(
    f'Trajectory Classification by Pain Group Source\n(χ²={chi2:.2f}, p={p:.4f})',
    fontsize=14,
    fontweight='bold'
)
plt.xlabel('Pain Group Source', fontweight='bold')
plt.ylabel('Trajectory Classification', fontweight='bold')
plt.tight_layout()
plt.savefig(
    f'{FIGPATH}/pain_group_trajectory_contingency.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# %%