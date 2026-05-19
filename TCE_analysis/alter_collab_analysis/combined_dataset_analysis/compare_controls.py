#%% 
import json
import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
FIGPATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures'
base_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data'
sql_path = f'{base_path}/combined_data.sqlite'

datasets = ['kneeOA', 'plosONE']
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

# Get kneeOA labels
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

# Create a single source column
all_trial_metrics['control_source'] = all_trial_metrics['dataset'].map({
    'plosONE': 'plosONE',
    'kneeOA': 'kneeOA_control'
})

# Add group labels to kneeOA, plosONE is all control
all_trial_metrics = all_trial_metrics.merge(
    kneeoa_groups[['subject', 'group_label']],
    on='subject',
    how='left'
)

all_trial_metrics.loc[all_trial_metrics['dataset'] == 'plosONE', 'group_label'] = 'Control'
all_trial_metrics['group_label'] = all_trial_metrics['group_label'].replace({'control': 'Control'})

# Keep only controls
df = all_trial_metrics[all_trial_metrics['group_label'] == 'Control'].copy()

# Standardize trial labels
df['trial_type'] = df['trial_type'].replace('inv', 'onset')
df = df[df['trial_type'].isin(['onset', 'offset', 't1_hold', 't2_hold'])].copy()

print(df['control_source'].value_counts())
print(df['trial_type'].value_counts())

# Add preceding variables
def get_preceding_value(row, col, data):
    prev = data[(data['subject'] == row['subject']) & (data['trial_num'] == row['trial_num'] - 1)]
    return prev.iloc[0][col] if not prev.empty else np.nan

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

trajectory_df = pd.read_csv(
    f'{base_path}/trajectory_classifications.csv'
)
df = df.merge(
    trajectory_df[['subject', 'trajectory_classification']],
    on='subject',
    how='left'
)
#%% Are there differences in OA and OH across datasets?
# LME of onset trials 
onset_df = contrast_df[contrast_df['trial_type'] == 'onset'].copy()

model_onset = smf.mixedlm(
    "abs_normalized_pain_change ~ C(control_source)",
    data=onset_df,
    groups=onset_df["subject"]
)

result_onset = model_onset.fit(reml=False)
print("Onset Trials LME Results:")
print(result_onset.summary())

# LME of offset trials
offset_df = contrast_df[contrast_df['trial_type'] == 'offset'].copy()

model_offset = smf.mixedlm(
    "abs_normalized_pain_change ~ C(control_source)",
    data=offset_df,
    groups=offset_df["subject"]
)

result_offset = model_offset.fit(reml=False)
print("Offset Trials LME Results:")
print(result_offset.summary())

# Plot with stat annotations
DATASET_COLORS = {
    'plosONE': 'blue',
    'kneeOA_control': 'orange'
}

plt.figure(figsize=(10, 7))
trial_order = ['onset', 'offset']
ax = sns.violinplot(
    data=contrast_df,
    x='trial_type',
    y='abs_normalized_pain_change',
    hue='control_source',
    palette=DATASET_COLORS,
    inner='box',
    order=trial_order
)
summary_stats = contrast_df.groupby(['trial_type', 'control_source']).agg(
    n_trials=('abs_normalized_pain_change', 'count'),
    n_subjects=('subject', 'nunique'),
    mean=('abs_normalized_pain_change', 'mean')
).reset_index()

# Positions for annotations
x_positions = {
    ('onset', 'kneeOA_control'): -0.2,
    ('onset', 'plosONE'): 0.2,
    ('offset', 'kneeOA_control'): 0.8,
    ('offset', 'plosONE'): 1.2
}

# Add text labels
for _, row in summary_stats.iterrows():
    x = x_positions[(row['trial_type'], row['control_source'])]
    plt.text(
        x=x,
        y=plt.ylim()[0],
        s=f"n subj = {row['n_subjects']}\n"
          f"n trials = {row['n_trials']}",
        ha='center',
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

# onset stats
onset_p = result_onset.pvalues['C(control_source)[T.plosONE]']
# offset stats
offset_p = result_offset.pvalues['C(control_source)[T.plosONE]']
# significance formatting
def format_p(p):
    if p < 0.001:
        return 'p < 0.001 ***'
    elif p < 0.01:
        return f'p = {p:.3f} **'
    elif p < 0.05:
        return f'p = {p:.3f} *'
    else:
        return f'p = {p:.3f} ns'

# Add brackets + p-values
sig_y = 105
sig_h = 5

# ONSET
plt.plot(
    [-0.2, -0.2, 0.2, 0.2],
    [sig_y, sig_y + sig_h, sig_y + sig_h, sig_y],
    color='black'
)

plt.text(
    0,
    sig_y + sig_h + 2,
    format_p(onset_p),
    ha='center',
    fontsize=11,
    fontweight='bold'
)

# OFFSET
plt.plot(
    [0.8, 0.8, 1.2, 1.2],
    [sig_y, sig_y + sig_h, sig_y + sig_h, sig_y],
    color='black'
)

plt.text(
    1,
    sig_y + sig_h + 2,
    format_p(offset_p),
    ha='center',
    fontsize=11,
    fontweight='bold'
)

plt.ylim(-150, 135)
plt.title(
    'Temporal Contrast Effects by Control Group',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Trial Type', fontsize=12, fontweight='bold')
plt.ylabel('Normalized Pain Change (%)', fontsize=12, fontweight='bold')
plt.legend(title='Study')
plt.tight_layout()
plt.savefig(
    f'{FIGPATH}/control_source_temporal_contrast_stats.png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
#%% Are there differences in preceding trial pain change vs. current pain change across datasets?
from statsmodels.stats.multitest import multipletests

analyses = [
    ('onset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('onset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'negative', 'Negative Normalized Change'),
    ('offset', 'preceding_abs_normalized_pain_change', 'positive', 'Positive Normalized Change'),
]

DATASET_COLORS = {
    'plosONE': 'blue',
    'kneeOA_control': 'orange'
}

print("Creating plots and collecting correlations...")
all_correlations = []

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

for idx, (trial_type, preceding_metric, direction, metric_label) in enumerate(analyses):

    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Y-axis formatting EXACTLY like original
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
        plot_data = base_data[
            base_data[preceding_metric] > 0
        ]
        ax.set_xlim(0, 101)

    elif direction == 'negative':
        plot_data = base_data[
            base_data[preceding_metric] < 0
        ]
        ax.set_xlim(0, -101)

    else:
        plot_data = base_data

    # Title
    total_n = len(plot_data)

    ax.set_title(
        f'{trial_type.title()} - {metric_label}\n(N={total_n})',
        fontweight='bold',
        fontsize=10
    )

    # Plotting
    if len(plot_data) > 10:

        text_y_positions = [0.95, 0.85]

        for group_idx, source in enumerate(['plosONE', 'kneeOA_control']):

            group_data = plot_data[
                plot_data['control_source'] == source
            ]

            if len(group_data) > 3:

                # SCATTER
                ax.scatter(
                    group_data[preceding_metric],
                    group_data['abs_normalized_pain_change'],
                    color=DATASET_COLORS[source],
                    alpha=0.6,
                    label=f'{source}',
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )

                # CORRELATION
                r, p = stats.pearsonr(
                    group_data[preceding_metric],
                    group_data['abs_normalized_pain_change']
                )

                # STORE RESULTS
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

    # Formatting
    ax.set_xlabel(
        preceding_metric
        .replace("preceding_abs_", "")
        .replace("_", " ")
        .title()
    )

    ax.set_ylabel('Current Normalized Pain Change (%)')

    ax.legend(fontsize=8, loc='upper right')

    ax.grid(True, alpha=0.3)

# ========================================================
# FDR CORRECTION + SIGNIFICANT LINES ONLY
# ========================================================

if all_correlations:

    p_values = [corr['p_raw'] for corr in all_correlations]

    rejected, p_corrected, _, _ = multipletests(
        p_values,
        method='fdr_bh',
        alpha=0.05
    )

    for i, corr in enumerate(all_correlations):

        corr['p_corrected'] = p_corrected[i]
        corr['significant'] = rejected[i]

        # ONLY PLOT LINE IF SIGNIFICANT
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
                color=DATASET_COLORS[corr['group']],
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

        text_y_positions = [0.95, 0.85]

        corr['ax'].text(
            0.05,
            text_y_positions[corr['group_idx']],
            f'{corr["group"]}: r={corr["r"]:.2f}, '
            f'p={corr["p_corrected"]:.3f} {sig_marker}',
            transform=corr['ax'].transAxes,
            fontsize=8,
            color=DATASET_COLORS[corr['group']],
            fontweight='bold',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8
            )
        )

    print(f"Multiple comparisons correction applied to {len(all_correlations)} tests")

    significant_count = sum(
        corr['significant'] for corr in all_correlations
    )

    print(f"Significant after FDR correction: {significant_count}/{len(all_correlations)}")

plt.suptitle(
    'Trial Sequence Effects by Control Source - Split by Direction (FDR Corrected)',
    fontsize=16,
    fontweight='bold'
)
plt.tight_layout()
plt.show()

# %% Are there significant differences in trajectory classification?
# Subject-level dataframe only
subject_df = df.drop_duplicates('subject').copy()

# Create contingency table
contingency_table = pd.crosstab(
    subject_df['trajectory_classification'],
    subject_df['control_source']
)
print("Contingency Table:")
print(contingency_table)

# Run chi-squared test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-Squared Results:")
print(f"chi2 = {chi2:.3f}")
print(f"p = {p:.4f}")
print(f"dof = {dof}")

# Expected counts
expected_df = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)
print("\nExpected Counts:")
print(expected_df)
plt.figure(figsize=(8,6))
sns.heatmap(
    contingency_table,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False
)
plt.title(
    'Trajectory Classification by Control Dataset',
    fontsize=14,
    fontweight='bold'
)
plt.xlabel('Control Source')
plt.ylabel('Trajectory Classification')
plt.tight_layout()
plt.show()
# %%
