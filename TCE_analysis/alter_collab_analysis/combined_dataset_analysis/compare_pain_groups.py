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

FIGPATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/cross-dataset_comparisons/pains'
base_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data'
sql_path = f'{base_path}/combined_data.sqlite'

datasets = ['kneeOA', 'cLBP'] # add sEEG as soon as possible
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
    "group" AS group_label
FROM metadata
WHERE study = 'kneeOA'
ORDER BY subject
""", conn)
conn.close()

kneeoa_groups['subject'] += 1000
kneeoa_groups['dataset'] = 'kneeOA'

# Get cLBP group labels
conn = sqlite3.connect(sql_path)
clbp_groups = pd.read_sql_query("""
SELECT DISTINCT
    subject,
    "group" AS group_label
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
classification_df = pd.read_csv(f'{base_path}/holdslope_classifications.csv')
df = df.merge(
    classification_df[['subject', 'classification']],
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

onset_df = contrast_df[contrast_df['trial_type'] == 'onset'].copy().dropna(subset=['abs_normalized_pain_change', 'pain_group_source', 'subject']).reset_index(drop=True)

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

offset_df = contrast_df[contrast_df['trial_type'] == 'offset'].copy().dropna(subset=['abs_normalized_pain_change', 'pain_group_source', 'subject']).reset_index(drop=True)

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
# If significant, follow up with post-hoc tests 
if p < 0.05:
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests

    pairs = list(combinations(group_order, 2))
    raw_p = []
    pair_results = []

    for g1, g2 in pairs:
        # Pull just those two columns from the contingency table
        sub_table = contingency_table[[g1, g2]]
        # Drop rows where both are 0 (avoids degenerate chi2)
        sub_table = sub_table[sub_table.sum(axis=1) > 0]
        chi2_pair, p_pair, _, _ = stats.chi2_contingency(sub_table)
        raw_p.append(p_pair)
        pair_results.append((g1, g2, chi2_pair, p_pair))

    reject, p_fdr, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')

    print("\nPairwise Chi-Squared Post-Hoc (FDR corrected):")
    for (g1, g2, chi2_pair, p_raw), keep, p_corr in zip(pair_results, reject, p_fdr):
        sig = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
        print(f"  {g1} vs {g2}: chi2={chi2_pair:.3f}, p_raw={p_raw:.4f}, p_FDR={p_corr:.4f} {sig}")
#%% Systematic dataset comparison - basic statistics
from scipy.stats import mannwhitneyu, kruskal

SOURCE_COL = 'pain_group_source'
COLORS = PAIN_GROUP_COLORS
all_trial_types = ['onset', 'offset', 't1_hold', 't2_hold']
# cross-dataset same-pain-level pairs for systematic dataset effect
_cross_pairs = [('kneeOA_High', 'cLBP_High'), ('kneeOA_Low', 'cLBP_Low')]
# positions in group_order: kneeOA_Low=0, kneeOA_High=1, cLBP_Low=2, cLBP_High=3
_cross_pair_xpos = [(1, 3), (0, 2)]

# ── helpers ───────────────────────────────────────────────────────────────────
def _sig_bracket_pg(ax, x0, x1, y_top, p, h=None):
    h = (abs(y_top) * 0.06 + 2) if h is None else h
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else None
    if star:
        ax.plot([x0, x0, x1, x1], [y_top + h*0.2, y_top + h, y_top + h, y_top + h*0.2],
                'k-', lw=1.2)
        ax.text((x0+x1)/2, y_top + h*1.3, star, ha='center', va='bottom',
                fontsize=11, fontweight='bold')

def _sig_text_pg(ax, p, x=0.5, y=0.97):
    star = '***' if p < 0.001 else f'p={p:.3f} **' if p < 0.01 else f'p={p:.3f} *' if p < 0.05 else None
    if star:
        ax.text(x, y, star, transform=ax.transAxes, ha='center', va='top',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

def _get_curves_pg(ts_df, trial_type, subject_ids, time_grid, col='pain', t2_temps=None):
    subset = ts_df[(ts_df['trial_type'] == trial_type) & (ts_df['subject'].isin(subject_ids))]
    curves = []
    for (subj, _), grp in subset.groupby(['subject', 'trial_num']):
        grp = grp.sort_values('aligned_time')
        t = grp['aligned_time'].values
        v = pd.to_numeric(grp[col], errors='coerce').values
        valid = ~np.isnan(v)
        if valid.sum() < 5:
            continue
        interp = np.interp(time_grid, t[valid], v[valid])
        if t2_temps is not None:
            t2 = t2_temps.get(int(subj))
            if t2 is None:
                continue
            interp = interp - t2
        curves.append(interp)
    if not curves:
        return None, None, 0
    arr = np.array(curves)
    return np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1) / np.sqrt(len(curves)), len(curves)

# ── load time series data ─────────────────────────────────────────────────────
_ts_parts_pg = []
for _ds, _id_offset in [('kneeOA', 1000), ('cLBP', 2000)]:
    with open(f'{base_path}/{_ds}_trial_data_cleaned_aligned.json') as _f:
        _tdf = pd.DataFrame(json.load(_f))
    _tdf['subject'] = _tdf['subject'].astype(int) + _id_offset
    _tdf['dataset'] = _ds
    # Shift cLBP forward 5s to align with kneeOA (different recording start reference)
    if _ds == 'cLBP':
        _tdf['aligned_time'] = _tdf['aligned_time'] + 4.5
    _ts_parts_pg.append(_tdf)
ts_all_pg = pd.concat(_ts_parts_pg, ignore_index=True)
ts_all_pg['pain'] = pd.to_numeric(ts_all_pg['pain'], errors='coerce')
ts_all_pg['temperature'] = pd.to_numeric(ts_all_pg['temperature'], errors='coerce')

_t2_temps_pg = {}
for _s in ts_all_pg['subject'].unique():
    _t = ts_all_pg[(ts_all_pg['subject'] == _s) & (ts_all_pg['trial_type'] == 'offset')]['temperature'].dropna()
    if not _t.empty:
        _t2_temps_pg[int(_s)] = float(_t.max())

_time_grid_pg = np.arange(10, 40, 0.1)

_TS_COLORS_PG = {
    ('offset', 'kneeOA_High'): '#8B0000',
    ('offset', 'kneeOA_Low'): '#CD5C5C',
    ('offset', 'cLBP_High'): '#DC143C',
    ('offset', 'cLBP_Low'): '#FFB6C1',
    ('t1_hold', 'kneeOA_High'): '#00008B',
    ('t1_hold', 'kneeOA_Low'): '#4169E1',
    ('t1_hold', 'cLBP_High'): '#1E90FF',
    ('t1_hold', 'cLBP_Low'): '#87CEEB',
    ('onset', 'kneeOA_High'): '#006400',
    ('onset', 'kneeOA_Low'): '#228B22',
    ('onset', 'cLBP_High'): '#32CD32',
    ('onset', 'cLBP_Low'): '#90EE90',
    ('t2_hold', 'kneeOA_High'): '#8B4513',
    ('t2_hold', 'kneeOA_Low'): '#D2691E',
    ('t2_hold', 'cLBP_High'): '#CD853F',
    ('t2_hold', 'cLBP_Low'): '#DEB887',
}

# ── 1. Time series: pain + temp curves ───────────────────────────────────────
for hold_type, stepped_type in [('t1_hold', 'offset'), ('t2_hold', 'onset')]:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={'height_ratios': [1, 3]})
    for trial_type in [hold_type, stepped_type]:
        for source in group_order:
            subj_ids = df[df[SOURCE_COL] == source]['subject'].unique()
            color = _TS_COLORS_PG.get((trial_type, source), 'gray')
            from scipy.ndimage import gaussian_filter1d
            mt, st, nt = _get_curves_pg(ts_all_pg, trial_type, subj_ids, _time_grid_pg, 'temperature', _t2_temps_pg)
            if mt is not None:
                mt_s = gaussian_filter1d(mt, sigma=5)
                axes[0].plot(_time_grid_pg, mt_s, color=color, lw=2, label=f'{trial_type} ({source}, n={nt})')
                axes[0].fill_between(_time_grid_pg, mt_s - st, mt_s + st, color=color, alpha=0.15)
            mp, sp, np_ = _get_curves_pg(ts_all_pg, trial_type, subj_ids, _time_grid_pg, 'pain')
            if mp is not None:
                mp_s = gaussian_filter1d(mp, sigma=5)
                axes[1].plot(_time_grid_pg, mp_s, color=color, lw=2, label=f'{trial_type} ({source}, n={np_})')
                axes[1].fill_between(_time_grid_pg, mp_s - 1.96*sp, mp_s + 1.96*sp, color=color, alpha=0.15)

    axes[0].set_ylabel('Temp. rel. to T2 (°C)', fontsize=11)
    axes[0].set_ylim(-2, 1)
    axes[0].legend(fontsize=7, loc='lower right', ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{hold_type} vs {stepped_type}: Temperature', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Aligned Time (s)', fontsize=11)
    axes[1].set_ylabel('Pain Rating', fontsize=11)
    axes[1].set_xlim(10, 40)
    axes[1].set_ylim(0, 80)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'{hold_type} vs {stepped_type}: Pain', fontsize=12, fontweight='bold')
    plt.suptitle(f'Cross-Dataset: {hold_type} vs {stepped_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGPATH}/pain_groups_timeseries_{hold_type}_vs_{stepped_type}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ── 2. AUC Total distributions + significance ─────────────────────────────────
fig, axes = plt.subplots(1, len(all_trial_types), figsize=(20, 5), sharey=True)
for ax_idx, trial_type in enumerate(all_trial_types):
    ax = axes[ax_idx]
    subset = df[df['trial_type'] == trial_type]
    sns.histplot(data=subset, x='auc_total', hue=SOURCE_COL,
                 palette=COLORS, multiple='layer', bins=25, kde=True, alpha=0.5, ax=ax)
    ax.set_title(trial_type.title())
    ax.set_xlabel('AUC Total')
    if ax_idx > 0:
        ax.set_ylabel('')
    grp = [subset[subset[SOURCE_COL] == s]['auc_total'].dropna() for s in group_order if len(subset[subset[SOURCE_COL] == s]) > 3]
    if len(grp) >= 2:
        _, p = kruskal(*grp)
        _sig_text_pg(ax, p)
plt.suptitle('AUC Total Distribution by Pain Group Source', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_auc_total.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 3. Abs Max Val distributions + significance ───────────────────────────────
fig, axes = plt.subplots(1, len(all_trial_types), figsize=(20, 5), sharey=True)
for ax_idx, trial_type in enumerate(all_trial_types):
    ax = axes[ax_idx]
    subset = df[df['trial_type'] == trial_type]
    sns.histplot(data=subset, x='abs_max_val', hue=SOURCE_COL,
                 palette=COLORS, multiple='layer', bins=25, kde=True, alpha=0.5, ax=ax)
    ax.set_title(trial_type.title())
    ax.set_xlabel('Absolute Max Value')
    if ax_idx > 0:
        ax.set_ylabel('')
    grp = [subset[subset[SOURCE_COL] == s]['abs_max_val'].dropna() for s in group_order if len(subset[subset[SOURCE_COL] == s]) > 3]
    if len(grp) >= 2:
        _, p = kruskal(*grp)
        _sig_text_pg(ax, p)
plt.suptitle('Absolute Max Value Distribution by Pain Group Source', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_abs_max_val.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 4. AUC Total vs Abs Max Val scatter ───────────────────────────────────────
plt.figure(figsize=(10, 7))
for source in group_order:
    subset = df[df[SOURCE_COL] == source].dropna(subset=['auc_total', 'abs_max_val'])
    plt.scatter(subset['auc_total'], subset['abs_max_val'],
                color=COLORS[source], alpha=0.4, label=source, s=30)
plt.xlabel('AUC Total')
plt.ylabel('Absolute Max Value')
plt.title('AUC Total vs Absolute Max Value by Pain Group Source')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_auc_vs_maxval.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 5. Time-yoked normalized pain change: hold vs stepped ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax_idx, (hold_type, stepped_type) in enumerate([('t1_hold', 'offset'), ('t2_hold', 'onset')]):
    ax = axes[ax_idx]
    records = []
    for source in group_order:
        src_df = df[df[SOURCE_COL] == source]
        hold = src_df[src_df['trial_type'] == hold_type][['subject', 'time_yoked_normalized_pain_change']].copy()
        hold['category'] = f'{hold_type}\n(time-yoked)'
        hold['value'] = hold['time_yoked_normalized_pain_change']
        hold[SOURCE_COL] = source
        stepped = src_df[src_df['trial_type'] == stepped_type][['subject', 'abs_normalized_pain_change']].copy()
        stepped['category'] = stepped_type
        stepped['value'] = stepped['abs_normalized_pain_change']
        stepped[SOURCE_COL] = source
        records += [hold[['subject', 'category', 'value', SOURCE_COL]],
                    stepped[['subject', 'category', 'value', SOURCE_COL]]]
    plot_df = pd.concat(records, ignore_index=True).dropna(subset=['value'])
    x_order = [f'{hold_type}\n(time-yoked)', stepped_type]
    sns.violinplot(data=plot_df, x='category', y='value', hue=SOURCE_COL,
                   palette=COLORS, inner='box', order=x_order, hue_order=group_order, ax=ax)
    # Paired t-test within each pain group source
    y_ann = plot_df['value'].quantile(0.97)
    for s_idx, source in enumerate(group_order):
        h_means = plot_df[(plot_df[SOURCE_COL] == source) & (plot_df['category'] == x_order[0])].groupby('subject')['value'].mean()
        s_means = plot_df[(plot_df[SOURCE_COL] == source) & (plot_df['category'] == x_order[1])].groupby('subject')['value'].mean()
        common = h_means.index.intersection(s_means.index)
        if len(common) > 3:
            _, p = stats.ttest_rel(h_means[common], s_means[common])
            dodge = 0.8 / len(group_order)
            x_off = -0.4 + dodge * (s_idx + 0.5)
            y_line = y_ann + s_idx * 6
            ax.plot([x_off, 1 + x_off], [y_line, y_line], '-', color=COLORS[source], lw=1.2)
            sig = '***' if p < 0.001 else f'**' if p < 0.01 else f'*' if p < 0.05 else 'ns'
            ax.text(0.5 + x_off, y_line + 0.5, sig, ha='center', fontsize=7,
                    fontweight='bold', color=COLORS[source])
    ax.set_title(f'{hold_type} vs {stepped_type}', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Normalized Pain Change (%)' if ax_idx == 0 else '')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-200, 200)
plt.suptitle('Time-Yoked (Hold) vs Absolute (Stepped) Normalized Pain Change by Pain Group',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_timeyoked_vs_abs_pain_change.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 6. Abs normalized pain change: hold vs stepped ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax_idx, (hold_type, stepped_type) in enumerate([('t1_hold', 'offset'), ('t2_hold', 'onset')]):
    ax = axes[ax_idx]
    records = []
    for source in group_order:
        src_df = df[df[SOURCE_COL] == source]
        for tt in [hold_type, stepped_type]:
            d = src_df[src_df['trial_type'] == tt][['subject', 'abs_normalized_pain_change']].copy()
            d['category'] = tt
            d['value'] = d['abs_normalized_pain_change']
            d[SOURCE_COL] = source
            records.append(d[['subject', 'category', 'value', SOURCE_COL]])
    plot_df = pd.concat(records, ignore_index=True).dropna(subset=['value'])
    x_order = [hold_type, stepped_type]
    sns.violinplot(data=plot_df, x='category', y='value', hue=SOURCE_COL,
                   palette=COLORS, inner='box', order=x_order, hue_order=group_order, ax=ax)
    y_ann = plot_df['value'].quantile(0.97)
    for s_idx, source in enumerate(group_order):
        h_means = plot_df[(plot_df[SOURCE_COL] == source) & (plot_df['category'] == x_order[0])].groupby('subject')['value'].mean()
        s_means = plot_df[(plot_df[SOURCE_COL] == source) & (plot_df['category'] == x_order[1])].groupby('subject')['value'].mean()
        common = h_means.index.intersection(s_means.index)
        if len(common) > 3:
            _, p = stats.ttest_rel(h_means[common], s_means[common])
            dodge = 0.8 / len(group_order)
            x_off = -0.4 + dodge * (s_idx + 0.5)
            y_line = y_ann + s_idx * 6
            ax.plot([x_off, 1 + x_off], [y_line, y_line], '-', color=COLORS[source], lw=1.2)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.5 + x_off, y_line + 0.5, sig, ha='center', fontsize=7,
                    fontweight='bold', color=COLORS[source])
    ax.set_title(f'{hold_type} vs {stepped_type}: Abs Pain Change', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Normalized Pain Change (%)' if ax_idx == 0 else '')
    ax.grid(True, alpha=0.3, axis='y')
plt.suptitle('Absolute Normalized Pain Change: Hold vs Stepped by Pain Group',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_abs_pain_change_hold_vs_stepped.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 7. Normalized pain change violins (all trial types) + significance ─────────
# Significance: Kruskal-Wallis overall; brackets for cross-dataset same-pain-level pairs
fig, axes = plt.subplots(1, len(all_trial_types), figsize=(20, 6))
for ax_idx, trial_type in enumerate(all_trial_types):
    ax = axes[ax_idx]
    subset = df[df['trial_type'] == trial_type].dropna(subset=['abs_normalized_pain_change'])
    sns.violinplot(data=subset, x=SOURCE_COL, y='abs_normalized_pain_change',
                   palette=COLORS, inner='box', order=group_order, ax=ax)
    ax.set_title(trial_type.title())
    ax.set_xlabel('')
    ax.set_ylabel('Normalized Pain Change (%)' if ax_idx == 0 else '')
    ax.tick_params(axis='x', rotation=45)
    y_top = subset['abs_normalized_pain_change'].quantile(0.97)
    for (g1, g2), (x1, x2) in zip(_cross_pairs, _cross_pair_xpos):
        d1 = subset[subset[SOURCE_COL] == g1]['abs_normalized_pain_change'].dropna()
        d2 = subset[subset[SOURCE_COL] == g2]['abs_normalized_pain_change'].dropna()
        if len(d1) > 3 and len(d2) > 3:
            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            _sig_bracket_pg(ax, x1, x2, y_top, p)
            y_top = y_top + abs(y_top) * 0.12 + 8
plt.suptitle('Normalized Pain Change by Trial Type and Pain Group Source', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_normalized_pain_change.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 8. Latency to max pain (all trial types) + significance ───────────────────
fig, axes = plt.subplots(1, len(all_trial_types), figsize=(20, 6))
for ax_idx, trial_type in enumerate(all_trial_types):
    ax = axes[ax_idx]
    subset = df[df['trial_type'] == trial_type].dropna(subset=['abs_max_time'])
    sns.violinplot(data=subset, x=SOURCE_COL, y='abs_max_time',
                   palette=COLORS, inner='box', order=group_order, ax=ax)
    ax.set_title(trial_type.title())
    ax.set_xlabel('')
    ax.set_ylabel('Latency to Max Pain (s)' if ax_idx == 0 else '')
    ax.tick_params(axis='x', rotation=45)
    y_top = subset['abs_max_time'].quantile(0.97)
    for (g1, g2), (x1, x2) in zip(_cross_pairs, _cross_pair_xpos):
        d1 = subset[subset[SOURCE_COL] == g1]['abs_max_time'].dropna()
        d2 = subset[subset[SOURCE_COL] == g2]['abs_max_time'].dropna()
        if len(d1) > 3 and len(d2) > 3:
            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            _sig_bracket_pg(ax, x1, x2, y_top, p)
            y_top = y_top + abs(y_top) * 0.12 + 3
plt.suptitle('Latency to Max Pain by Trial Type and Pain Group Source', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_latency.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 9. Peak-to-peak (onset and offset) + significance ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax_idx, trial_type in enumerate(['onset', 'offset']):
    ax = axes[ax_idx]
    subset = df[df['trial_type'] == trial_type].dropna(subset=['abs_peak_to_peak'])
    sns.violinplot(data=subset, x=SOURCE_COL, y='abs_peak_to_peak',
                   palette=COLORS, inner='box', order=group_order, ax=ax)
    ax.set_title(f'{trial_type.title()} Trials')
    ax.set_xlabel('')
    ax.set_ylabel('Abs Peak-to-Peak' if ax_idx == 0 else '')
    ax.tick_params(axis='x', rotation=45)
    y_top = subset['abs_peak_to_peak'].quantile(0.97)
    for (g1, g2), (x1, x2) in zip(_cross_pairs, _cross_pair_xpos):
        d1 = subset[subset[SOURCE_COL] == g1]['abs_peak_to_peak'].dropna()
        d2 = subset[subset[SOURCE_COL] == g2]['abs_peak_to_peak'].dropna()
        if len(d1) > 3 and len(d2) > 3:
            _, p = mannwhitneyu(d1, d2, alternative='two-sided')
            _sig_bracket_pg(ax, x1, x2, y_top, p)
            y_top = y_top + abs(y_top) * 0.12 + 3
plt.suptitle('Peak-to-Peak by Pain Group Source', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_peak_to_peak.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 10. Hold trial normalized pain change + significance ──────────────────────
hold_df = df[df['trial_type'].isin(['t1_hold', 't2_hold'])].dropna(subset=['abs_normalized_pain_change'])
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=hold_df, x='trial_type', y='abs_normalized_pain_change',
               hue=SOURCE_COL, palette=COLORS, hue_order=group_order, inner='box', ax=ax)
ax.set_xlabel('Trial Type')
ax.set_ylabel('Normalized Pain Change (%)')
ax.set_title('Hold Trial Normalized Pain Change by Pain Group Source')
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_hold_trials.png', dpi=300, bbox_inches='tight')
plt.show()

# ── 11. Onset vs Offset magnitude: subject-level ──────────────────────────────
onset_subj = (
    df[df['trial_type'] == 'onset']
    .groupby(['subject', SOURCE_COL])['abs_normalized_pain_change']
    .mean().reset_index()
    .rename(columns={'abs_normalized_pain_change': 'onset_mean'})
)
offset_subj = (
    df[df['trial_type'] == 'offset']
    .groupby(['subject', SOURCE_COL])['abs_normalized_pain_change']
    .mean().reset_index()
    .rename(columns={'abs_normalized_pain_change': 'offset_mean'})
)
prop_df = onset_subj.merge(offset_subj[['subject', 'offset_mean']], on='subject').dropna()
prop_df['onset_abs'] = prop_df['onset_mean'].abs()
prop_df['offset_abs'] = prop_df['offset_mean'].abs()

plt.figure(figsize=(8, 7))
for source in group_order:
    subset = prop_df[prop_df[SOURCE_COL] == source]
    if subset.empty:
        continue
    plt.scatter(subset['offset_abs'], subset['onset_abs'],
                color=COLORS[source], alpha=0.7, label=source,
                s=60, edgecolors='black', linewidth=0.5)
    if len(subset) > 3:
        r, p = stats.pearsonr(subset['offset_abs'], subset['onset_abs'])
        slope, intercept, *_ = stats.linregress(subset['offset_abs'], subset['onset_abs'])
        x_range = np.linspace(subset['offset_abs'].min(), subset['offset_abs'].max(), 100)
        plt.plot(x_range, slope * x_range + intercept, color=COLORS[source],
                 linestyle='--', lw=1.5, label=f'{source}: r={r:.2f}, p={p:.3f}')

lim = max(prop_df['onset_abs'].max(), prop_df['offset_abs'].max()) * 1.05
plt.plot([0, lim], [0, lim], 'k:', alpha=0.4, label='y=x')
plt.xlabel('|Offset Mean Normalized Pain Change| (%)')
plt.ylabel('|Onset Mean Normalized Pain Change| (%)')
plt.title('Onset vs Offset Magnitude: Subject-Level by Pain Group Source')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGPATH}/pain_groups_dataset_comparison_onset_offset_magnitude.png', dpi=300, bbox_inches='tight')
plt.show()
# %%