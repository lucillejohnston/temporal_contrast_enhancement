#%%
"""
Builds a wide-format table of raw pain-intensity values during period C (t3, the 20s hold
epoch) for stepped trials (offset/onset/inv) and their paired constant-control hold trials
(t1_hold/t2_hold), matched via the reference_trial_num field saved during extraction.

Pairing: offset <-> t1_hold, onset/inv <-> t2_hold (via reference_trial_num_offset /
reference_trial_num_onset / reference_trial_num), per the trial_type_info reference scheme
in data_pipeline/extract_metrics.py. stepdown references are not used here.

Each pair produces two rows (one for the stepped trial, one for the hold trial) sharing a
pair_id, with one column per second of period C (t0, t1, ...) holding the raw pain value.
Take the per-second difference between the two rows of a pair afterward as needed.

This script only reads from data/alter_collab_data/*_trial_metrics.json and
*_trial_data_trimmed_downsampled.json (the 1Hz-consistent time series) -- it does not
modify or overwrite any existing files.
"""
import json
import re
import sqlite3
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

base_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data'
sql_path = f'{base_path}/combined_data.sqlite'
FIGPATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures'
OUTPUT_PATH = f'{base_path}/stepped_hold_periodC_table.csv'
VALIDATION_PATH = f'{base_path}/stepped_hold_periodC_validation_problems.csv'
DIFFERENCES_PATH = f'{base_path}/stepped_hold_periodC_differences.csv'

GROUP_COLORS = {
    'Control': '#2E8B57',    # Green
    'Low': '#FF8C00',        # Orange
    'High': '#DC143C'        # Red
}

def p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

datasets = ['kneeOA', 'plosONE', 'cLBP']

# OA = offset - t1_hold (offset analgesia) ; OH = onset/inv - t2_hold (onset hyperalgesia)
PAIR_TYPE_LABELS = {
    'offset_t1_hold': 'OA',
    'onset_t2_hold': 'OH',
    'inv_t2_hold': 'OH',
}

# (stepped trial_type, hold trial_type, reference_trial_num column on the hold trial's metrics row)
PAIR_RULES = {
    'kneeOA':  [('offset', 't1_hold', 'reference_trial_num_offset'),
                ('onset',  't2_hold', 'reference_trial_num_onset')],
    'cLBP':    [('offset', 't1_hold', 'reference_trial_num_offset'),
                ('onset',  't2_hold', 'reference_trial_num_onset')],
    'plosONE': [('offset', 't1_hold', 'reference_trial_num_offset'),
                ('inv',    't2_hold', 'reference_trial_num')],
}

#%%
# ==================================================================================================================
######################################## BUILD PAIRS + VALIDATE UNIQUENESS ########################################
# ==================================================================================================================
all_rows = []
validation_problems = []

for dataset in datasets:
    metrics_path = f'{base_path}/{dataset}_trial_metrics.json'
    data_path = f'{base_path}/{dataset}_trial_data_trimmed_downsampled.json'

    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    metrics_records = []
    for subject_id, trials in metrics_data.items():
        for trial_num, trial_data in trials.items():
            metrics_records.append({'subject': int(subject_id), 'trial_num': int(trial_num), **trial_data})
    metrics_df = pd.DataFrame(metrics_records)

    # 1Hz raw time series has no trial_type column -- attach it via lookup (subject, trial_num) -> trial_type
    trial_data_df = pd.read_json(data_path, orient='records')
    type_lookup = metrics_df[['subject', 'trial_num', 'trial_type']].drop_duplicates()
    trial_data_df = trial_data_df.merge(type_lookup, on=['subject', 'trial_num'], how='left')

    for stepped_type, hold_type, ref_col in PAIR_RULES[dataset]:
        if ref_col not in metrics_df.columns:
            continue
        pair_label = f'{stepped_type}_{hold_type}'
        hold_rows = metrics_df[(metrics_df['trial_type'] == hold_type) & metrics_df[ref_col].notna()].copy()
        hold_rows[ref_col] = hold_rows[ref_col].astype(int)

        # (a) a stepped trial should be referenced by at most one hold trial (one-to-one match)
        dup_mask = hold_rows.duplicated(subset=['subject', ref_col], keep=False)
        for _, r in hold_rows[dup_mask].iterrows():
            validation_problems.append({
                'dataset': dataset, 'pair_type': pair_label, 'issue': 'stepped_trial_referenced_by_multiple_hold_trials',
                'subject': r['subject'], 'hold_trial_num': r['trial_num'], 'stepped_trial_num': r[ref_col],
            })

        # (b) the referenced stepped trial must actually exist for that subject
        stepped_keys = set(map(tuple, metrics_df.loc[metrics_df['trial_type'] == stepped_type, ['subject', 'trial_num']].values))
        missing_mask = ~hold_rows.apply(lambda r: (r['subject'], r[ref_col]) in stepped_keys, axis=1)
        for _, r in hold_rows[missing_mask].iterrows():
            validation_problems.append({
                'dataset': dataset, 'pair_type': pair_label, 'issue': 'referenced_stepped_trial_not_found',
                'subject': r['subject'], 'hold_trial_num': r['trial_num'], 'stepped_trial_num': r[ref_col],
            })

        clean_hold_rows = hold_rows[~(dup_mask | missing_mask)]

        for _, hrow in clean_hold_rows.iterrows():
            subject, hold_trial_num, stepped_trial_num = hrow['subject'], hrow['trial_num'], hrow[ref_col]
            stepped_row = metrics_df[(metrics_df['subject'] == subject) &
                                      (metrics_df['trial_type'] == stepped_type) &
                                      (metrics_df['trial_num'] == stepped_trial_num)].iloc[0]
            pair_id = f'{dataset}_s{subject}_{stepped_type}{stepped_trial_num}_{hold_type}{hold_trial_num}'

            for role, trial_type, trial_num, c_start, c_end in [
                ('stepped', stepped_type, stepped_trial_num, stepped_row['C_start'], stepped_row['C_end']),
                ('hold', hold_type, hold_trial_num, hrow['C_start'], hrow['C_end']),
            ]:
                ts = trial_data_df[
                    (trial_data_df['subject'] == subject) &
                    (trial_data_df['trial_num'] == trial_num) &
                    (trial_data_df['trial_type'] == trial_type) &
                    (trial_data_df['aligned_time'] >= c_start) &
                    (trial_data_df['aligned_time'] <= c_end)
                ].sort_values('aligned_time')

                row = {
                    'dataset': dataset, 'subject': subject, 'pair_id': pair_id, 'pair_type': pair_label,
                    'role': role, 'trial_type': trial_type, 'trial_num': trial_num,
                    'c_start': c_start, 'c_end': c_end, 'n_samples': len(ts),
                }
                for i, val in enumerate(ts['pain'].values):
                    row[f't{i}'] = val
                all_rows.append(row)

periodC_table = pd.DataFrame(all_rows)
validation_df = pd.DataFrame(validation_problems)

print(f'Built {len(periodC_table)} rows ({periodC_table["pair_id"].nunique()} pairs) across {len(datasets)} datasets.')
if not validation_df.empty:
    print(f'\n*** {len(validation_df)} validation problems found -- these pairs were EXCLUDED from the table ***')
    print(validation_df.to_string(index=False))
    validation_df.to_csv(VALIDATION_PATH, index=False)
    print(f'\nValidation problems saved to {VALIDATION_PATH}')
else:
    print('No validation problems -- every hold trial matched exactly one existing stepped trial.')

periodC_table.to_csv(OUTPUT_PATH, index=False)
print(f'\nPeriod C table saved to {OUTPUT_PATH}')

#%%
# ==================================================================================================================
######################################## LOAD CLINICAL GROUP LABELS ########################################
# ==================================================================================================================
conn = sqlite3.connect(sql_path)
kneeoa_groups = pd.read_sql_query('''
    SELECT DISTINCT subject, COALESCE(NULLIF("group", ""), 'control') AS group_label
    FROM metadata WHERE study = 'kneeOA' ORDER BY subject
''', conn)
conn.close()
kneeoa_groups['dataset'] = 'kneeOA'

conn = sqlite3.connect(sql_path)
clbp_groups = pd.read_sql_query('''
    SELECT DISTINCT subject, COALESCE(NULLIF("group", ''), 'High') AS group_label
    FROM metadata WHERE study LIKE 'cLBP%' ORDER BY subject
''', conn)
conn.close()
clbp_groups['dataset'] = 'cLBP'

plosone_subjects = periodC_table.loc[periodC_table['dataset'] == 'plosONE', 'subject'].unique()
plosone_groups = pd.DataFrame({'subject': plosone_subjects, 'group_label': 'Control', 'dataset': 'plosONE'})

all_groups = pd.concat([kneeoa_groups, clbp_groups, plosone_groups], ignore_index=True)

# 4 cLBP subject numbers (10, 40, 48, 53) have two conflicting group_label values because they
# appear in both the cLBP_DPOP and cLBP_MBPR sub-studies. Checked: DPOP and MBPR have very
# different trial_num ranges for these subjects (e.g. subject 10: DPOP has 24 trials up to
# trial_num 112, MBPR has 12 trials up to trial_num 12) -- the underlying data is not identical,
# so these are most likely two different people who happen to share a raw subject number across
# sub-studies, not one person with two group labels. Excluded from the group lookup rather than
# letting the merge duplicate their real trial rows into both group buckets.
ambiguous_mask = all_groups.duplicated(subset=['dataset', 'subject'], keep=False)
if ambiguous_mask.any():
    print(f'*** {all_groups.loc[ambiguous_mask, "subject"].nunique()} subject(s) have conflicting group_label across duplicate metadata rows -- excluded from group-based analyses ***')
    print(all_groups[ambiguous_mask].sort_values(['dataset', 'subject']).to_string(index=False))
    all_groups = all_groups[~ambiguous_mask]

periodC_table = periodC_table.merge(all_groups[['dataset', 'subject', 'group_label']], on=['dataset', 'subject'], how='left')

order = ['Control', 'Low', 'High']

#%%
# ==================================================================================================================
######################################## FULL 0-30s TRACE (PERIOD A THROUGH C) PER TRIAL ########################################
# ==================================================================================================================
# Same trials as periodC_table (post-validation), but pulling the full period-A-start -> period-C-end
# window (elapsed time relative to that trial's own A_start, i.e. t=0 is the start of period A) so
# periods A and B leading into the stepped/hold response are visible too, not just period C.
FULL_TRACE_PATH = f'{base_path}/stepped_hold_full_trace_table.csv'
full_trace_rows = []

for dataset in datasets:
    metrics_path = f'{base_path}/{dataset}_trial_metrics.json'
    data_path = f'{base_path}/{dataset}_trial_data_trimmed_downsampled.json'

    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
    metrics_records = []
    for subject_id, trials in metrics_data.items():
        for trial_num, trial_data in trials.items():
            metrics_records.append({'subject': int(subject_id), 'trial_num': int(trial_num), **trial_data})
    metrics_df = pd.DataFrame(metrics_records)

    trial_data_df = pd.read_json(data_path, orient='records')
    type_lookup = metrics_df[['subject', 'trial_num', 'trial_type']].drop_duplicates()
    trial_data_df = trial_data_df.merge(type_lookup, on=['subject', 'trial_num'], how='left')

    dataset_pairs = periodC_table.loc[
        periodC_table['dataset'] == dataset,
        ['subject', 'pair_id', 'pair_type', 'role', 'trial_type', 'trial_num', 'group_label']
    ].drop_duplicates()

    for _, prow in dataset_pairs.iterrows():
        subject, trial_num, trial_type = prow['subject'], prow['trial_num'], prow['trial_type']
        m_row = metrics_df[(metrics_df['subject'] == subject) &
                            (metrics_df['trial_type'] == trial_type) &
                            (metrics_df['trial_num'] == trial_num)].iloc[0]
        a_start, c_start, c_end = m_row['A_start'], m_row['C_start'], m_row['C_end']

        ts = trial_data_df[
            (trial_data_df['subject'] == subject) &
            (trial_data_df['trial_num'] == trial_num) &
            (trial_data_df['trial_type'] == trial_type) &
            (trial_data_df['aligned_time'] >= a_start) &
            (trial_data_df['aligned_time'] <= c_end)
        ].sort_values('aligned_time')

        row = {
            'dataset': dataset, 'subject': subject, 'pair_id': prow['pair_id'], 'pair_type': prow['pair_type'],
            'role': prow['role'], 'trial_type': trial_type, 'trial_num': trial_num, 'group_label': prow['group_label'],
            'c_start_rel': c_start - a_start, 'c_end_rel': c_end - a_start, 'n_samples': len(ts),
        }
        for i, val in enumerate(ts['pain'].values):
            row[f't{i}'] = val
        full_trace_rows.append(row)

full_trace_table = pd.DataFrame(full_trace_rows)
full_t_cols = sorted((c for c in full_trace_table.columns if re.fullmatch(r't\d+', c)), key=lambda c: int(c[1:]))

full_trace_table.to_csv(FULL_TRACE_PATH, index=False)
print(f'Full 0-30s trace table saved to {FULL_TRACE_PATH} ({len(full_trace_table)} rows, {len(full_t_cols)} timepoints)')

#%%
# ==================================================================================================================
######################################## PLOT FULL 0-30s TRACES BY GROUP (SEPARATE OH / OA FIGURES) ########################################
# ==================================================================================================================
# Replicate trials averaged within-subject first (same logic as the subject-averaging step later
# in this script), then mean +/- 95% CI across subjects at each timepoint. One figure per OH/OA
# pair, 3 panels stacked (Control/Low/High) sharing both axes so they're directly comparable.
# Period C is shaded using the median C-period boundary across all trials in the table.
subject_full_trace = (
    full_trace_table
    .groupby(['dataset', 'subject', 'pair_type', 'role', 'group_label'], as_index=False)[full_t_cols]
    .mean()
)

c_highlight_start = full_trace_table['c_start_rel'].median()
c_highlight_end = full_trace_table['c_end_rel'].median()
x_vals = np.arange(len(full_t_cols))

FULL_TRACE_PANELS = [
    ('OH', ['onset_t2_hold', 'inv_t2_hold'], 'OH (onset/inv vs t2_hold): Onset Hyperalgesia'),
    ('OA', ['offset_t1_hold'], 'OA (offset vs t1_hold): Offset Analgesia'),
]
ROLE_STYLE = {'stepped': '-', 'hold': '--'}

for panel_label, pair_types, fig_title in FULL_TRACE_PANELS:
    sub_all = subject_full_trace[subject_full_trace['pair_type'].isin(pair_types)]

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True, sharey=True)
    for ax, group in zip(axes, order):
        for role in ['stepped', 'hold']:
            role_sub = sub_all[(sub_all['group_label'] == group) & (sub_all['role'] == role)]
            if role_sub.empty:
                continue
            vals = role_sub[full_t_cols].to_numpy()
            mean = np.nanmean(vals, axis=0)
            ci95 = stats.sem(vals, axis=0, nan_policy='omit') * 1.96
            ax.plot(x_vals, mean, color=GROUP_COLORS[group], linestyle=ROLE_STYLE[role],
                    label=f'{role} (n={role_sub.shape[0]})')
            ax.fill_between(x_vals, mean - ci95, mean + ci95, color=GROUP_COLORS[group], alpha=0.2)

        ax.axvspan(c_highlight_start, c_highlight_end, color='gray', alpha=0.12,
                   label='Period C' if group == order[0] else None)
        ax.set_title(group, loc='left', fontweight='bold', color=GROUP_COLORS[group])
        ax.set_ylabel('Pain Intensity')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time since start of Period A (s)')
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.savefig(f'{FIGPATH}/full_trace_{panel_label}_by_group.png',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

#%%
# ==================================================================================================================
######################################## COMPUTE OH / OA DIFFERENCES (stepped - hold) ########################################
# ==================================================================================================================
t_cols = sorted((c for c in periodC_table.columns if re.fullmatch(r't\d+', c)), key=lambda c: int(c[1:]))

diff_rows = []
skipped_pairs = []
for pair_id, grp in periodC_table.groupby('pair_id'):
    stepped = grp[grp['role'] == 'stepped']
    hold = grp[grp['role'] == 'hold']
    if len(stepped) != 1 or len(hold) != 1:
        skipped_pairs.append(pair_id)
        continue
    stepped, hold = stepped.iloc[0], hold.iloc[0]
    pair_type = stepped['pair_type']

    row = {
        'dataset': stepped['dataset'],
        'subject': stepped['subject'],
        'pair_id': pair_id,
        'pair_type': pair_type,
        'label': PAIR_TYPE_LABELS.get(pair_type),
        'stepped_trial_type': stepped['trial_type'],
        'stepped_trial_num': stepped['trial_num'],
        'hold_trial_type': hold['trial_type'],
        'hold_trial_num': hold['trial_num'],
    }
    for c in t_cols:
        s_val, h_val = stepped[c], hold[c]
        row[c] = (s_val - h_val) if pd.notna(s_val) and pd.notna(h_val) else np.nan
    diff_rows.append(row)

differences_table = pd.DataFrame(diff_rows)

print(f'\nComputed OH/OA differences for {len(differences_table)} pairs.')
print(differences_table['label'].value_counts())
if skipped_pairs:
    print(f'Skipped {len(skipped_pairs)} pair_id(s) missing a stepped or hold row: {skipped_pairs}')

differences_table.to_csv(DIFFERENCES_PATH, index=False)
print(f'Differences table saved to {DIFFERENCES_PATH}')

# %% Average OA and OH trials per subject to calculate a per-subject OH and OA effect
# ==================================================================================================================
######################################## AVERAGE OH / OA DIFFERENCES PER SUBJECT ########################################
# ==================================================================================================================
SUBJECT_AVG_PATH = f'{base_path}/stepped_hold_periodC_differences_subject_avg.csv'

subject_avg_table = (
    differences_table
    .groupby(['dataset', 'subject', 'label'], as_index=False)
    .agg(n_pairs=('pair_id', 'count'), **{c: (c, 'mean') for c in t_cols})
)

print(f'\nAveraged to {len(subject_avg_table)} subject-level OH/OA rows.')
print(subject_avg_table.groupby('label')['n_pairs'].describe())

subject_avg_table.to_csv(SUBJECT_AVG_PATH, index=False)
print(f'Subject-averaged differences saved to {SUBJECT_AVG_PATH}')

# %%
# ==================================================================================================================
######################################## COMPARE PERIOD-C OH / OA MAGNITUDE ACROSS CLINICAL GROUPS ########################################
# ==================================================================================================================
# Same approach as combining_datasets.py's "Plot OH and OA magnitude by clinical group" block,
# but using the period-C difference (stepped - hold, mean across t0-t19) instead of
# abs_normalized_pain_change as the per-subject magnitude.

# Reduce each subject's difference curve to a single magnitude: mean across t0-t19
subject_avg_table['period_c_magnitude'] = subject_avg_table[t_cols].mean(axis=1)

# Clinical group labels (all_groups) were already loaded earlier, right after periodC_table was built.
subject_avg_table = subject_avg_table.merge(all_groups[['dataset', 'subject', 'group_label']], on=['dataset', 'subject'], how='left')

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
panel_data = {}
tests_by_panel = {0: [], 1: []}

# single source of truth for panel order/titles -- avoids the label/title getting out of sync
PANEL_ORDER = [
    ('OH', 'OH (onset/inv - t2_hold): Onset Hyperalgesia'),
    ('OA', 'OA (offset - t1_hold): Offset Analgesia'),
]

for ax_idx, (label, title) in enumerate(PANEL_ORDER):
    subset = subject_avg_table[subject_avg_table['label'] == label].copy()
    panel_data[ax_idx] = subset

    group_series = {g: subset.loc[subset['group_label'] == g, 'period_c_magnitude'].dropna() for g in order}
    present_groups = [g for g in order if len(group_series[g]) > 1]

    raw_tests = []
    for g1, g2 in combinations(present_groups, 2):
        u_stat, p_raw = stats.mannwhitneyu(group_series[g1], group_series[g2], alternative='two-sided')
        raw_tests.append({'g1': g1, 'g2': g2, 'u_stat': u_stat, 'p_raw': p_raw})

    if raw_tests:
        pvals = [t['p_raw'] for t in raw_tests]
        reject, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, t in enumerate(raw_tests):
            t['p_fdr'] = p_fdr[i]
            t['sig'] = bool(reject[i])
            tests_by_panel[ax_idx].append(t)

for ax_idx, ax in enumerate(axes):
    subset = panel_data[ax_idx]
    label, title = PANEL_ORDER[ax_idx]

    sns.violinplot(data=subset, x='group_label', y='period_c_magnitude', order=order,
                    palette=GROUP_COLORS, inner='box', ax=ax)
    sns.stripplot(data=subset, x='group_label', y='period_c_magnitude', order=order,
                   color='black', alpha=0.35, size=4, ax=ax)

    ax.set_title(f'{title}: Subject-Averaged Period C Difference')
    ax.set_xlabel('Clinical Group')
    ax.set_ylabel('Period C Pain Intensity Difference (stepped - hold)')
    ax.grid(True, alpha=0.3)

    counts = subset.groupby('group_label')['subject'].nunique().reindex(order).fillna(0).astype(int)
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-6)
    y_n = y_min + 0.03 * y_span
    for xi, g in enumerate(order):
        ax.text(xi, y_n, f'n={counts[g]}', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    sig_tests = [t for t in tests_by_panel[ax_idx] if t['sig']]
    if sig_tests:
        x_map = {g: i for i, g in enumerate(order)}
        base_y = y_max + 0.04 * y_span
        step = 0.08 * y_span
        h = 0.02 * y_span
        for k, t in enumerate(sig_tests):
            x1, x2 = x_map[t['g1']], x_map[t['g2']]
            y = base_y + k * step
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.4, c='black')
            ax.text((x1 + x2) / 2, y + h, p_to_stars(t['p_fdr']), ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='black')
        top = base_y + (len(sig_tests) - 1) * step + h + 0.06 * y_span
        ax.set_ylim(y_min, top)

plt.tight_layout()
plt.savefig(f'{FIGPATH}/periodC_OH_OA_magnitude_by_group.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# %%
