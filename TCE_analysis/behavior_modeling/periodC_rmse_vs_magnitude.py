# -*- coding: utf-8 -*-
"""
Does the per-trial model fit worse in period C on trials with bigger OA / OH?

x: period-C RMSE of each trial's OWN whole-trial fit (rmse_trial, period == 'C'
   in *_per_trial_fits_per_period.csv, written by per_period_per_trial.py).
   The parameters come from the whole-trial fit, not a period-C refit, so the
   error is not circular with the parameters -- see that script's docstring.
y: OA magnitude (offset trials) / OH magnitude (onset trials), taken as
   abs_normalized_pain_change from *_trial_metrics.json -- the same measure
   FENS_figures.py uses for the OA/OH-by-group panels.

One dot per trial, coloured by pain group. Forearm only, because the
per-period file is.

Author: Lucille Johnston
Updated: 11/05/26
"""
#%%
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
DATASETS = ['plosONE', 'kneeOA', 'cLBP']
GROUP_ORDER = ['Control', 'Low', 'High']
GROUP_COLORS = {'Control': '#DBBBAF', 'Low': '#91472A', 'High': '#491400'}
MAG_COL = 'abs_normalized_pain_change'
STAMP = f'{datetime.now():%Y%m%d}'

#%%
# ------------------------------------------------------------------
# Load period-C error and trial metrics, join on (subject_uid, trial_num)
# ------------------------------------------------------------------
per_csv = sorted(glob.glob(RESULTS_PATH + '*_per_trial_fits_per_period.csv'))[-1]
per = pd.read_csv(per_csv)
c = per[(per['period'] == 'C') & per['rmse_trial'].notna()]
print(f'period-C rows : {os.path.basename(per_csv)} ({len(c):,} trials)')

recs = []
for ds in DATASETS:
    with open(f'{DATA_PATH}{ds}_trial_metrics.json') as f:
        m = json.load(f)
    # metrics keys are already canonical (cLBP on DPOP_IDs), which is the same
    # number build_combined_traces.py zero-pads into subject_uid
    for subj, trials in m.items():
        for tn, t in trials.items():
            recs.append({'subject_uid': f'{ds}_{int(subj):04d}',
                         'trial_num': int(tn),
                         'trial_type_metrics': t.get('trial_type'),
                         MAG_COL: t.get(MAG_COL)})
met = pd.DataFrame(recs)
# plosONE calls onset trials 'inv'; extract_metrics.py scores both identically
# and combining_datasets.py renames them the same way
met['trial_type_metrics'] = met['trial_type_metrics'].replace({'inv': 'onset'})

df = c.merge(met, on=['subject_uid', 'trial_num'], how='inner')
bad = (df['trial_type'] != df['trial_type_metrics']).sum()
if bad:
    print(f'WARNING: {bad} joined trials disagree on trial_type -- dropped')
    df = df[df['trial_type'] == df['trial_type_metrics']]
df = df[df['group_label'].isin(GROUP_ORDER) & df[MAG_COL].notna()]
for tt in ('offset', 'onset'):
    n_c = (c['trial_type'] == tt).sum()
    n_j = (df['trial_type'] == tt).sum()
    print(f'{tt:6s}: {n_j:,}/{n_c:,} period-C trials have a metrics match')

#%%
# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
panels = [('offset', 'OA magnitude (offset trials)'),
          ('onset', 'OH magnitude (onset trials)')]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (tt, ylab) in zip(axes, panels):
    s = df[df['trial_type'] == tt]
    # draw Control last so the pale dots are not buried under the dark ones
    for g in ['High', 'Low', 'Control']:
        sg = s[s['group_label'] == g]
        ax.scatter(sg['rmse_trial'], sg[MAG_COL], s=12, alpha=.6,
                   color=GROUP_COLORS[g], linewidths=0,
                   label=f'{g} (n={len(sg)} trials, '
                         f'{sg["subject_uid"].nunique()} subj)')
    lines = []
    for g in GROUP_ORDER:
        sg = s[s['group_label'] == g]
        if len(sg) > 3:
            rho, p = st.spearmanr(sg['rmse_trial'], sg[MAG_COL])
            lines.append(f'{g}: ρ={rho:+.2f}, p={p:.2g}')
    rho, p = st.spearmanr(s['rmse_trial'], s[MAG_COL])
    lines.append(f'all: ρ={rho:+.2f}, p={p:.2g}')
    ax.text(.98, .98, '\n'.join(lines), transform=ax.transAxes, ha='right',
            va='top', fontsize=8)
    ax.axhline(0, color='0.7', lw=.8, zorder=0)
    ax.set(xlabel='period-C RMSE, per-trial fit (VAS)', ylabel=ylab)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='center right')
fig.suptitle('Period-C fit error vs. OA / OH magnitude, by pain group '
             '(one dot per trial; Spearman ρ is trial-level, ignores '
             'subject nesting)', fontsize=10)
fig.tight_layout()
out = FIG_PATH + f'{STAMP}_periodC_rmse_vs_OA_OH_magnitude.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'-> {out}')
plt.show()

#%%
# ------------------------------------------------------------------
# Subject level: the trial-level rho above counts every trial as independent,
# but each subject contributes several. Two ways round that:
#   (a) collapse to one median per subject and correlate those
#   (b) keep every trial in a linear mixed model with a random intercept per
#       subject, so a subject's repeated trials share one baseline
# ------------------------------------------------------------------
import statsmodels.formula.api as smf
from scipy.stats import chi2

subj = (df.groupby(['subject_uid', 'dataset', 'group_label', 'trial_type'])
          .agg(rmse_C=('rmse_trial', 'median'), mag=(MAG_COL, 'median'),
               n_trials=(MAG_COL, 'size'))
          .reset_index())

lmm_rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (tt, ylab) in zip(axes, panels):
    # ---- (a) subject medians ----
    s = subj[subj['trial_type'] == tt]
    for g in ['High', 'Low', 'Control']:
        sg = s[s['group_label'] == g]
        ax.scatter(sg['rmse_C'], sg['mag'], s=22, alpha=.8,
                   color=GROUP_COLORS[g], edgecolor='k', linewidths=.3,
                   label=f'{g} (n={len(sg)} subj)')
    lines = []
    for g in GROUP_ORDER:
        sg = s[s['group_label'] == g]
        rho, p = st.spearmanr(sg['rmse_C'], sg['mag'])
        lines.append(f'{g}: ρ={rho:+.2f}, p={p:.2g}')
    rho, p = st.spearmanr(s['rmse_C'], s['mag'])
    lines.append(f'all: ρ={rho:+.2f}, p={p:.2g}')

    # ---- (b) mixed models on the trials ----
    t = (df[df['trial_type'] == tt]
         .rename(columns={'rmse_trial': 'rmse_C', MAG_COL: 'mag',
                          'group_label': 'grp'}))
    t['grp'] = pd.Categorical(t['grp'], categories=GROUP_ORDER)
    # ML rather than REML so the two models can be compared by likelihood ratio
    pooled = smf.mixedlm('mag ~ rmse_C + grp', t,
                         groups=t['subject_uid']).fit(reml=False)
    by_grp = smf.mixedlm('mag ~ grp + grp:rmse_C', t,
                         groups=t['subject_uid']).fit(reml=False)
    lr = 2 * (by_grp.llf - pooled.llf)
    p_lr = chi2.sf(lr, df=len(GROUP_ORDER) - 1)
    ci = pooled.conf_int().loc['rmse_C']
    lmm_rows.append({'trial_type': tt, 'group': 'pooled',
                     'slope': pooled.params['rmse_C'], 'ci_lo': ci[0],
                     'ci_hi': ci[1], 'p': pooled.pvalues['rmse_C'],
                     'n_trials': len(t), 'n_subj': t['subject_uid'].nunique(),
                     'p_slopes_differ': p_lr})
    for g in GROUP_ORDER:
        k = f'grp[{g}]:rmse_C'
        ci = by_grp.conf_int().loc[k]
        lmm_rows.append({'trial_type': tt, 'group': g,
                         'slope': by_grp.params[k], 'ci_lo': ci[0],
                         'ci_hi': ci[1], 'p': by_grp.pvalues[k],
                         'n_trials': int((t['grp'] == g).sum()),
                         'n_subj': t.loc[t['grp'] == g, 'subject_uid'].nunique(),
                         'p_slopes_differ': np.nan})
    lines.append(f'\nLMM slope {pooled.params["rmse_C"]:+.2f} /VAS RMSE, '
                 f'p={pooled.pvalues["rmse_C"]:.2g}')
    lines.append(f'group slopes differ: p={p_lr:.2g}')

    ax.text(.98, .98, '\n'.join(lines), transform=ax.transAxes, ha='right',
            va='top', fontsize=8)
    ax.axhline(0, color='0.7', lw=.8, zorder=0)
    ax.set(xlabel='median period-C RMSE, per-trial fit (VAS)',
           ylabel=f'median {ylab}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='center right')
fig.suptitle('Subject level: one dot per subject (medians over trials); '
             'LMM = all trials, random intercept per subject', fontsize=10)
fig.tight_layout()
out = FIG_PATH + f'{STAMP}_periodC_rmse_vs_OA_OH_magnitude_subject.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'-> {out}')
plt.show()

lmm = pd.DataFrame(lmm_rows)
print('\nLMM: magnitude ~ period-C RMSE (+ group), random intercept per subject')
print(lmm.round(4).to_string(index=False))
lmm_out = RESULTS_PATH + f'{STAMP}_periodC_rmse_vs_magnitude_lmm.csv'
lmm.to_csv(lmm_out, index=False)
print(f'-> {lmm_out}')
