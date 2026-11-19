# -*- coding: utf-8 -*-
"""
Period C of the PER-TRIAL fits, two figures:

  1. period-C RMSE by trial type and pain group (box + jittered trials)
  2. signed error (rating - model) in period C for offset (OA) and onset (OH)
     trials, averaged per subject, pain groups pooled

Source is the period == 'C' rows of *_per_trial_fits_per_period.csv
(per_period_per_trial.py): each trial's own whole-trial parameters, scored
inside the measured period-C window. Forearm only.

Author: Lucille Johnston
Updated: 11/05/26
"""
#%%
import glob
import os
from datetime import datetime

from scipy import stats as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
GROUP_ORDER = ['Control', 'Low', 'High']
GROUP_COLORS = {'Control': '#DBBBAF', 'Low': '#91472A', 'High': '#491400'}
TYPE_ORDER = ['offset', 'onset', 't1_hold', 't2_hold']
STAMP = f'{datetime.now():%Y%m%d}'

#%%
per_csv = sorted(glob.glob(RESULTS_PATH + '*_per_trial_fits_per_period.csv'))[-1]
per = pd.read_csv(per_csv)
c = per[(per['period'] == 'C') & per['rmse_trial'].notna()
        & per['group_label'].isin(GROUP_ORDER)
        & per['trial_type'].isin(TYPE_ORDER)]
print(f'{os.path.basename(per_csv)}: {len(c):,} period-C trials')

#%%
# ------------------------------------------------------------------
# Box + jittered trials
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8))
kw = dict(data=c, x='trial_type', y='rmse_trial', hue='group_label',
          order=TYPE_ORDER, hue_order=GROUP_ORDER, palette=GROUP_COLORS,
          ax=ax)
sns.boxplot(**kw, showfliers=False, width=.8, linewidth=.9,
            boxprops=dict(alpha=.6), medianprops=dict(color='k', lw=1.5))
sns.stripplot(**kw, dodge=True, size=2.2, alpha=.55, jitter=.22,
              edgecolor='k', linewidth=.2, legend=False)
ax.tick_params(axis='x', length=0, labelsize=11)
ax.set(xlabel='', ylabel='period-C RMSE, per-trial fit (VAS)')
ax.set_ylim(bottom=0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='', frameon=False, fontsize=8)
ax.set_title('Period-C fit error by trial type', fontsize=10)
fig.tight_layout()
out = FIG_PATH + f'{STAMP}_periodC_rmse_by_trial_type.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'-> {out}')

print('\nmedian period-C RMSE (VAS)')
print(c.pivot_table(index='trial_type', columns='group_label',
                    values='rmse_trial', aggfunc='median')
       .reindex(index=TYPE_ORDER, columns=GROUP_ORDER).round(2))

#%%
# ------------------------------------------------------------------
# Signed error in period C, stepped trials, per-trial fits
# ------------------------------------------------------------------
# signed_trial = mean(rating) - mean(model) inside the period-C window.
#   < 0: the model sits ABOVE the ratings
#   > 0: the model sits BELOW the ratings
# Averaged per subject first so each person counts once; pain groups pooled.
STEPPED = ['offset', 'onset']
TRIAL_LABELS = {'offset': 'Offset (OA)', 'onset': 'Onset (OH)'}
TRIAL_COLORS = {'offset': '#026A07', 'onset': '#DA1C1C'}   # per_period_fits.py
subj = (c[c['trial_type'].isin(STEPPED) & c['signed_trial'].notna()]
        .groupby(['subject_uid', 'trial_type'])['signed_trial']
        .mean().reset_index())


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


print('\n--- Period C signed error (rating - model), per-subject means ---')
for tt in STEPPED:
    v = subj.loc[subj['trial_type'] == tt, 'signed_trial']
    _, p = st.wilcoxon(v)
    print(f'  {TRIAL_LABELS[tt]:12s} n={len(v)}  median {v.median():+5.2f}  '
          f'IQR [{v.quantile(.25):+.2f}, {v.quantile(.75):+.2f}]  '
          f'vs 0: p={p:.2g}  {(v < 0).mean():.0%} of subjects < 0')

fig, ax = plt.subplots(figsize=(4.8, 5.2))
sns.violinplot(data=subj, x='trial_type', y='signed_trial', order=STEPPED,
               hue='trial_type', palette=TRIAL_COLORS, legend=False, cut=0,
               inner='quart', linewidth=1.0, saturation=1.0, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.85)
ax.axhline(0, color='0.3', lw=1.3, ls=(0, (4, 3)), zorder=0)
top, bot = subj['signed_trial'].max(), subj['signed_trial'].min()
span = top - bot
for i, tt in enumerate(STEPPED):
    v = subj.loc[subj['trial_type'] == tt, 'signed_trial']
    _, p = st.wilcoxon(v)
    ax.text(i, bot - .03 * span, f'{v.median():+.1f}\n{stars(p)}',
            ha='center', va='top', fontsize=9, color='0.2')
ax.set_ylim(bot - .14 * span, top + .05 * span)
ax.set_xticks(range(len(STEPPED)))
ax.set_xticklabels([TRIAL_LABELS[t] for t in STEPPED], fontsize=11)
ax.text(1.01, .98, 'model too low', transform=ax.transAxes, fontsize=7.5,
        color='0.4', va='top', rotation=90)
ax.text(1.01, .02, 'model too high', transform=ax.transAxes, fontsize=7.5,
        color='0.4', va='bottom', rotation=90)
ax.set(xlabel='', ylabel='period C: rating − model (VAS) per subject',
       title='Signed Error in Period C, Per-Trial Fits')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
out = FIG_PATH + f'{STAMP}_periodC_signed_error_stepped_per_trial.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'-> {out}')
plt.show()

#%%
# ------------------------------------------------------------------
# One example of each: rating vs per-trial model, period C shaded
# ------------------------------------------------------------------
# The example is the trial whose period-C signed error is closest to the
# per-subject median in the figure above, so it shows the typical miss rather
# than a hand-picked one.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (prepare_trials_for_optimization,
                                              pack_trials, simulate_trials_full)

TRACES_FILE = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/'
               'Python/temporal_contrast_enhancement/data/alter_collab_data/'
               'combined_traces_1Hz.pkl')
traces = pd.read_pickle(TRACES_FILE)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, tt in zip(axes, STEPPED):
    target = subj.loc[subj['trial_type'] == tt, 'signed_trial'].median()
    cand = c[c['trial_type'] == tt]
    ex = cand.loc[(cand['signed_trial'] - target).abs().idxmin()]
    trials = prepare_trials_for_optimization(
        traces[traces['subject_uid'] == ex['subject_uid']])
    tr = next(t for t in trials if t['trial_num'] == ex['trial_num'])
    params = {k: ex[k] for k in ('alpha', 'beta', 'gamma', 'lam', 'theta')}
    n = len(tr['time'])
    pred = simulate_trials_full(params, pack_trials([tr]))[0, :n]

    win = pd.read_csv(sorted(glob.glob(RESULTS_PATH
                                       + '*_measured_trial_windows.csv'))[-1])
    w = win[(win['subject_uid'] == ex['subject_uid'])
            & (win['trial_num'] == ex['trial_num'])].iloc[0]
    ax.axvspan(w['C_start'], w['C_end'], color='0.92', zorder=0)
    ax.text((w['C_start'] + w['C_end']) / 2, 1.0, 'period C',
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=8, color='0.4')
    ax.plot(tr['time'], tr['pain'], color='k', lw=1.8, label='rating')
    ax.plot(tr['time'], pred, color=TRIAL_COLORS[tt], lw=1.8,
            label='model (per-trial fit)')
    ax.set_title(f'{TRIAL_LABELS[tt]}: {ex["subject_uid"]}, trial '
                 f'{int(ex["trial_num"])}\nperiod C rating − model = '
                 f'{ex["signed_trial"]:+.1f} VAS', fontsize=9.5, pad=14)
    ax.set_xlabel('time (s)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower center')
    print(f'{tt}: {ex["subject_uid"]} trial {int(ex["trial_num"])} '
          f'({ex["group_label"]}), period-C signed {ex["signed_trial"]:+.2f}, '
          f'RMSE {ex["rmse_trial"]:.2f}, at_bounds={ex["at_bounds"]}')
axes[0].set_ylabel('pain (VAS)')
plt.tight_layout()
out = FIG_PATH + f'{STAMP}_periodC_signed_error_examples.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'-> {out}')
plt.show()
