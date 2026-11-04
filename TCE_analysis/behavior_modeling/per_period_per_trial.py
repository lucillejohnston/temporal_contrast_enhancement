# -*- coding: utf-8 -*-
"""
Score the PER-TRIAL Eq. 1 fits inside the stimulus periods.

per_period_fits.py does this for the one-set-per-subject fits. This is the
same scoring with each trial's own fitted parameters substituted in, which
answers two different questions:

  1. How much of the per-trial fit's advantage lands in period C, the window
     the offset effect lives in, rather than in the easy parts of the trial.

  2. Whether the parameters predict where the model fails. Only one term in
     Eq. 1 can produce offset analgesia -- the rate term gamma*(T' - lam)*p --
     so on offset trials the model should miss period C exactly where that
     term is weakest: small gamma, large lam, or heavy damping. Hold trials
     are the control, because a hold has no step to miss, so any relationship
     there is general noise rather than a failure to capture OA.

NOTE on why the parameters here come from the whole-trial fit rather than
from a period-C-only refit. Refitting on period C would make the parameters
and the period-C error two outputs of the same optimisation: a trial would
score badly precisely because its parameters could not do better, so a
scatter of one against the other is partly circular. The whole-trial fits are
estimated without reference to how the period-C window alone is scored, which
is the independence this question needs. A period-C-only refit answers a
different question -- what parameters period C on its own would prefer -- and
is a separate job.

Author: Lucille Johnston
Updated: 11/01/26
"""
#%%
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (prepare_trials_for_optimization,
                                              pack_trials, simulate_trials_full)

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
TRACES_FILE = DATA_PATH + 'combined_traces_1Hz.pkl'

SITE = 'forearm'
PERIODS = ['A', 'B', 'C']
MIN_POINTS = 5            # same floor as per_period_fits.py
MAIN_TYPES = ['offset', 'onset', 't1_hold', 't2_hold']
STAMP = f'{datetime.now():%Y%m%d}'
OUT_CSV = RESULTS_PATH + f'{STAMP}_per_trial_fits_per_period.csv'

# offset is paired with t1_hold and onset with t2_hold because those pairs sit
# at the same temperature during period C, so a difference between them is
# about the step rather than about the heat
PAIRS = [('offset', 't1_hold'), ('onset', 't2_hold')]
TYPE_COLORS = {'offset': '#B03030', 'onset': '#2B6CA3',
               't1_hold': '#C8B8A8', 't2_hold': '#A8B8C8'}

#%%
# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
traces = pd.read_pickle(TRACES_FILE)
if 'site' not in traces.columns:
    traces['site'] = SITE
traces['site'] = traces['site'].fillna(SITE)
traces = traces[traces['site'] == SITE]

pt_csv = sorted(glob.glob(RESULTS_PATH + '*_per_trial_fits_full_fixedtheta.csv'))[-1]
win_csv = sorted(glob.glob(RESULTS_PATH + '*_measured_trial_windows.csv'))[-1]
pt = pd.read_csv(pt_csv)
win = pd.read_csv(win_csv)
win = win[win['site'] == SITE]
print(f'per-trial fits : {os.path.basename(pt_csv)} ({len(pt):,} trials)')
print(f'windows        : {os.path.basename(win_csv)} ({len(win):,} trials)')

bounds = win.set_index(['subject_uid', 'trial_num'])
fits = pt.set_index(['subject_uid', 'trial_num'])

#%%
# ------------------------------------------------------------------
# Simulate each trial with its own parameters and score the windows
# ------------------------------------------------------------------
# Both parameter sets are simulated in the same pass so the comparison is
# exactly like-for-like: same trial, same window mask, same scoring.
rows = []
n_no_bounds = 0
uids = sorted(set(pt['subject_uid']) & set(traces['subject_uid']))
print(f'\nscoring {len(uids)} subjects...')

for i, uid in enumerate(uids):
    trials = prepare_trials_for_optimization(traces[traces['subject_uid'] == uid])
    if not trials:
        continue
    for tr in trials:
        key = (uid, tr['trial_num'])
        if key not in fits.index:
            continue
        f = fits.loc[key]
        if isinstance(f, pd.DataFrame):      # duplicate trial_num, skip
            continue
        packed = pack_trials([tr])
        n = len(tr['time'])
        pred = {}
        for tag, cols in (('trial', ('alpha', 'beta', 'gamma', 'lam', 'theta')),
                          ('subject', ('alpha_subject', 'beta_subject',
                                       'gamma_subject', 'lam_subject',
                                       'theta_subject'))):
            p = dict(zip(('alpha', 'beta', 'gamma', 'lam', 'theta'),
                         (f[c] for c in cols)))
            if any(pd.isna(v) for v in p.values()):
                pred[tag] = None
                continue
            pred[tag] = simulate_trials_full(p, packed)[0, :n]

        masks = {'full': np.ones(n, dtype=bool)}
        if key in bounds.index:
            b = bounds.loc[key]
            for per in PERIODS:
                lo, hi = b[f'{per}_start'], b[f'{per}_end']
                if pd.notna(lo) and pd.notna(hi):
                    masks[per] = (tr['time'] >= lo) & (tr['time'] <= hi)
        else:
            n_no_bounds += 1

        for per, m in masks.items():
            if m.sum() < MIN_POINTS:
                continue
            obs = tr['pain'][m]
            row = {'subject_uid': uid, 'dataset': f['dataset'],
                   'group_label': f['group_label'], 'trial_num': tr['trial_num'],
                   'trial_type': tr['trial_type'], 'period': per,
                   'n_points': int(m.sum()), 'mean_obs': obs.mean(),
                   'trial_order': f.get('trial_order', np.nan),
                   # the parameters are a property of the trial, repeated on
                   # each of its window rows so a period can be selected
                   # without going back to the fit table
                   **{c: f[c] for c in ('alpha', 'beta', 'gamma', 'lam',
                                        'theta', 'at_bounds')}}
            for tag in ('trial', 'subject'):
                if pred[tag] is None:
                    row[f'rmse_{tag}'] = np.nan
                    row[f'signed_{tag}'] = np.nan
                    continue
                pw = pred[tag][m]
                row[f'rmse_{tag}'] = float(np.sqrt(np.mean((obs - pw) ** 2)))
                # positive = the model sits BELOW the rating
                row[f'signed_{tag}'] = float(obs.mean() - pw.mean())
            rows.append(row)
    if (i + 1) % 50 == 0:
        print(f'  {i + 1}/{len(uids)}')

per = pd.DataFrame(rows)
# gamma and lam enter the prediction only through these two, so they are the
# quantities to test against fit quality rather than the raw parameters
per['gl'] = per['gamma'] * per['lam']
per['zeta2'] = per['beta'] ** 2 / (4 * per['gl'])
per['gain_drop'] = per['rmse_subject'] - per['rmse_trial']
per.to_csv(OUT_CSV, index=False)
print(f'\n{len(per):,} window rows -> {os.path.basename(OUT_CSV)}')
if n_no_bounds:
    print(f'{n_no_bounds} trials had no measured window boundaries')

#%%
# ==================================================================
# 1. WHERE DOES THE PER-TRIAL FIT'S ADVANTAGE LAND?
# ==================================================================
print('\n' + '=' * 74)
print('1. PER-TRIAL vs PER-SUBJECT FIT, BY WINDOW')
print('=' * 74)
m = per[per['trial_type'].isin(MAIN_TYPES)]
tab = (m.groupby(['period', 'trial_type'])[['rmse_subject', 'rmse_trial']]
       .median().round(2))
tab['gain'] = (tab['rmse_subject'] - tab['rmse_trial']).round(2)
print(tab.to_string())
print('\n  gain = how many VAS the per-trial fit buys in that window.')

#%%
# ==================================================================
# 2. DO THE PARAMETERS PREDICT WHERE THE MODEL FAILS IN PERIOD C?
# ==================================================================
# The prediction, stated before looking: on OFFSET trials period-C error
# should fall as the rate term gets stronger -- higher gamma, lower lam, less
# damping -- because that term is the only route to offset analgesia in Eq. 1.
# On t1_hold there is no step, so the same relationship should be absent or
# much weaker. The hold panel is what makes this a test rather than a
# description; without it any correlation could just be "badly fitted trials
# have odd parameters".
print('\n' + '=' * 74)
print('2. PERIOD-C ERROR vs THE RATE-TERM PARAMETERS')
print('=' * 74)
PREDICTORS = [('gamma', r'$\gamma$   (rate gain)'),
              ('lam', r'$\lambda$   (rate offset)'),
              ('zeta2', r'$\beta^2/4\gamma\lambda$   (>1 = overdamped)')]
c = per[(per['period'] == 'C') & per['trial_type'].isin(MAIN_TYPES)].copy()
c = c[c['rmse_trial'].notna()]
print(f'{len(c):,} period-C rows, {c["subject_uid"].nunique()} subjects\n')
print(f'{"":10s} ' + ' '.join(f'{t:>18s}' for t in MAIN_TYPES))
for k, _ in PREDICTORS:
    cells = []
    for tt in MAIN_TYPES:
        s = c[(c['trial_type'] == tt) & (c[k] > 0)]
        rho, pv = st.spearmanr(np.log10(s[k]), s['rmse_trial'])
        cells.append(f'{rho:+.2f} (p={pv:.1g})')
    print(f'{k:10s} ' + ' '.join(f'{x:>18s}' for x in cells))
print('\n  negative rho on offset = the model does BETTER in period C when the\n'
      '  rate term is stronger, which is the predicted direction.')

#%%
fig, axes = plt.subplots(len(PREDICTORS), len(PAIRS) * 2,
                         figsize=(4.0 * len(PAIRS) * 2, 3.4 * len(PREDICTORS)),
                         sharey='row')
order = [tt for pair in PAIRS for tt in pair]
for r, (k, klabel) in enumerate(PREDICTORS):
    for col, tt in enumerate(order):
        ax = axes[r, col]
        s = c[(c['trial_type'] == tt) & (c[k] > 0)]
        ax.scatter(s[k], s['rmse_trial'], s=7, alpha=.35, linewidths=0,
                   color=TYPE_COLORS[tt])
        rho, pv = st.spearmanr(np.log10(s[k]), s['rmse_trial'])
        ax.text(.03, .96, f'$\\rho$ = {rho:+.2f}\np = {pv:.1g}',
                transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
                color='0.15')
        ax.set_xscale('log')
        if k == 'zeta2':
            ax.axvline(1.0, color='#B03030', lw=1.2, ls='--', zorder=4)
        if r == 0:
            ax.set_title(tt, fontsize=11)
        # every row, not just the bottom one: each ROW is a different
        # predictor, so a single label at the foot of the grid would leave the
        # other rows' axes unexplained
        ax.set_xlabel(klabel, fontsize=10)
        if col == 0:
            ax.set_ylabel('period-C RMSE (VAS)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Period-C error against the rate-term parameters, one dot per '
             'trial.   Stepped trials first in each pair, then its '
             'temperature-matched hold.', y=1.005)
plt.tight_layout()
plt.savefig(FIG_PATH + f'{STAMP}_periodC_error_vs_params.png',
            dpi=200, bbox_inches='tight')
plt.show()
