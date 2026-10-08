# -*- coding: utf-8 -*-
"""
Score both models inside the stimulus periods, not just over the whole trial.

Whole-trial r is dominated by the rise phase, which every model fits well, so
it buries the part that matters. Offset analgesia happens in period C -- the
20s after the temperature steps back down -- and that window is a third of the
trial. A model can miss the entire offset transient and still score r = 0.82
over the trial.

Periods come from the A/B/C boundaries in *_trial_metrics.json, which follow
the stimulus protocol (5s at T1, 5s at T2, 20s back at T1):

    A   5s  baseline plateau at T1
    B   5s  the perturbation (up for offset trials, down for onset)
    C  20s  back at T1 -- where offset analgesia appears

Onset trials use the same windows, so C is comparable across trial types:
for offset it follows a step down, for onset a step up, and for the hold
trials the stimulus never moves, which makes them the control.

Two cautions about that metrics file, neither of which affects the timings.
It was built from *_trial_data_trimmed_downsampled.json, which carries no
'study' column, so for the 7 cLBP subject numbers shared by a DPOP and an
MBPR subject there is one boundary record where there should be two. Period
boundaries come from the stimulus protocol rather than from the person, so
sharing them is sound -- but the AUCs in that same file are computed from the
ratings and ARE blended for those subjects, so nothing here uses them.

Author: Lucille Johnston
Updated: 9/25/26
"""
#%%
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (prepare_trials_for_optimization,
                                              simulate_trials_analytic,
                                              pack_trials, simulate_trials_full,
                                              fit_metrics)

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
TRACES_FILE = DATA_PATH + 'combined_traces_1Hz.pkl'

MIN_POINTS = 5     # r on fewer than this is not worth computing
MIN_PAIN_SD = 2.0  # VAS; a flat window has no correlation to speak of

# kneeOA was stimulated at forearm and knee; everything here is forearm, which
# is what extract_metrics.py (and so the period boundaries) assumes, and what
# makes kneeOA comparable to the single-site plosONE and cLBP.
SITE = 'forearm'

# The Eq. 1 fits predate the site split. For plosONE and cLBP that is harmless
# -- single site, unchanged traces -- but kneeOA's Eq. 1 fits were run on
# forearm and knee pooled, while sharing a key with the new forearm-only Eq. 2
# fits. Joining them would compare a forearm fit against a pooled one without
# any warning, so kneeOA is dropped from the Eq. 1 columns until it is refit.
EQ1_STALE_DATASETS = ['kneeOA']

STAMP = f'{datetime.now():%Y%m%d}'
OUT_CSV = RESULTS_PATH + f'{STAMP}_period_metrics.csv'


#%%
# ------------------------------------------------------------------
# Load traces, both sets of fits, and the period boundaries
# ------------------------------------------------------------------
data_df = pd.read_pickle(TRACES_FILE)

simple_csv = [f for f in sorted(glob.glob(RESULTS_PATH + '*model_fits_subject_*.csv'))
              if 'full' not in os.path.basename(f)][-1]
full_csv = sorted(glob.glob(RESULTS_PATH + '*_model_fits_subject_full.csv'))[-1]
simple = pd.read_csv(simple_csv)
full = pd.read_csv(full_csv)
print(f'Eq. 2 fits : {os.path.basename(simple_csv)} ({len(simple)})')
print(f'Eq. 1 fits : {os.path.basename(full_csv)} ({len(full)})')

# Fits carried forward from an earlier run predate the site columns; they are
# all single-site forearm and their fit unit is just the subject id.
if 'site' not in simple.columns:
    simple['site'] = SITE
simple['site'] = simple['site'].fillna(SITE)
if 'subject' not in simple.columns:
    simple['subject'] = simple['subject_uid']
simple['subject'] = simple['subject'].fillna(simple['subject_uid'])

before = len(simple)
simple = simple[simple['site'] == SITE]
print(f'site filter ({SITE}): {before} -> {len(simple)} fits')

full = full[~full['dataset'].isin(EQ1_STALE_DATASETS)]
print(f'Eq. 1 usable (excluding {EQ1_STALE_DATASETS} as pre-site-split): {len(full)}')

simple = simple.set_index('subject_uid')
full = full.set_index('subject_uid')

rows = []
for ds in ['plosONE', 'kneeOA', 'cLBP']:
    path = f'{DATA_PATH}{ds}_trial_metrics.json'
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    for subj, trials_d in d.items():
        for tnum, v in trials_d.items():
            rows.append({'dataset': ds, 'subject_orig': int(subj),
                         'trial_num': int(tnum),
                         **{k: v.get(k) for k in
                            ['A_start', 'A_end', 'B_start', 'B_end',
                             'C_start', 'C_end']}})
bounds = pd.DataFrame(rows)
print(f'period boundaries: {len(bounds):,} trials from '
      f'{bounds.groupby(["dataset", "subject_orig"]).ngroups} subject records')

# The boundaries are in aligned_time, the same clock the traces use.
lookup = bounds.set_index(['dataset', 'subject_orig', 'trial_num'])

PERIODS = ['A', 'B', 'C']


#%%
# ------------------------------------------------------------------
# Recompute predictions and score each window
# ------------------------------------------------------------------
# Predictions are regenerated from the fitted parameters rather than read back
# out of the result pickles: it avoids depending on their internals, and it is
# cheap (Eq. 2 is a closed form; Eq. 1 is ~30ms per subject).
out = []
missing_bounds = 0
data_df = data_df[data_df['site'].fillna(SITE) == SITE]
uids = sorted(simple.index)
has_eq1 = set(full.index)
print(f'\nscoring {len(uids)} {SITE} fits '
      f'({len(set(uids) & has_eq1)} of them also have a usable Eq. 1 fit)...')

for i, uid in enumerate(uids):
    sd = data_df[data_df['subject_uid'] == uid]   # forearm units keep the plain id
    trials = prepare_trials_for_optimization(sd)
    if not trials:
        continue

    s = simple.loc[uid]
    pred2 = simulate_trials_analytic(
        {'alpha_bar': s['alpha_bar'], 'gamma_bar': s['gamma_bar'],
         'theta': s['theta']}, trials)
    pred1_mat = None
    if uid in has_eq1:
        f = full.loc[uid]
        pred1_mat = simulate_trials_full(
            {'alpha': f['alpha'], 'beta': f['beta'], 'gamma': f['gamma'],
             'lam': f['lam'], 'theta': f['theta']}, pack_trials(trials))

    dataset = sd['dataset'].iloc[0]
    subject_orig = int(sd['subject_orig'].iloc[0])

    for j, tr in enumerate(trials):
        t = tr['time']
        obs = tr['pain']
        p2 = pred2[j]
        p1 = pred1_mat[j, :len(t)] if pred1_mat is not None else None

        base = {'subject_uid': uid, 'dataset': dataset, 'study': sd['study'].iloc[0],
                'group_label': sd['group_label'].iloc[0],
                'subject_orig': subject_orig, 'trial_num': tr['trial_num'],
                'trial_type': tr['trial_type']}

        key = (dataset, subject_orig, tr['trial_num'])
        b = lookup.loc[key] if key in lookup.index else None
        if b is None:
            missing_bounds += 1

        windows = {'full': np.ones(len(t), dtype=bool)}
        if b is not None:
            for p in PERIODS:
                lo, hi = b[f'{p}_start'], b[f'{p}_end']
                if pd.notna(lo) and pd.notna(hi):
                    windows[p] = (t >= lo) & (t <= hi)

        for pname, mask in windows.items():
            n = int(mask.sum())
            o = obs[mask]
            row = {**base, 'period': pname, 'n_points': n,
                   'pain_sd': float(np.std(o)) if n else np.nan,
                   't_start': float(t[mask][0]) if n else np.nan,
                   't_end': float(t[mask][-1]) if n else np.nan}
            if n >= MIN_POINTS:
                m2 = fit_metrics(o, p2[mask])
                row.update({'r_eq2': m2['r'], 'r2_eq2': m2['r2'], 'mse_eq2': m2['mse'],
                            'obs_range': float(o.max() - o.min())})
                if p1 is not None:
                    m1 = fit_metrics(o, p1[mask])
                    row.update({'r_eq1': m1['r'], 'r2_eq1': m1['r2'],
                                'mse_eq1': m1['mse']})
            out.append(row)

    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(uids)}')

P = pd.DataFrame(out)
P['visit'] = P['trial_num'] // 100
P['trial_in_visit'] = P['trial_num'] % 100
P['trial_order'] = (P.groupby(['subject_uid', 'visit'])['trial_in_visit']
                    .rank(method='dense').astype(int))
P.to_csv(OUT_CSV, index=False)

print(f'\n{len(P):,} rows (trial x period) -> {OUT_CSV}')
print(f'trials with no boundary record: {missing_bounds}')
print('\nrows per period:')
print(P.groupby('period').agg(
    n=('n_points', 'size'), median_points=('n_points', 'median'),
    scorable=('r_eq2', lambda x: x.notna().sum())).to_string())


#%%
# ------------------------------------------------------------------
# Does windowing change the picture?
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('FIT BY PERIOD -- whole trial vs the windows that matter')
print('=' * 70)

# A flat window gives an unstable correlation, so exclude those rather than
# let them add noise; the count is reported so the exclusion is visible.
ok = P[(P['pain_sd'] >= MIN_PAIN_SD) & P['r_eq2'].notna()].copy()
print(f'{len(ok):,}/{P["r_eq2"].notna().sum():,} scorable rows have '
      f'rating SD >= {MIN_PAIN_SD} VAS')

print('\nmedian r by period and trial type (Eq. 2 | Eq. 1):')
piv = (ok.groupby(['trial_type', 'period'])[['r_eq2', 'r_eq1']]
       .median().round(3).unstack('period'))
print(piv.to_string())

print('\nPeriod C only -- the offset analgesia window:')
c = ok[ok['period'] == 'C']
print(c.groupby('trial_type').agg(
    n=('r_eq2', 'size'), r_eq2=('r_eq2', 'median'), r_eq1=('r_eq1', 'median'),
    gain=('r_eq1', 'median')).assign(
    gain=lambda d: (c.groupby('trial_type')
                    .apply(lambda g: (g['r_eq1'] - g['r_eq2']).median()))
    ).round(3).to_string())

print('\nHow much worse is period C than the whole trial? (Eq. 2, median r)')
w = (ok.pivot_table(index=['subject_uid', 'trial_num', 'trial_type'],
                    columns='period', values='r_eq2'))
w = w.dropna(subset=['full', 'C'])
w['C_minus_full'] = w['C'] - w['full']
print(w.groupby('trial_type')[['full', 'C', 'C_minus_full']]
      .median().round(3).to_string())

# %%
