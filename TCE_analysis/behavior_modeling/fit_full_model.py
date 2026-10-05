# -*- coding: utf-8 -*-
"""
Fit the FULL second-order Cecchi 2012 model (their Eq. 1) per subject.

    p''(t) = alpha*F(T,theta) - beta*p'(t) + gamma*(T'(t) - lambda)*p(t)

Psychophysical_Modeling.py fits Eq. 2, the first-order simplification. That
simplification assumes both beta >> 1 and lambda >> 1; the second assumption
collapses (T' - lambda) to a constant and so removes the model's sensitivity
to the RATE of temperature change -- which is the mechanism Cecchi describe as
producing offset analgesia. It is a plausible reason Eq. 2 fits offset trials
worst (median r 0.82 vs 0.91 for onset, paired t = 14.6 across 165 subjects),
and this script produces the comparison that tests it.

Runtime. The full model is stiff: a fitted beta near 37 puts a ~0.03s process
inside a model fitted to 1 Hz data, so integration needs a very small step.
Calling solve_ivp per trial would take ~790 hours for this dataset. Stepping
a subject's trials forward together in numpy (pack_trials / simulate_trials_full)
is ~765x faster, which brings the whole run to roughly 7-9 hours. Expect to
run this overnight. It checkpoints after every subject and skips work already
done, so an interrupted run resumes rather than restarting.

Author: Lucille Johnston
Updated: 9/24/26
"""
#%%
import glob
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (optimize_cecchi_full,
                                              seed_full_from_simplified)

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
TRACES_FILE = DATA_PATH + 'combined_traces_1Hz.pkl'

# Set to a small number for a pilot; None fits everyone.
N_SUBJECTS = None
POPSIZE = 12        # differential evolution population = POPSIZE * 5 params
MAXITER = 60        # generations; with POPSIZE this is ~3600 evaluations
SEED = 0

STAMP = f'{datetime.now():%Y%m%d}'
CHECKPOINT = RESULTS_PATH + f'{STAMP}_full_model_checkpoint.pkl'
SUBJECT_CSV = RESULTS_PATH + f'{STAMP}_model_fits_subject_full.csv'
TRIAL_CSV = RESULTS_PATH + f'{STAMP}_model_fits_trial_full.csv'


#%%
# ------------------------------------------------------------------
# Load traces and the Eq. 2 fits used to seed the search
# ------------------------------------------------------------------
data_df = pd.read_pickle(TRACES_FILE)

# Most recent first-order fit. Petre 2017's published parameters are not used
# as a seed -- see seed_full_from_simplified() for why.
simple_csv = sorted(glob.glob(RESULTS_PATH + '*model_fits_subject_*.csv'))
simple_csv = [f for f in simple_csv if 'full' not in os.path.basename(f)][-1]
simple = pd.read_csv(simple_csv).set_index('subject_uid')
print(f'traces : {os.path.basename(TRACES_FILE)} '
      f'({data_df["subject_uid"].nunique()} subjects, '
      f'{data_df.groupby(["subject_uid", "trial_num"]).ngroups:,} trials)')
print(f'seeds  : {os.path.basename(simple_csv)} ({len(simple)} subjects)')

subjects_to_fit = sorted(simple.index)
if N_SUBJECTS is not None:
    subjects_to_fit = subjects_to_fit[:N_SUBJECTS]

results = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT, 'rb') as f:
        results = pickle.load(f).get('results', {})
    print(f'resuming: {len(results)} subjects already fitted')

print(f'\n{"="*60}')
print(f'FULL SECOND-ORDER MODEL (Cecchi 2012 Eq. 1)')
print(f'{"="*60}')
print(f'subjects to fit : {len(subjects_to_fit)} ({len(results)} done)')
print(f'optimiser       : differential evolution, popsize={POPSIZE}, maxiter={MAXITER}')
print(f'checkpoint      : {os.path.basename(CHECKPOINT)}')
print(f'{"="*60}\n')


#%%
# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------
t_start = time.time()
for i, uid in enumerate(subjects_to_fit):
    if uid in results:
        continue

    sd = data_df[data_df['subject_uid'] == uid]
    row = simple.loc[uid]
    seed_params = seed_full_from_simplified(row['alpha_bar'], row['gamma_bar'],
                                            row['theta'])

    print(f'[{i+1}/{len(subjects_to_fit)}] {uid} '
          f'({sd["trial_num"].nunique()} trials)')
    t0 = time.time()
    try:
        best, _ = optimize_cecchi_full(sd, initial_params=seed_params,
                                       popsize=POPSIZE, maxiter=MAXITER,
                                       seed=SEED, verbose=True)
    except Exception as e:
        print(f'    ❌ {e}')
        continue
    if best is None:
        print('    ❌ fit failed')
        continue

    best['dataset'] = row['dataset']
    best['study'] = row['study']
    best['group_label'] = row['group_label']
    # the Eq. 2 fit for this subject, so the two models can be compared
    # without re-merging later
    best['r_simplified'] = row.get('r', np.nan)
    best['alpha_bar_simplified'] = row['alpha_bar']
    best['gamma_bar_simplified'] = row['gamma_bar']
    best['theta_simplified'] = row['theta']
    results[uid] = best

    elapsed = time.time() - t_start
    done = len([u for u in subjects_to_fit if u in results])
    remaining = (len(subjects_to_fit) - done) * elapsed / max(done, 1)
    print(f'    {time.time()-t0:.0f}s  |  {done}/{len(subjects_to_fit)} done, '
          f'~{remaining/3600:.1f}h left\n')

    with open(CHECKPOINT, 'wb') as f:
        pickle.dump({'results': results,
                     'config': {'popsize': POPSIZE, 'maxiter': MAXITER,
                                'seed': SEED, 'traces': TRACES_FILE,
                                'seed_csv': simple_csv, 'stamp': STAMP}}, f)

print(f'\nfitted {len(results)} subjects in {(time.time()-t_start)/3600:.2f} h')


#%%
# ------------------------------------------------------------------
# Write the flat tables
# ------------------------------------------------------------------
subject_rows, trial_rows = [], []
for uid, p in results.items():
    base = {'subject_uid': uid, 'dataset': p.get('dataset'),
            'study': p.get('study'), 'group_label': p.get('group_label')}
    subject_rows.append({
        **base,
        'alpha': p['alpha'], 'beta': p['beta'], 'gamma': p['gamma'],
        'lam': p['lam'], 'theta': p['theta'],
        'r': p['r'], 'r2': p['r2'], 'mse': p['mse'], 'sse': p['sse'],
        'n_trials': p['n_trials'], 'n_points': p['n_points'],
        'at_bounds': ','.join(p.get('at_bounds') or []),
        # matched Eq. 2 result for the model comparison
        'r_simplified': p.get('r_simplified'),
        'alpha_bar_simplified': p.get('alpha_bar_simplified'),
        'gamma_bar_simplified': p.get('gamma_bar_simplified'),
        'theta_simplified': p.get('theta_simplified'),
    })
    for tf in p.get('trial_fits', []):
        trial_rows.append({**base, 'alpha': p['alpha'], 'beta': p['beta'],
                           'gamma': p['gamma'], 'lam': p['lam'],
                           'theta': p['theta'], **tf})

fits = pd.DataFrame(subject_rows).sort_values('subject_uid')
trials_out = pd.DataFrame(trial_rows).sort_values(['subject_uid', 'trial_num'])
trials_out['visit'] = trials_out['trial_num'] // 100
trials_out['trial_in_visit'] = trials_out['trial_num'] % 100
trials_out['trial_order'] = (trials_out.groupby(['subject_uid', 'visit'])
                             ['trial_in_visit'].rank(method='dense').astype(int))

fits.to_csv(SUBJECT_CSV, index=False)
trials_out.to_csv(TRIAL_CSV, index=False)
print(f'\n{SUBJECT_CSV}\n{TRIAL_CSV}')

print('\n=== FULL vs SIMPLIFIED ===')
print(f'median r, Eq. 1 (full)       : {fits["r"].median():.3f}')
print(f'median r, Eq. 2 (simplified) : {fits["r_simplified"].median():.3f}')
better = (fits['r'] > fits['r_simplified']).mean()
print(f'subjects where the full model wins: {better:.1%}')
print(f'\nsubjects with a parameter on a bound: '
      f'{(fits["at_bounds"] != "").sum()}/{len(fits)}')
print(fits[fits['at_bounds'] != '']['at_bounds'].value_counts().to_string())

print('\nfitted parameters (median by dataset):')
print(fits.groupby('dataset')[['alpha', 'beta', 'gamma', 'lam', 'theta']]
      .median().round(4).to_string())

print('\nper-trial r by trial type -- the offset comparison this run exists for:')
print(trials_out.groupby('trial_type')['r'].agg(
    n='size', median='median').round(3).to_string())

# %%
