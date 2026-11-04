# -*- coding: utf-8 -*-
"""
Fit the simplified Cecchi 2012 model to the three sEEG patients (RCS08, RCS09,
PR08), with the same settings as Psychophysical_Modeling.py (differential
evolution, theta fitted, one parameter set per patient).

These fits exist only to seed fit_full_model.py, which starts its search from
the Eq. 2 solution, exactly as it does for the cohort. Run this, then
fit_full_model.py and per_trial_fits.py with COHORT = 'sEEG'.

Outputs go to model_fit_results/sEEG/ and figures/sEEG/. Keeping them out of
the top-level folders matters: the cohort scripts pick their inputs with
non-recursive globs like '*model_fits_subject_*.csv', which would otherwise
pick up a patient file as the newest cohort fit.

Input is sEEG_traces_1Hz.pkl from
alter_collab_analysis/data_pipeline/build_sEEG_traces.py.

Author: Lucille Johnston
Updated: 11/3/26
"""
#%%
import os
import pickle
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import *
import numpy as np
import pandas as pd

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/sEEG_traces_1Hz.pkl')
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, 'model_fit_results', 'sEEG') + '/'
FIG_PATH = os.path.join(HERE, 'figures', 'sEEG') + '/'

OPTIMIZER = 'de'
OPTIMIZE_THETA = True

timestamp = datetime.now().strftime('%Y%m%d')
RESULTS_FILE = f'{RESULTS_PATH}sEEG_optimization_results_{timestamp}.pkl'
SUBJECT_CSV = f'{RESULTS_PATH}sEEG_model_fits_subject_{timestamp}.csv'
TRIAL_CSV = f'{RESULTS_PATH}sEEG_model_fits_trial_{timestamp}.csv'

data_df = pd.read_pickle(DATA_PATH)
print(f'Loaded {len(data_df):,} rows, {data_df["subject_uid"].nunique()} patients, '
      f'{data_df.groupby(["subject_uid", "trial_num"]).ngroups} trials')


#%%
# ------------------------------------------------------------------
# Fit each patient
# ------------------------------------------------------------------
optimization_results = {}
for uid, subject_data in data_df.groupby('subject_uid'):
    print(f'\n=== {uid} ===')
    best_params, best_result = optimize_cecchi_simplified(
        subject_data, threshold=None, initial_params=None,
        optimizer=OPTIMIZER, verbose=False)
    if best_params is None:
        print(f'  optimization failed')
        continue

    meta = subject_data.iloc[0]
    for col in ['dataset', 'site', 'study', 'group_label']:
        best_params[col] = meta[col]
    best_params['subject'] = uid
    optimization_results[uid] = {'params': best_params, 'result': best_result,
                                 'timestamp': datetime.now().isoformat()}
    print(f'  alpha_bar={best_params["alpha_bar"]:.4f}  '
          f'gamma_bar={best_params["gamma_bar"]:.4f}  theta={best_params["theta"]:.2f}  '
          f'r={best_params["r"]:.3f}  at_bounds={best_params.get("at_bounds") or "-"}')

with open(RESULTS_FILE, 'wb') as f:
    pickle.dump({'optimization_results': optimization_results,
                 'config': {'optimizer': OPTIMIZER, 'optimize_theta': OPTIMIZE_THETA,
                            'traces_file': DATA_PATH, 'timestamp': timestamp}}, f)


#%%
# ------------------------------------------------------------------
# Flat tables, same layout as Psychophysical_Modeling.py writes
# ------------------------------------------------------------------
subject_rows, trial_rows = [], []
for uid, res in optimization_results.items():
    p = res['params']
    base = {'subject_uid': uid, 'subject': p.get('subject'), 'site': p.get('site'),
            'dataset': p.get('dataset'), 'study': p.get('study'),
            'group_label': p.get('group_label')}
    subject_rows.append({
        **base,
        'alpha_bar': p['alpha_bar'], 'gamma_bar': p['gamma_bar'], 'theta': p['theta'],
        'r': p.get('r'), 'r2': p.get('r2'), 'mse': p.get('mse'), 'sse': p.get('sse'),
        'n_trials': p.get('n_trials'), 'n_points': p.get('n_points'),
        'at_bounds': ','.join(p.get('at_bounds') or []),
    })
    for tf in p.get('trial_fits', []):
        trial_rows.append({**base, 'alpha_bar': p['alpha_bar'],
                           'gamma_bar': p['gamma_bar'], 'theta': p['theta'], **tf})

fit_summary = pd.DataFrame(subject_rows).sort_values('subject_uid')
trial_summary = pd.DataFrame(trial_rows).sort_values(['subject_uid', 'trial_num'])
trial_summary['visit'] = 0
trial_summary['trial_in_visit'] = trial_summary['trial_num']
trial_summary['trial_order'] = (trial_summary.groupby('subject_uid')['trial_num']
                                .rank(method='dense').astype(int))

fit_summary.to_csv(SUBJECT_CSV, index=False)
trial_summary.to_csv(TRIAL_CSV, index=False)
print(f'\nSaved\n  {SUBJECT_CSV}\n  {TRIAL_CSV}\n  {RESULTS_FILE}')


#%%
# ------------------------------------------------------------------
# Figures: every trial for every patient (only three of them)
# ------------------------------------------------------------------
for uid in optimization_results:
    n_trials = data_df.loc[data_df['subject_uid'] == uid, 'trial_num'].nunique()
    plot_subject_trials(data_df[data_df['subject_uid'] == uid],
                        optimization_results[uid]['params'],
                        subject_uid=uid, max_trials=n_trials, ncols=6,
                        save_path=f'{FIG_PATH}fit_{uid}_{timestamp}.png')

# %%
