# -*- coding: utf-8 -*-
"""
Fit the simplified Cecchi 2012 model to each trial separately, and ask whether
the fitted parameters drift across a session.

Psychophysical_Modeling.py fits one parameter set per subject across all of
that subject's trials. That asks "what dynamics describe this person?". This
script asks a different question: "do those dynamics change as the session goes
on?" -- which is a direct test of non-stationarity, rather than an indirect one
via fit quality.

Why this is a better test of drift than the alternatives:

  - Looking at per-trial residuals from a single whole-subject fit (as
    analyze_model_fits.py Q3 does) is confounded, because the optimiser has
    already spread its error to minimise total MSE. That biases toward a U
    shape rather than a trend.
  - Leave-one-trial-out cross-validation does not fix this. Every fold trains
    on "all the other trials", so every fold's parameters land near the
    subject's session average. If parameters drift steadily, then predicting
    the first trial and predicting the last trial both fail by about the same
    amount, and the result is again a U shape. LOTO is symmetric in time, so
    it cannot recover the direction of a drift. (It is still worth running as
    a generalisation measure -- just not for this.)

Fitting each trial independently has neither problem: there is one estimate
per trial, and their order is meaningful.

theta is held at the subject-level fitted value rather than refitted per
trial. It is a pain threshold, a property of the person, and a single ~45s
trace does not identify it well once alpha and gamma are also free. Set
FIT_THETA_PER_TRIAL = True to test threshold drift specifically.

Author: Lucille Johnston
Updated: 9/20/26
"""
#%%
import glob
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (prepare_trials_for_optimization,
                                              simulate_trials_analytic,
                                              fit_metrics)

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
TRACES_FILE = DATA_PATH + 'combined_traces_1Hz.pkl'

FIT_THETA_PER_TRIAL = False   # see module docstring
N_PERTURBED_STARTS = 3        # extra local starts around the warm start
ALPHA_BOUNDS = (0.001, 50.0)
GAMMA_BOUNDS = (0.001, 3.0)
THETA_BOUNDS = (30.0, 52.0)


def fit_one_trial(trial, warm_start, fit_theta=False):
    """
    Fit alpha_bar (and gamma_bar, optionally theta) to a single trial.

    Warm-started from the subject's whole-session fit. That is what makes this
    affordable: a single trial's optimum sits near the subject's average, so a
    local search from there finds it, and the global search that
    Psychophysical_Modeling.py needs (to escape the flat region where theta
    exceeds the hottest stimulus) is unnecessary here. A few perturbed starts
    guard against the local search stalling.
    """
    obs = trial['pain']
    theta_fixed = warm_start['theta']

    bounds = [ALPHA_BOUNDS, GAMMA_BOUNDS] + ([THETA_BOUNDS] if fit_theta else [])

    def objective(x):
        params = {'alpha_bar': x[0], 'gamma_bar': x[1],
                  'theta': x[2] if fit_theta else theta_fixed}
        try:
            pred = simulate_trials_analytic(params, [trial])[0]
            mse = np.mean((obs - pred) ** 2)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    x0 = [warm_start['alpha_bar'], warm_start['gamma_bar']]
    if fit_theta:
        x0.append(theta_fixed)

    starts = [x0]
    rng = np.random.default_rng(abs(hash(trial['trial_num'])) % 2**32)
    for _ in range(N_PERTURBED_STARTS):
        starts.append([float(np.clip(v * rng.uniform(0.4, 2.5), b[0], b[1]))
                       for v, b in zip(x0, bounds)])

    best = None
    for s in starts:
        res = minimize(objective, s, method='L-BFGS-B', bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res

    params = {'alpha_bar': best.x[0], 'gamma_bar': best.x[1],
              'theta': best.x[2] if fit_theta else theta_fixed}
    pred = simulate_trials_analytic(params, [trial])[0]
    at_bounds = [n for n, v, (lo, hi) in
                 zip(['alpha_bar', 'gamma_bar', 'theta'], best.x, bounds)
                 if abs(v - lo) < 1e-6 * max(1.0, abs(lo))
                 or abs(v - hi) < 1e-6 * max(1.0, abs(hi))]
    return params, pred, at_bounds


#%%
# ------------------------------------------------------------------
# Fit every trial
# ------------------------------------------------------------------
data_df = pd.read_pickle(TRACES_FILE)

subject_csv = sorted(glob.glob(RESULTS_PATH + 'model_fits_subject_*.csv'))[-1]
subject_fits = pd.read_csv(subject_csv).set_index('subject_uid')
print(f'warm starts from {os.path.basename(subject_csv)} '
      f'({len(subject_fits)} subjects)')

rows = []
t0 = time.time()
uids = sorted(subject_fits.index)
for i, uid in enumerate(uids):
    warm = subject_fits.loc[uid]
    sd = data_df[data_df['subject_uid'] == uid]
    trials = prepare_trials_for_optimization(sd)

    for trial in trials:
        # A trial with a flat rating carries no information about the
        # dynamics, and its correlation is undefined.
        if np.std(trial['pain']) == 0:
            continue
        params, pred, at_bounds = fit_one_trial(
            trial, warm, fit_theta=FIT_THETA_PER_TRIAL)
        rows.append({
            'subject_uid': uid,
            'dataset': warm['dataset'], 'study': warm['study'],
            'group_label': warm['group_label'],
            'trial_num': trial['trial_num'], 'trial_type': trial['trial_type'],
            **params,
            # the subject-level values, for comparison
            'alpha_subject': warm['alpha_bar'], 'gamma_subject': warm['gamma_bar'],
            'at_bounds': ','.join(at_bounds), 'n_points': len(pred),
            **fit_metrics(trial['pain'], pred),
        })

    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f'  {i+1}/{len(uids)} subjects, {len(rows):,} trials, '
              f'{el:.0f}s elapsed, ~{el/(i+1)*(len(uids)-i-1):.0f}s left')

pt = pd.DataFrame(rows)
pt['visit'] = pt['trial_num'] // 100
pt['trial_in_visit'] = pt['trial_num'] % 100
pt['trial_order'] = (pt.groupby(['subject_uid', 'visit'])['trial_in_visit']
                     .rank(method='dense').astype(int))

OUT = RESULTS_PATH + f'{datetime.now():%Y%m%d}_per_trial_fits_{"freetheta" if FIT_THETA_PER_TRIAL else "fixedtheta"}.csv'
pt.to_csv(OUT, index=False)
print(f'\n{len(pt):,} trials fitted in {time.time()-t0:.0f}s -> {OUT}')


#%%
# ------------------------------------------------------------------
# Does per-trial fit beat the one-set-per-subject fit?
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('PER-TRIAL vs PER-SUBJECT FIT QUALITY')
print('=' * 70)
trial_csv = sorted(glob.glob(RESULTS_PATH + 'model_fits_trial_*.csv'))[-1]
whole = pd.read_csv(trial_csv)[['subject_uid', 'trial_num', 'r']].rename(
    columns={'r': 'r_subject_fit'})
cmp = pt.merge(whole, on=['subject_uid', 'trial_num'], how='left')
print(f"median r, one set per subject : {cmp['r_subject_fit'].median():.3f}")
print(f"median r, one set per trial   : {cmp['r'].median():.3f}")
print(f"median per-trial gain         : +{(cmp['r']-cmp['r_subject_fit']).median():.3f}")
print('\nNote: per-trial fitting must score higher -- it has many more free '
      'parameters.\nThe useful part is not the gain but whether the parameters '
      'move systematically.')


#%%
# ------------------------------------------------------------------
# Do the parameters drift across a session?
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('PARAMETER DRIFT ACROSS TRIALS')
print('=' * 70)

clean = pt[pt['at_bounds'] == ''].copy()
print(f'{len(clean):,}/{len(pt):,} trials with no parameter on a bound')

# alpha and gamma are positive rate constants and right-skewed, so they are
# modelled on a log scale; a coefficient then reads as proportional change
# per trial.
clean['log_alpha'] = np.log(clean['alpha_bar'])
clean['log_gamma'] = np.log(clean['gamma_bar'])
clean['trial_type_c'] = clean['trial_type'].astype(str)

print('\nmedian parameter by position in session (first 12 trials):')
for p in ['alpha_bar', 'gamma_bar']:
    print(f'\n  {p}:')
    print(clean[clean['trial_order'] <= 12]
          .groupby(['dataset', 'trial_order'])[p].median().unstack().round(2).to_string())

# trial_type is in the model because trial order and trial type are not fully
# crossed -- in cLBP, for instance, every session opens with an offset trial.
# Random slopes because subjects may drift at different rates.
print('\n' + '-' * 70)
for label, col in [('alpha_bar (log)', 'log_alpha'), ('gamma_bar (log)', 'log_gamma')]:
    m = smf.mixedlm(f'{col} ~ trial_order + C(dataset) + C(trial_type_c)', clean,
                    groups=clean['subject_uid'], re_formula='~trial_order').fit()
    slope, p = m.params['trial_order'], m.pvalues['trial_order']
    pct = (np.exp(slope) - 1) * 100
    verdict = ('no reliable drift' if p >= 0.05 else
               f'{"INCREASES" if slope > 0 else "DECREASES"} '
               f'{abs(pct):.2f}% per trial')
    print(f'{label:16s} slope = {slope:+.5f} (p = {p:.3g})  -> {verdict}')
    if p < 0.05:
        print(f'{"":16s} over 12 trials: {(np.exp(slope*12)-1)*100:+.1f}%')

# Per-subject slopes: a group-level effect can be carried by a few subjects,
# so check how many individuals move in the same direction.
print('\nper-subject slopes (sign test against no drift):')
for label, col in [('alpha', 'log_alpha'), ('gamma', 'log_gamma')]:
    slopes = []
    for uid, g in clean.groupby('subject_uid'):
        if g['trial_order'].nunique() >= 5:
            slopes.append(np.polyfit(g['trial_order'], g[col], 1)[0])
    slopes = np.array(slopes)
    from scipy import stats as st
    n_pos = (slopes > 0).sum()
    p_sign = st.binomtest(n_pos, len(slopes), 0.5).pvalue
    print(f'  {label:6s} {n_pos}/{len(slopes)} subjects increasing, '
          f'median slope {np.median(slopes):+.4f}, sign test p = {p_sign:.3g}')


#%%
# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, p, name in [(axes[0][0], 'alpha_bar', 'ᾱ  (drive)'),
                    (axes[0][1], 'gamma_bar', 'γ̄  (decay rate)')]:
    for ds, g in clean[clean['trial_order'] <= 24].groupby('dataset'):
        s = g.groupby('trial_order')[p].agg(['median', 'size'])
        s = s[s['size'] >= 20]
        ax.plot(s.index, s['median'], marker='o', ms=4, label=ds)
    ax.set(xlabel='trial position within session', ylabel=name,
           title=f'{name} across a session')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Parameters by trial type: if they differ, trial type must stay in any model
# of trial order, because the two are not fully crossed.
sns.boxplot(data=clean[clean['trial_type'].isin(
    ['offset', 'onset', 't1_hold', 't2_hold'])],
    x='trial_type', y='alpha_bar', ax=axes[1][0], color='lightsteelblue',
    fliersize=2)
axes[1][0].set(xlabel='', ylabel='ᾱ', title='ᾱ by trial type')
axes[1][0].set_yscale('log')

sns.boxplot(data=clean[clean['trial_type'].isin(
    ['offset', 'onset', 't1_hold', 't2_hold'])],
    x='trial_type', y='gamma_bar', ax=axes[1][1], color='lightsteelblue',
    fliersize=2)
axes[1][1].set(xlabel='', ylabel='γ̄', title='γ̄ by trial type')
axes[1][1].set_yscale('log')

plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_per_trial_parameter_drift.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
