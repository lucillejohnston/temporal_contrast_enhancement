# -*- coding: utf-8 -*-
"""
Fit the simplified Cecchi 2012 model to each trial separately, and ask whether
the fitted parameters drift across a session.

Psychophysical_Modeling.py fits one parameter set per subject across all of
that subject's trials. That asks "what dynamics describe this person?". This
script asks a different question: "do those dynamics change as the session goes
on?" -- which is a direct test of non-stationarity, rather than an indirect one
via fit quality.

theta is held at the subject-level fitted value rather than refitted per
trial. It is a pain threshold, a property of the person, and a single ~45s
trace does not identify it well once alpha and gamma are also free. Set
FIT_THETA_PER_TRIAL = True to test threshold drift specifically.

Author: Lucille Johnston
Updated: 11/02/26
"""
#%%
# --------------------------
# 
import glob
import os
import time

# Set before numpy imports: each worker gets one BLAS thread, so the inner
# threading does not contend with the outer parallelism across subjects.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
from scipy import stats as st
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (prepare_trials_for_optimization,
                                              pack_trials, full_predictions_flat,
                                              fit_metrics)

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
TRACES_FILE = DATA_PATH + 'combined_traces_1Hz.pkl'

# 'sEEG' fits the three sEEG patients instead, warm-started from the sEEG run
# of fit_full_model.py. Inputs, outputs and figures all live in an sEEG/
# subfolder: the globs below would otherwise mix the two cohorts' fit tables,
# and the date-only figure names would overwrite same-day cohort figures.
COHORT = 'alter_collab'   # 'alter_collab' or 'sEEG'
if COHORT == 'sEEG':
    TRACES_FILE = DATA_PATH + 'sEEG_traces_1Hz.pkl'
    RESULTS_PATH = RESULTS_PATH + 'sEEG/'
    FIG_PATH = FIG_PATH + 'sEEG/'

FIT_THETA_PER_TRIAL = False   # see module docstring
N_PERTURBED_STARTS = 3        # extra local starts around the warm start

# The fit is a ~7 hour run. True loads the newest saved table for the current
# FIT_THETA_PER_TRIAL setting instead, so the analysis cells below can be run
# on their own. Set to False only to actually refit.
RELOAD_EXISTING = True

PARAM_LABEL = {'alpha': r'$\alpha$ (drive)', 'beta': r'$\beta$ (damping)',
               'gamma': r'$\gamma$ (rate gain)',
               'lam': r'$\lambda$ (rate offset)',
               'theta': r'$\theta$ (threshold)'}
MAIN_TYPES = ['offset', 'onset', 't1_hold', 't2_hold']
DATASET_COLORS = {'plosONE': '#8FA8B8', 'kneeOA': '#A7B29A', 'cLBP': '#AFA3B5'}

# Set to an integer to fit only that many subjects -- for timing and for
# checking how often each parameter lands on a bound before committing to the
# whole run. None fits everybody.
N_SUBJECTS = None

# 14 cores on this machine; 12 leaves headroom so the laptop stays usable.
N_JOBS = 12

# Fit forearm only. kneeOA was stimulated at forearm and knee and both have
# subject-level fits, but every downstream analysis is forearm-only, and the
# knee fits would add ~40% to a run already measured in hours.
FIT_SITE = 'forearm'

# Same bounds as optimize_cecchi_full(), deliberately: a per-trial fit that
# used a different box would not be comparable to the subject-level fit it is
# warm-started from. The lam ceiling in particular is meaningful rather than
# practical -- see that function's docstring. A fit that lands there is the
# data preferring the simplified form, in which the rate term is constant.
PARAM_NAMES = ['alpha', 'beta', 'gamma', 'lam', 'theta']
FULL_BOUNDS = {'alpha': (0.001, 500.0), 'beta': (0.05, 60.0),
               'gamma': (1e-4, 50.0), 'lam': (1e-4, 50.0),
               'theta': (30.0, 52.0)}


def fit_one_trial(trial, warm_start, fit_theta=False):
    """
    Fit Eq. 1's alpha, beta, gamma, lam (and optionally theta) to one trial.

    Warm-started from the subject's whole-session Eq. 1 fit. That is what
    makes this affordable: a single trial's optimum sits near the subject's
    average, so a local search from there finds it, and the differential
    evolution that optimize_cecchi_full() needs (to escape the flat region
    where theta exceeds the hottest stimulus) is unnecessary. A few perturbed
    starts guard against the local search stalling.

    Expect heavy bound-hitting on gamma and lam. optimize_cecchi_full()'s own
    docstring notes that the two enter only as gamma*lam and gamma, so they
    separate only where the rate term measurably matters -- and that beta's
    1/beta timescale is far below the 1 Hz sampling. Those cautions are for a
    fit using a whole session (~12 trials, ~540 points, 5 free parameters).
    Here there are ~45 points and 4. The bound-hit rates this script reports
    are therefore a result in their own right, not just diagnostics.
    """
    names = [n for n in PARAM_NAMES if n != 'theta' or fit_theta]
    bounds = [FULL_BOUNDS[n] for n in names]
    theta_fixed = float(warm_start['theta'])
    packed = pack_trials([trial])
    obs = packed['observed_flat']

    def unpack(x):
        p = dict(zip(names, x))
        p.setdefault('theta', theta_fixed)
        return p

    def objective(x):
        try:
            pred = full_predictions_flat(unpack(x), packed)
            mse = np.mean((obs - pred) ** 2)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    x0 = [float(np.clip(warm_start[n], *FULL_BOUNDS[n])) for n in names]

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

    params = unpack(best.x)
    pred = full_predictions_flat(params, packed)
    at_bounds = [n for n, v, (lo, hi) in zip(names, best.x, bounds)
                 if abs(v - lo) < 1e-6 * max(1.0, abs(lo))
                 or abs(v - hi) < 1e-6 * max(1.0, abs(hi))]
    return params, pred, at_bounds


#%%
# ------------------------------------------------------------------
# Fit every trial, or reload the saved table
#
# Everything below this cell works off `pt` alone -- no later cell touches
# data_df, subject_fits or any fit-loop variable -- so RELOAD_EXISTING can
# skip the whole fit and the analysis cells still run unchanged.
# ------------------------------------------------------------------
def fit_one_subject(uid, subject_data, warm, fit_theta=False):
    """Every trial for one subject. Split out so it can run in a worker."""
    out = []
    for trial in prepare_trials_for_optimization(subject_data):
        # A trial with a flat rating carries no information about the dynamics.
        if np.std(trial['pain']) == 0:
            continue
        params, pred, at_bounds = fit_one_trial(trial, warm, fit_theta=fit_theta)
        out.append({
            'subject_uid': uid,
            'subject': subject_data['subject_uid'].iloc[0],
            'site': subject_data['site'].iloc[0],
            'dataset': warm['dataset'], 'study': warm['study'],
            'group_label': warm['group_label'],
            'trial_num': trial['trial_num'], 'trial_type': trial['trial_type'],
            **params,
            # the subject-level values, for comparison
            **{f'{n}_subject': warm[n] for n in PARAM_NAMES},
            'at_bounds': ','.join(at_bounds), 'n_points': len(pred),
            **fit_metrics(trial['pain'], pred),
        })
    return out


if RELOAD_EXISTING:
    # trial_order, visit, rmse and at_bounds are all written into the CSV by
    # the fit branch below, so a plain read gives the analysis cells every
    # column they expect. TEST runs carry a _TEST<n> suffix and so never match.
    _tag = 'freetheta' if FIT_THETA_PER_TRIAL else 'fixedtheta'
    _saved = sorted(glob.glob(RESULTS_PATH + f'*_per_trial_fits_full_{_tag}.csv'))
    if not _saved:
        raise FileNotFoundError(
            f'RELOAD_EXISTING is True but no *_per_trial_fits_full_{_tag}.csv '
            f'exists in {RESULTS_PATH}. Set RELOAD_EXISTING = False to fit.')
    OUT = _saved[-1]
    pt = pd.read_csv(OUT)
    print(f'RELOAD_EXISTING: {len(pt):,} trials loaded from '
          f'{os.path.basename(OUT)} -- fit skipped')
else:
    data_df = pd.read_pickle(TRACES_FILE)

    # The fitting unit is (subject, site), matching Psychophysical_Modeling.py:
    # forearm keeps the plain subject id, any other site carries an '@site'
    # suffix.
    #
    # This matters even though each trial is fitted on its own. alpha and gamma
    # are per-trial and so are site-correct whatever the keying, but THETA is
    # not -- it is held at the subject-level value, and the warm start comes
    # from there too. Keyed on subject_uid alone, a kneeOA trial would inherit a
    # theta fitted across both forearm and knee. Those thetas differ (44.20 vs
    # 44.73, p=3.5e-07), and since the forcing is F = T - theta with F only
    # 2-3 degC, a quarter-degree error is ~10% of the drive, which alpha then
    # absorbs.
    if 'site' not in data_df.columns:
        data_df['site'] = 'forearm'
    data_df['site'] = data_df['site'].fillna('forearm')
    data_df['fit_unit'] = np.where(data_df['site'] == 'forearm',
                                   data_df['subject_uid'],
                                   data_df['subject_uid'] + '@' + data_df['site'])

    # Warm starts come from the Eq. 1 subject fits, not the Eq. 2 ones -- this
    # script fits the full model now, so the simplified parameters are not a
    # starting point for it and its alpha is on a different scale entirely.
    subject_csv = sorted(glob.glob(RESULTS_PATH + '*model_fits_subject_full.csv'))[-1]
    subject_fits = pd.read_csv(subject_csv).set_index('subject_uid')
    print(f'warm starts from {os.path.basename(subject_csv)} '
          f'({len(subject_fits)} subjects, Eq. 1)')

    t0 = time.time()
    if FIT_SITE is not None:
        n_before = data_df['fit_unit'].nunique()
        data_df = data_df[data_df['site'] == FIT_SITE]
        print(f'  site filter ({FIT_SITE}): {n_before} -> '
              f'{data_df["fit_unit"].nunique()} fit units')
    available = set(data_df['fit_unit'].unique())
    uids = sorted(set(subject_fits.index) & available)
    missing = sorted(set(subject_fits.index) - available)
    if missing:
        print(f'  {len(missing)} subject fits have no matching traces, skipped')
    if N_SUBJECTS is not None:
        uids = uids[:N_SUBJECTS]
        print(f'  TEST RUN: {len(uids)} subjects only '
              f'(set N_SUBJECTS = None for all)')

    # Subjects are independent, so they fan out across cores. Serially this is a
    # ~24h run at ~20s per trial; the full model integrates with RK4 substeps
    # that scale with beta, which is what makes it so much slower than Eq. 2's
    # closed form. loky rather than the default spawn: numpy here is built on
    # Accelerate, and forking a process with a threaded BLAS can deadlock.
    #
    # Each worker is handed only that subject's rows, not the whole traces
    # table. BLAS threads are pinned to 1 per worker -- the arrays are small
    # enough that inner threading only contends with the outer parallelism.
    jobs = [(uid, data_df[data_df['fit_unit'] == uid].copy(),
             subject_fits.loc[uid].to_dict()) for uid in uids]
    print(f'fitting {len(jobs)} subjects on {N_JOBS} workers...')
    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
        delayed(fit_one_subject)(uid, sd, warm, FIT_THETA_PER_TRIAL)
        for uid, sd, warm in jobs)

    pt = pd.DataFrame([r for sub in results for r in sub])
    pt['visit'] = pt['trial_num'] // 100
    pt['trial_in_visit'] = pt['trial_num'] % 100
    pt['trial_order'] = (pt.groupby(['subject_uid', 'visit'])['trial_in_visit']
                         .rank(method='dense').astype(int))

    pt['rmse'] = np.sqrt(pt['mse'])

    tag = 'freetheta' if FIT_THETA_PER_TRIAL else 'fixedtheta'
    if N_SUBJECTS is not None:
        tag += f'_TEST{N_SUBJECTS}'
    OUT = RESULTS_PATH + f'{datetime.now():%Y%m%d}_per_trial_fits_full_{tag}.csv'
    pt.to_csv(OUT, index=False)
    elapsed = time.time() - t0
    print(f'\n{len(pt):,} trials fitted in {elapsed:.0f}s '
          f'({elapsed/max(len(pt),1):.2f}s per trial) -> {OUT}')

    if N_SUBJECTS is not None:
        full_n = len(set(subject_fits.index) & available)
        # Wall time is governed by how many WAVES of N_JOBS subjects are needed,
        # not by the subject count alone. A test with fewer subjects than
        # workers leaves most of them idle, so its elapsed time is simply the
        # slowest subject and says nothing about throughput -- which is why the
        # estimate below is only trustworthy once N_SUBJECTS >= N_JOBS.
        waves_tested = max(1, int(np.ceil(len(uids) / N_JOBS)))
        per_wave = elapsed / waves_tested
        waves_full = int(np.ceil(full_n / N_JOBS))
        print(f'\nprojected full run: {full_n} subjects in {waves_full} waves '
              f'of {N_JOBS} -> ~{per_wave * waves_full / 3600:.1f} hours')
        if len(uids) < N_JOBS:
            print(f'  (rough -- only {len(uids)} of {N_JOBS} workers were busy. '
                  f'Set N_SUBJECTS = {N_JOBS} to measure throughput properly.)')

# The headline diagnostic for this script. gamma and lam separate only where
# the rate term measurably matters, and a single ~45s trial gives it little to
# work with, so these rates say how much of the per-trial parameter table is
# an actual estimate rather than the edge of the search box.
print('\nfraction of trials with each parameter on a bound:')
fitted_names = [n for n in PARAM_NAMES if n != 'theta' or FIT_THETA_PER_TRIAL]
for n in fitted_names:
    hit = pt['at_bounds'].fillna('').str.split(',').apply(lambda s: n in s)
    print(f'  {n:6s} {hit.mean()*100:5.1f}%')
print(f'  any    {(pt["at_bounds"].fillna("") != "").mean()*100:5.1f}%')


#%%
# ------------------------------------------------------------------
# Does per-trial fit beat the one-set-per-subject fit?
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('PER-TRIAL vs PER-SUBJECT FIT QUALITY (both Eq. 1)')
print('=' * 70)
# model_fits_trial_full.csv, not model_fits_trial_*.csv: the comparison has to
# be Eq. 1 scored per trial from a SUBJECT-level fit against Eq. 1 scored per
# trial from a PER-TRIAL fit. The non-full file is Eq. 2 and would make this a
# model comparison rather than a fitting-unit comparison.
trial_csv = sorted(glob.glob(RESULTS_PATH + '*model_fits_trial_full.csv'))[-1]
whole = pd.read_csv(trial_csv)[['subject_uid', 'trial_num', 'mse']].rename(
    columns={'mse': 'mse_subject_fit'})
cmp = pt.merge(whole, on=['subject_uid', 'trial_num'], how='left')
cmp['rmse_subject_fit'] = np.sqrt(cmp['mse_subject_fit'])
print(f'warm-start scores from {os.path.basename(trial_csv)}')
print(f"median RMSE, one set per subject : "
      f"{cmp['rmse_subject_fit'].median():.2f} VAS")
print(f"median RMSE, one set per trial   : {cmp['rmse'].median():.2f} VAS")
print(f"median improvement               : "
      f"{(cmp['rmse_subject_fit'] - cmp['rmse']).median():+.2f} VAS")
print('\nNote: per-trial fitting MUST score better -- it has four free '
      'parameters per trial\ninstead of five for the whole session. The useful '
      'part is not the gain but whether\nthe parameters move systematically.')

# Collapse to one number per subject so each person counts once, the same unit
# the per-subject violins in analyze_model_fits.py use. The merge above is
# complete (3,306/3,306 trials matched), so both columns rest on exactly the
# same trials and the pairing is not an artefact of differing coverage.
per_subj = (cmp.groupby(['subject_uid', 'dataset'])
            [['rmse_subject_fit', 'rmse']].median().reset_index())

n_better = int((per_subj['rmse'] < per_subj['rmse_subject_fit']).sum())
print(f'\nper subject (n = {len(per_subj)}): {n_better} fit better per trial, '
      f'median gain {(per_subj["rmse_subject_fit"] - per_subj["rmse"]).median():+.2f} VAS')

# Grey for the reference condition, colour for the one under test -- the same
# convention as KIND_COLORS in per_period_fits.py.
UNIT_LABELS = {'rmse_subject_fit': 'One set\nper subject',
               'rmse': 'One set\nper trial'}
UNIT_COLORS = {'One set\nper subject': '#CFCFCF',
               'One set\nper trial': '#2B6CA3'}
UNIT_ORDER = list(UNIT_COLORS)

long = per_subj.melt(id_vars=['subject_uid', 'dataset'],
                     value_vars=list(UNIT_LABELS),
                     var_name='fit_unit', value_name='median_rmse')
long['fit_unit'] = long['fit_unit'].map(UNIT_LABELS)

fig, ax = plt.subplots(figsize=(6.2, 5.8))

# One faint line per subject, centre to centre, drawn first so the violins
# sit on top. The violins are narrowed to .62 so the lines stay visible in the
# gap between them -- those lines are the point of the figure, they show
# whether the gain is uniform or carried by a handful of people.
for _, r in per_subj.iterrows():
    ax.plot([0, 1], [r['rmse_subject_fit'], r['rmse']],
            color='0.35', lw=.45, alpha=.22, zorder=1)

sns.violinplot(data=long, x='fit_unit', y='median_rmse', order=UNIT_ORDER,
               hue='fit_unit', palette=UNIT_COLORS, legend=False,
               cut=0, inner='box', linewidth=1.0, saturation=1.0,
               width=.62, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.88)
    coll.set_zorder(3)

ax.set(xlabel='', ylabel='model error (RMSE, VAS) per subject',
       title='Fit per subject vs fit per trial')
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()   # lower RMSE is a better fit, so better reads as higher

low = long['median_rmse'].min()
span = long['median_rmse'].max() - low
for i, u in enumerate(UNIT_ORDER):
    vals = long.loc[long['fit_unit'] == u, 'median_rmse']
    ax.text(i, vals.min() - .012 * span, f'{vals.median():.1f}', va='bottom',
            ha='center', fontsize=9.5, color='0.15', zorder=4)
ax.set_ylim(long['median_rmse'].max() + .04 * span, low - .07 * span)

plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_unit_comparison.png',
            dpi=200, bbox_inches='tight')
plt.show()


#%%
# ------------------------------------------------------------------
# Per-trial parameters vs the one-set-per-subject parameters
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('PER-TRIAL vs PER-SUBJECT PARAMETERS')
print('=' * 70)

CMP_PARAMS = [n for n in PARAM_NAMES if n != 'theta']   # theta is held fixed

# Everything below is on log10. These parameters are positive and span several
# orders of magnitude (alpha runs 1e-3 to 5e2), so an absolute SD would be
# meaningless and would be dominated by whichever subjects happen to sit high.
# On a log scale an SD is a FOLD-spread: 0.3 is about 2-fold, 1.0 is 10-fold.
# That also puts the four parameters on one comparable axis despite their
# different units.
lg = pt.copy()

# The two combinations the data actually constrain, carried alongside the raw
# parameters. In the overdamped regime these fits sit in, Eq. 1 collapses to
#     p' = (alpha/beta) F - (gamma*lam/beta) p
# so the predicted trajectory depends on alpha/beta and gamma*lam/beta and NOT
# on alpha, beta, gamma or lam separately. Doubling alpha and beta together
# leaves the prediction, and therefore the MSE, unchanged. The objective
# surface is not a bowl with one bottom but a flat-floored trench, and an
# optimiser stops wherever it happens to land along the floor.
#
# That is why both quantities are plotted below. A per-trial fit that lands
# 6x away from the session fit in alpha, while matching it in alpha/beta, has
# not disagreed about the subject -- the two fits predict the same curve from
# different points on the same trench floor. Without the ratio panels the raw
# scatter in alpha and beta reads as instability that isn't there.
for suf in ('', '_subject'):
    lg['abar' + suf] = lg['alpha' + suf] / lg['beta' + suf]
    lg['gbar' + suf] = lg['gamma' + suf] * lg['lam' + suf] / lg['beta' + suf]
COMBOS = ['abar', 'gbar']
PLOT_QTY = CMP_PARAMS + COMBOS
# Titles are the bare symbol. The agreement statistic goes inside the panel in
# words instead, because 'rho' and 'shift 0.50x' both invite the wrong read:
# rho is rank-based over four decades, so it is near 1 even where the values
# disagree 18-fold.
SYMBOL = {'alpha': r'$\alpha$', 'beta': r'$\beta$', 'gamma': r'$\gamma$',
          'lam': r'$\lambda$', 'abar': r'$\alpha\,/\,\beta$',
          'gbar': r'$\gamma\lambda\,/\,\beta$'}

for n in PLOT_QTY:
    lg[f'lg_{n}'] = np.log10(lg[n])
    lg[f'lg_{n}_subject'] = np.log10(lg[f'{n}_subject'])

rows = []
for (uid, ds), g in lg.groupby(['subject_uid', 'dataset']):
    r = {'subject_uid': uid, 'dataset': ds, 'n_trials': len(g)}
    for n in PLOT_QTY:
        r[f'{n}_trial_med'] = g[f'lg_{n}'].median()
        r[f'{n}_within_sd'] = g[f'lg_{n}'].std(ddof=1)
        # constant within a subject by construction, so 'first' is the value
        r[f'{n}_subj'] = g[f'lg_{n}_subject'].iloc[0]
    rows.append(r)
ps = pd.DataFrame(rows)
print(f'{len(ps)} subjects, {ps["n_trials"].min()}-{ps["n_trials"].max()} '
      f'trials each')

# ---- 1. does the per-trial median land on the session value? --------------
print('\n1. MEDIAN per-trial parameter vs the session fit')
print(f'{"":8s} {"rho":>7s} {"p":>10s} {"median fold-shift":>19s}')
for n in CMP_PARAMS:
    rho, pv = st.spearmanr(ps[f'{n}_subj'], ps[f'{n}_trial_med'])
    fold = 10 ** (ps[f'{n}_trial_med'] - ps[f'{n}_subj']).median()
    print(f'{n:8s} {rho:+7.3f} {pv:10.3g} {fold:18.2f}x')
print('  rho near 0 means a subject\'s per-trial fits say nothing about the\n'
      '  session fit; fold-shift is how far the typical per-trial median sits\n'
      '  from it (1.0x = dead on).')

# ---- 2. how much does a subject's own parameter move trial to trial? ------
# Variance decomposition. "Between" is the spread of the session-level fits
# across subjects -- real individual differences. "Within" is the average
# spread of one subject's own trials around their own median. The ratio says
# what fraction of the variation in the per-trial table is signal about who
# the person is rather than trial-to-trial instability of the fit.
print('\n2. WITHIN-subject spread of per-trial parameters (log10 SD)')
print(f'{"":8s} {"within":>7s} {"between":>8s} {"fold":>7s} {"% between":>10s}')
spread = {}
for n in CMP_PARAMS:
    within_var = np.nanmean(ps[f'{n}_within_sd'] ** 2)
    between_var = ps[f'{n}_subj'].var(ddof=1)
    w, b = np.sqrt(within_var), np.sqrt(between_var)
    pct_between = 100 * between_var / (between_var + within_var)
    spread[n] = {'within': w, 'between': b, 'pct_between': pct_between}
    print(f'{n:8s} {w:7.3f} {b:8.3f} {10**w:6.1f}x {pct_between:9.1f}%')
print('  within  = typical fold-spread of ONE subject across their own trials\n'
      '  between = spread of the session fits across subjects\n'
      '  % between = share of total variance that is real individual difference')

# Same numbers with bound-pinned trials dropped. A parameter sitting on a
# bound is not an estimate, and many trials pinned at the same bound would
# shrink the apparent within-subject spread, so this checks the conclusion is
# not an artefact of that.
clean = lg[lg['at_bounds'].fillna('') == '']
if len(clean) > 100:
    cw = clean.groupby('subject_uid').filter(lambda g: len(g) >= 3)
    print(f'\n  bound-free trials only ({len(cw):,} trials, '
          f'{cw["subject_uid"].nunique()} subjects with >=3):')
    for n in CMP_PARAMS:
        wv = np.nanmean(cw.groupby('subject_uid')[f'lg_{n}']
                        .std(ddof=1) ** 2)
        print(f'    {n:8s} within {np.sqrt(wv):.3f} ({10**np.sqrt(wv):.1f}x)')

# ---- 3. do the two fits agree on what the data constrain? -----------------
print('\n3. The same comparison for the identifiable combinations')
print(f'{"":14s} {"fold-shift":>11s} {"within 2x":>10s} {"IQR of disagreement":>21s}')
for n in PLOT_QTY:
    d = ps[f'{n}_trial_med'] - ps[f'{n}_subj']
    plain = {'abar': 'alpha/beta', 'gbar': 'gamma*lam/beta'}.get(n, n)
    print(f'{plain:14s} {10 ** d.median():10.2f}x '
          f'{100 * (d.abs() < np.log10(2)).mean():9.0f}% '
          f'{10 ** (d.quantile(.75) - d.quantile(.25)):20.1f}x')
print('  alpha and beta move together along a direction the data cannot see,\n'
      '  so they disagree individually while their ratio does not.')

# ---- plots ----------------------------------------------------------------
# Four raw parameters then the two ratios. The point of the figure is the
# contrast between the two halves: scatter blown out on the left, collapsed
# onto the identity line on the right, from the same fits.
fig, axes = plt.subplots(1, len(PLOT_QTY),
                         figsize=(3.1 * len(PLOT_QTY), 4.0))
for i, (ax, n) in enumerate(zip(np.atleast_1d(axes), PLOT_QTY)):
    # One point per subject: their session-level fit against the median of
    # their own per-trial fits. On the dashed identity line means the typical
    # trial lands where the session fit put that person.
    ax.scatter(10 ** ps[f'{n}_subj'], 10 ** ps[f'{n}_trial_med'],
               s=17, color='#2B6CA3', alpha=.55, linewidths=0)
    both = np.concatenate([ps[f'{n}_subj'], ps[f'{n}_trial_med']])
    lim = [10 ** (both.min() - .15), 10 ** (both.max() + .15)]
    ax.plot(lim, lim, color='0.35', lw=.9, ls='--', zorder=0)
    d = ps[f'{n}_trial_med'] - ps[f'{n}_subj']
    ax.text(.04, .96, f'{100 * (d.abs() < np.log10(2)).mean():.0f}% of subjects\n'
                      f'agree within 2x',
            transform=ax.transAxes, ha='left', va='top', fontsize=8,
            color='#1B4B73')
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim,
           xlabel='fitted per subject', title=SYMBOL[n])
    ax.title.set_fontsize(15)
    if i == 0:
        ax.set_ylabel('median of per-trial fits')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Per-trial vs per-subject parameters.   The four raw parameters '
             'disagree; the two combinations the data constrain do not.',
             y=1.03)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_per_trial_vs_subject_params.png',
            dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# Trial-to-trial spread of each subject's own parameters
# ------------------------------------------------------------------
# Descriptive only: how much does one person's fitted parameter move between
# their own trials. One dot is one subject within one trial type, and the
# value is simply the largest fit divided by the smallest, so a dot at 7x
# means that subject's alpha ranged over a factor of seven across their own
# trials of that type.
#
# The spread is computed WITHIN each of the four trial types and the resulting
# cells are then pooled into one distribution. That is not the same as taking
# the spread across all of a subject's trials at once: t1_hold and t2_hold are
# different temperatures and genuinely want different parameters, so a single
# pooled spread would fold that real between-type difference into what is
# meant to be trial-to-trial instability. Splitting first and pooling after
# keeps the quantity honest. A subject contributes one dot per trial type.
#
# Hold and stepped were plotted separately at one point and the two
# distributions were close enough that the split added nothing, so they are
# shown together.
#
# Cells need at least three trials: a max/min from two points is just those
# two points. The median cell has three, so no single dot carries much weight
# -- the shape of the swarm is the message, not any one subject.
KIND = {'offset': 'stepped', 'onset': 'stepped',
        't1_hold': 'hold', 't2_hold': 'hold'}
MIN_TRIALS_PER_CELL = 3
KIND_COLORS = {'hold': '#9AA7B0', 'stepped': '#2B6CA3'}

tt = lg[lg['trial_type'].isin(KIND)].copy()
tt['kind'] = tt['trial_type'].map(KIND)
cell = (tt.groupby(['subject_uid', 'trial_type', 'kind'])
        .agg(n_trials=('alpha', 'size'),
             # SD, not variance. Variance squares the units, and with cells
             # this skewed that turns every violin into a spike: alpha's
             # variance reaches 8e4 where its SD reaches 289. The SD is also
             # in the parameter's own units, so a value of 77 reads directly
             # as "this subject's alpha moved by about +/-77 between their
             # own trials".
             **{q: (q, lambda v: v.std(ddof=1)) for q in PLOT_QTY})
        .reset_index())
cell = cell[cell['n_trials'] >= MIN_TRIALS_PER_CELL]

print('\n' + '=' * 70)
print("TRIAL-TO-TRIAL SPREAD OF EACH SUBJECT'S OWN PARAMETERS")
print('=' * 70)
print(f'{len(cell)} subject x trial-type cells with >= {MIN_TRIALS_PER_CELL} '
      f'trials, from {cell["subject_uid"].nunique()} subjects '
      f'({(cell["kind"] == "hold").sum()} hold, '
      f'{(cell["kind"] == "stepped").sum()} stepped)')
print('\nSD of the parameter across a subject\'s own trials of one type:')
print(f'{"":16s} {"median":>12s} {"IQR":>26s}')
for q in PLOT_QTY:
    plain = {'abar': 'alpha/beta', 'gbar': 'gamma*lam/beta'}.get(q, q)
    v = cell[q]
    print(f'{plain:16s} {v.median():12.4g} '
          f'{v.quantile(.25):14.4g} - {v.quantile(.75):.4g}')

# Plotted as log10 of the ratio on a linear axis, with the ticks relabelled in
# plain multiples. Doing it this way rather than with a log y-axis keeps the
# swarm's collision detection working in the same units the dots are drawn in.
#
# The axis has to span about eight decades, which is itself the finding: the
# dots at the very top are subjects with at least one trial pinned at a bound
# and another pinned at the opposite one. alpha running from its 0.001 floor
# to its 500 ceiling is a ratio of 5e5, and that is not a wide estimate, it is
# no estimate at all. Ticks are one per decade for that reason -- finer ticks
# would imply a precision the top half of this axis does not have.
# One panel per quantity with its own y axis, because these are SDs in the
# parameters' own units and those units are not remotely comparable -- alpha's
# SD runs to 289 while gamma*lam/beta's median is 0.2. A shared axis would
# flatten every panel but one.
#
# Each panel is cut at its 95th percentile. The top few cells are subjects
# with a trial pinned at one bound and another at the opposite bound, and on a
# full-range axis those alone set the scale and squash everything else onto
# the baseline. The count above the cut is printed in the panel.
fig, axes = plt.subplots(1, len(PLOT_QTY), figsize=(2.9 * len(PLOT_QTY), 4.8))
for ax, q in zip(np.atleast_1d(axes), PLOT_QTY):
    v = cell[q].dropna()
    # the usual boxplot whisker rather than a fixed percentile: these six
    # distributions have very different tail weights, and a flat 95th left the
    # two ratio panels as spikes while clipping alpha and lam for no reason
    q1, q3 = v.quantile(.25), v.quantile(.75)
    hi = min(v.max(), q3 + 1.5 * (q3 - q1))
    sns.violinplot(y=v[v <= hi], color='#2B6CA3', cut=0, inner='box',
                   linewidth=1.0, saturation=1.0, width=.72, ax=ax)
    for coll in ax.collections:
        coll.set_alpha(.85)
    med = v.median()
    n_over = int((v > hi).sum())
    ax.text(.5, .97, f'median {med:.4g}'
                     + (f'\n{n_over} above axis' if n_over else ''),
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            color='0.15')
    ax.set_title(SYMBOL[q], fontsize=15)
    ax.set(xlabel='', ylabel='')
    ax.set_ylim(bottom=0, top=hi * 1.22)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel(
    "SD across a subject's own trials  (parameter's own units)")
fig.suptitle("Trial-to-trial spread of each subject's own parameters."
             "   One point = one subject within one trial type.", y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_per_trial_param_spread.png',
            dpi=200, bbox_inches='tight')
plt.show()


#%%
# ------------------------------------------------------------------
# Do the parameters drift across a session?
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('PARAMETER DRIFT ACROSS TRIALS')
print('=' * 70)

# Forearm only. kneeOA was stimulated at two sites and both are fitted, but
# including them here would enter each kneeOA person TWICE as if they were two
# independent subjects, which is exactly what the mixed model's random effects
# assume they are not. (The trial-order confound is separately handled by the
# fit unit: trial_order ranks within site, so 'later in the session' can no
# longer mean 'at the knee'.) Forearm also keeps kneeOA comparable to the
# single-site plosONE and cLBP.
ANALYSIS_SITE = 'forearm'
n_all = len(pt)
site_ok = pt['site'].fillna(ANALYSIS_SITE) == ANALYSIS_SITE
pt['at_bounds'] = pt['at_bounds'].fillna('')
clean = pt[site_ok & (pt['at_bounds'] == '')].copy()
print(f'site filter ({ANALYSIS_SITE}): {n_all:,} -> {site_ok.sum():,} trials')
print(f'{len(clean):,} of those have no parameter on a bound '
      f'({(1 - len(clean)/max(site_ok.sum(), 1))*100:.1f}% dropped)')

# Dropping bound-hit trials is only safe if they are spread evenly through a
# session. If late trials pin more often -- a plausible way for ratings to go
# flat as a session wears on -- then the exclusion changes WHICH trials
# represent early and late, which is exactly the comparison being made.
bb = pt[site_ok].assign(_bad=lambda d: d['at_bounds'] != '')
by_pos = bb[bb['trial_order'] <= 12].groupby('trial_order')['_bad'].agg(
    ['mean', 'size'])
print('\nfraction of trials with a parameter on a bound, by session position:')
print(by_pos.round(3).T.to_string())
if by_pos['size'].min() >= 10:
    rho, p_rho = st.spearmanr(bb['trial_order'], bb['_bad'].astype(int))
    print(f'  bound-hit vs trial order: rho = {rho:+.3f}, p = {p_rho:.3g}'
          + ('   <-- NOT evenly spread; the exclusion is not neutral'
             if p_rho < .05 else '   (evenly spread, exclusion is safe)'))

# All four are positive and right-skewed, so they are modelled on a log
# scale; a coefficient then reads as proportional change per trial.
DRIFT_PARAMS = [n for n in PARAM_NAMES if n != 'theta' or FIT_THETA_PER_TRIAL]
for n in DRIFT_PARAMS:
    clean[f'log_{n}'] = np.log(clean[n])
clean['trial_type_c'] = clean['trial_type'].astype(str)

print('\nmedian parameter by position in session (first 12 trials):')
for p in DRIFT_PARAMS:
    print(f'\n  {p}:')
    print(clean[clean['trial_order'] <= 12]
          .groupby(['dataset', 'trial_order'])[p].median().unstack()
          .round(3).to_string())

# trial_type is in the model because trial order and trial type are not fully
# crossed -- in cLBP, for instance, every session opens with an offset trial.
# Random slopes because subjects may drift at different rates.
#
# A caution on reading a NULL here: every trial was warm-started from the
# subject's own session-level fit, so wherever the objective is flat the
# optimizer stops near that average. That pull works against detecting drift.
# A significant slope had to overcome it and is trustworthy; a null is weaker
# than it looks.
print('\n' + '-' * 70)
drift_slopes = {}
for n in DRIFT_PARAMS:
    try:
        m = smf.mixedlm(f'log_{n} ~ trial_order + C(dataset) + C(trial_type_c)',
                        clean, groups=clean['subject_uid'],
                        re_formula='~trial_order').fit()
        slope, p = m.params['trial_order'], m.pvalues['trial_order']
    except Exception as e:
        print(f'{n:16s} mixed model failed: {e}')
        continue
    drift_slopes[n] = (slope, p)
    pct = (np.exp(slope) - 1) * 100
    # A NaN p means the random-effects covariance did not converge, which is
    # not the same as a null -- it must not fall through to the significant
    # branch and print a drift that was never estimated.
    verdict = ('DID NOT CONVERGE (p is nan) -- ignore' if not np.isfinite(p)
               else 'no reliable drift' if p >= 0.05 else
               f'{"INCREASES" if slope > 0 else "DECREASES"} '
               f'{abs(pct):.2f}% per trial')
    print(f'{n + " (log)":16s} slope = {slope:+.5f} (p = {p:.3g})  -> {verdict}')
    if np.isfinite(p) and p < 0.05:
        print(f'{"":16s} over 12 trials: {(np.exp(slope*12)-1)*100:+.1f}%')

# Per-subject slopes: a group-level effect can be carried by a few subjects,
# so check how many individuals move in the same direction.
print('\nper-subject slopes (sign test against no drift):')
for n in DRIFT_PARAMS:
    slopes = [np.polyfit(g['trial_order'], g[f'log_{n}'], 1)[0]
              for _, g in clean.groupby('subject_uid')
              if g['trial_order'].nunique() >= 5]
    if len(slopes) < 10:
        print(f'  {n:6s} only {len(slopes)} subjects with >=5 trials, skipped')
        continue
    slopes = np.array(slopes)
    n_pos = int((slopes > 0).sum())
    p_sign = st.binomtest(n_pos, len(slopes), 0.5).pvalue
    print(f'  {n:6s} {n_pos}/{len(slopes)} subjects increasing, '
          f'median slope {np.median(slopes):+.4f}, sign test p = {p_sign:.3g}')


#%%
# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, len(DRIFT_PARAMS),
                         figsize=(3.6 * len(DRIFT_PARAMS), 7.6))
axes = np.atleast_2d(axes)

for c_i, p in enumerate(DRIFT_PARAMS):
    ax = axes[0][c_i]
    for ds, g in clean[clean['trial_order'] <= 24].groupby('dataset'):
        s = g.groupby('trial_order')[p].agg(['median', 'size'])
        s = s[s['size'] >= 20]
        ax.plot(s.index, s['median'], marker='o', ms=3.5, lw=1.4,
                color=DATASET_COLORS.get(ds), label=ds)
    sl = drift_slopes.get(p)
    show_sl = sl is not None and np.isfinite(sl[1])
    ax.set(xlabel='trial position in session', ylabel=PARAM_LABEL[p],
           title=(f'{PARAM_LABEL[p]}'
                  + (f'   {(np.exp(sl[0])-1)*100:+.2f}%/trial'
                     f'{"*" if sl[1] < .05 else ""}' if show_sl else '')))
    ax.spines[['top', 'right']].set_visible(False)
    if c_i == 0:
        ax.legend(fontsize=8, frameon=False)

    # Parameters by trial type: if they differ, trial type must stay in any
    # model of trial order, because the two are not fully crossed.
    ax = axes[1][c_i]
    sns.boxplot(data=clean[clean['trial_type'].isin(MAIN_TYPES)],
                x='trial_type', y=p, order=MAIN_TYPES, ax=ax,
                color='lightsteelblue', fliersize=1.5)
    ax.set(xlabel='', ylabel=PARAM_LABEL[p],
           title=f'{PARAM_LABEL[p]} by trial type')
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=8, rotation=30)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Eq. 1 parameters fitted per trial '
             f'({len(clean):,} trials, none on a bound)', fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_per_trial_parameter_drift.png',
            dpi=150, bbox_inches='tight')
plt.show()

# Within subject, how much variability is there for each parameter?

# 
# %%
