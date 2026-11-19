# -*- coding: utf-8 -*-
"""
Is there trial-to-trial autocorrelation in the PER-TRIAL fitted parameters?

plot_autocorrelations() in psychophysics_modeling_functions.py runs ACF/PACF
on one long series. Here each run (subject x visit) is only ~9-15 trials, too
short for a per-subject ACF, so the same lag-k correlation is pooled: every
within-run pair of trials k apart contributes one (x_t, x_t+k) point.

Three things would fake autocorrelation and are removed first, on log(param):
  - subject level: every subject has its own typical value -> demean per run
  - trial type: types come in a fixed order, and type shifts the parameters
    -> subtract a global trial-type effect (estimated within run, so it does
       not depend on trial order)
  - (second version only) slow drift across the session -> also remove each
    run's own linear trend over trial_order. What survives that is carryover
    from one trial to the next rather than a gradual change.

Demeaning a short series biases its ACF negative (about -1/(n-1)), so the
null is not zero: it is built by shuffling trial order within each run and
redoing the demean/detrend, 1000 times. Points outside the 95% band are
autocorrelation the shuffle cannot produce.

Author: Lucille Johnston
Updated: 11/05/26
"""
#%%
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
PARAMS = [('alpha', 'α'), ('beta', 'β'), ('gamma', 'γ'), ('lam', 'λ'),
          ('alpha_over_beta', 'α/β  (steady-state gain)'),
          ('gamma_lam_over_beta', 'γλ/β  (rate-term weight)')]
GROUP_ORDER = ['Control', 'Low', 'High']
GROUP_COLORS = {'Control': '#DBBBAF', 'Low': '#91472A', 'High': '#491400'}
VERSION_COLORS = {'demeaned': '#491400', 'detrended': '#B07A5E'}
MAX_LAG = 5
N_PERM = 1000
STAMP = f'{datetime.now():%Y%m%d}'
rng = np.random.default_rng(0)

#%%
pt_csv = sorted(glob.glob(RESULTS_PATH + '*_per_trial_fits_full_fixedtheta.csv'))[-1]
pt = pd.read_csv(pt_csv)
pt['alpha_over_beta'] = pt['alpha'] / pt['beta']
pt['gamma_lam_over_beta'] = pt['gamma'] * pt['lam'] / pt['beta']
pt['run'] = pt['subject_uid'] + '_v' + pt['visit'].astype(str)
pt = pt.sort_values(['run', 'trial_order']).reset_index(drop=True)
n_run = pt.groupby('run').size()
print(f'{os.path.basename(pt_csv)}: {len(pt):,} trials, {len(n_run)} runs '
      f'(median {n_run.median():.0f} trials/run)')
gaps = (pt.groupby('run')['trial_order'].diff().dropna() != 1).sum()
print(f'{gaps} gaps in trial_order (missing fits); pairs are matched on '
      f'trial_order, so a gap never counts as lag 1')

#%%
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def type_effect(y, run, ttype):
    """Global trial-type effect on y, estimated from within-run deviations."""
    dev = y - y.groupby(run).transform('mean')
    return dev.groupby(ttype).transform('mean')


def run_residual(y, run, order, detrend):
    """Remove each run's mean (and, optionally, its linear trend in order)."""
    r = y - y.groupby(run).transform('mean')
    if detrend:
        o = order - order.groupby(run).transform('mean')
        slope = (r * o).groupby(run).transform('sum') / \
                (o ** 2).groupby(run).transform('sum')
        r = r - slope.fillna(0) * o
    return r


def pooled_acf(r, run, order, max_lag):
    """Pearson r over all within-run pairs exactly k trials apart."""
    d = pd.DataFrame({'run': run.values, 'order': order.values, 'r': r.values})
    out = []
    for k in range(1, max_lag + 1):
        nxt = d.assign(order=d['order'] - k)
        m = d.merge(nxt, on=['run', 'order'], suffixes=('', '_k'))
        out.append((np.corrcoef(m['r'], m['r_k'])[0, 1], len(m)))
    return out


def lag_pairs_index(run, order, max_lag):
    """Row-index pairs for each lag, so the permutations can skip the merge."""
    d = pd.DataFrame({'run': run.values, 'order': order.values,
                      'i': np.arange(len(run))})
    idx = []
    for k in range(1, max_lag + 1):
        m = d.merge(d.assign(order=d['order'] - k), on=['run', 'order'],
                    suffixes=('', '_k'))
        idx.append((m['i'].to_numpy(), m['i_k'].to_numpy()))
    return idx

#%%
# ------------------------------------------------------------------
# Observed ACF and permutation null
# ------------------------------------------------------------------
rows = []
for p, _ in PARAMS:
    d = pt[pt[p] > 0].reset_index(drop=True)
    y = np.log10(d[p])
    y = y - type_effect(y, d['run'], d['trial_type'])
    pairs = lag_pairs_index(d['run'], d['trial_order'], MAX_LAG)
    run_codes = d['run'].astype('category').cat.codes.to_numpy()
    for version in ('demeaned', 'detrended'):
        r = run_residual(y, d['run'], d['trial_order'], version == 'detrended')
        obs = pooled_acf(r, d['run'], d['trial_order'], MAX_LAG)
        # null: shuffle the type-corrected values within each run, then redo
        # the run demean/detrend, so the null carries the same short-series bias
        null = np.empty((N_PERM, MAX_LAG))
        yv = y.to_numpy()
        for b in range(N_PERM):
            perm = np.lexsort((rng.random(len(yv)), run_codes))
            ys = pd.Series(yv[perm])
            rs = run_residual(ys, d['run'], d['trial_order'],
                              version == 'detrended').to_numpy()
            for k, (i, j) in enumerate(pairs):
                null[b, k] = np.corrcoef(rs[i], rs[j])[0, 1]
        for k in range(MAX_LAG):
            acf, n = obs[k]
            lo, hi = np.percentile(null[:, k], [2.5, 97.5])
            p_two = (np.sum(np.abs(null[:, k] - null[:, k].mean())
                            >= abs(acf - null[:, k].mean())) + 1) / (N_PERM + 1)
            rows.append({'param': p, 'version': version, 'lag': k + 1,
                         'acf': acf, 'null_mean': null[:, k].mean(),
                         'null_lo': lo, 'null_hi': hi, 'p_perm': p_two,
                         'n_pairs': n})
    print(f'  {p} done')
acf = pd.DataFrame(rows)
# d is sorted by run then trial_order, which the lexsort above relies on:
# shuffling within run = reordering rows inside each run's block

#%%
# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
# Traditional ACF / PACF (statsmodels), one column per parameter. Each session
# (subject x visit, 9-15 trials) is its own series in trial order; ACF and PACF
# are computed per session and averaged across sessions (stem), with a 95% CI
# of that mean. statsmodels' PACF needs lags < n/2, so a 9-trial session only
# reaches lag 3. Series are log10(param) with the trial-type effect removed,
# since trial types come in a fixed order. Dashed line = the same average with
# trial order shuffled within session: short series sit a little below zero
# even with no carryover, so that line is the reference, not zero.
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

ACF_LAGS = 4
N_SHUF = 200
LAGS = np.arange(ACF_LAGS + 1)
SHORT = ['α', 'β', 'γ', 'λ', 'α/β (gain)', 'γλ/β (rate weight)']


def acf_pacf(v):
    a = np.full(ACF_LAGS + 1, np.nan)
    pa = a.copy()
    k = min(ACF_LAGS, len(v) // 2 - 1)
    a[:k + 1] = sm_acf(v, nlags=k, fft=False)
    pa[:k + 1] = sm_pacf(v, nlags=k, method='ywm')
    return a, pa


fig, axes = plt.subplots(2, len(PARAMS), figsize=(15, 5.6), sharex=True,
                         sharey='row')
for j, (p, _) in enumerate(PARAMS):
    d = pt[pt[p] > 0].reset_index(drop=True)
    y = np.log10(d[p])
    d['y'] = y - type_effect(y, d['run'], d['trial_type'])
    obs, null = {'ACF': [], 'PACF': []}, {'ACF': [], 'PACF': []}
    for _, g in d.groupby('run'):
        v = g.sort_values('trial_order')['y'].to_numpy()
        if len(v) < 6 or np.std(v) == 0:
            continue
        a, pa = acf_pacf(v)
        obs['ACF'].append(a)
        obs['PACF'].append(pa)
        sa, spa = zip(*(acf_pacf(rng.permutation(v)) for _ in range(N_SHUF)))
        null['ACF'].append(np.nanmean(sa, axis=0))
        null['PACF'].append(np.nanmean(spa, axis=0))
    for i, kind in enumerate(('ACF', 'PACF')):
        ax = axes[i, j]
        o = np.array(obs[kind])
        m = np.nanmean(o, axis=0)
        ci = 1.96 * np.nanstd(o, axis=0) / np.sqrt(np.sum(~np.isnan(o), axis=0))
        ci[0] = 0
        nm = np.nanmean(np.array(null[kind]), axis=0)
        ax.vlines(LAGS, 0, m, color='#491400', lw=1.5)
        ax.errorbar(LAGS, m, yerr=ci, fmt='o', color='#491400', ms=4,
                    capsize=2.5, lw=1, label='observed (95% CI)')
        ax.plot(LAGS[1:], nm[1:], ls='--', color='0.5', lw=1,
                label='shuffled trial order')
        ax.axhline(0, color='0.7', lw=.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.set_title(SHORT[j], fontsize=10)
        else:
            ax.set_xlabel('lag (trials)')
        if j == 0:
            ax.set_ylabel(kind)
    print(f'  {p}: {len(obs["ACF"])} sessions')
axes[0, 0].set_xticks(LAGS)
axes[0, 0].set_ylim(-.35, 1.05)
axes[1, 0].set_ylim(-.35, 1.05)
axes[0, -1].legend(fontsize=8, frameon=False, loc='upper right')
fig.suptitle('Autocorrelation of per-trial parameters across trials '
             '(within session, averaged over sessions)', fontsize=10.5)
fig.tight_layout()
out = FIG_PATH + f'{STAMP}_per_trial_param_autocorrelation.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'-> {out}')

csv_out = RESULTS_PATH + f'{STAMP}_per_trial_param_autocorrelation.csv'
acf.to_csv(csv_out, index=False)
print(f'-> {csv_out}')
print('\n' + acf[acf['lag'] <= 2].round(3).to_string(index=False))
plt.show()
