# -*- coding: utf-8 -*-
"""
What the fitted parameters say about trial types, groups and time.

Author: Lucille Johnston
Updated: 10/28/26
"""
#%%
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')

SITE = 'forearm'
MAIN_TYPES = ['offset', 'onset', 't1_hold', 't2_hold']
MIN_TRIALS_PER_CELL = 3    # fewer than this and a within-cell SD means little
pd.set_option('display.width', 200)

# Same palette as analyze_model_fits.py: light -> dark with increasing pain.
# The database spells it both 'Control' and 'control' depending on the study,
# so labels are capitalised before anything groups or plots them.
GROUP_ORDER = ['Control', 'Low', 'High']
GROUP_COLORS = {'Control': '#DBBBAF', 'Low': '#91472A', 'High': '#491400'}

# Eq. 1 (the full second-order model) and its five parameters:
#   p'' = alpha*F(T,theta) - beta*p' + gamma*(T' - lam)*p
#
# Only gamma is plotted on a log axis, and only because its interquartile
# range (0.13 - 0.42) sits three orders of magnitude below the handful of
# subjects pinned at its upper bound of 50. On a linear axis those few set the
# scale and flatten every box into the baseline. The other four have ranges
# their boxes can live on directly, so they show actual values. The axis
# choice is cosmetic either way -- the Kruskal tests below are rank-based.
PLOT_PARAMS = [('alpha', r'$\alpha$  (drive)', False),
               ('beta', r'$\beta$  (damping)', False),
               ('gamma', r'$\gamma$  (rate-sensitivity gain)', True),
               ('lam', r'$\lambda$  (rate offset, °C s$^{-1}$)', False),
               ('theta', r'$\theta$  (threshold, °C)', False)]
EQ1_PARAMS = [p for p, _, _ in PLOT_PARAMS]


def fisher_z(r):
    """r is bounded at 1, so average and model it on the z scale."""
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


#%%
# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
# Eq. 2 is gone from this file. The one thing that still depends on it is the
# trial-to-trial variance section: per-trial fitting was only ever run for
# Eq. 2 (alpha_bar / gamma_bar), and there is no Eq. 1 equivalent to swap in.
per_trial_csv = sorted(glob.glob(RESULTS_PATH + '*_per_trial_fits_fixedtheta.csv'))[-1]
full_csv = sorted(glob.glob(RESULTS_PATH + '*_model_fits_subject_full.csv'))[-1]
period_csv = sorted(glob.glob(RESULTS_PATH + '*_model_fits_per_period.csv'))[-1]

trials = pd.read_csv(per_trial_csv)
full = pd.read_csv(full_csv)
periods = pd.read_csv(period_csv)
print(f'Eq. 1 subject  : {os.path.basename(full_csv)} ({len(full)})')
print(f'per-period     : {os.path.basename(period_csv)} ({len(periods):,} rows)')
print(f'per-trial fits : {os.path.basename(per_trial_csv)} ({len(trials):,} trials)'
      f'  [Eq. 2 only -- section 2 ]')

for d in (trials, full):
    if 'site' not in d.columns:
        d['site'] = SITE
    d['site'] = d['site'].fillna(SITE)
trials = trials[trials['site'] == SITE]
full = full[full['site'] == SITE]


#%%
# ==================================================================
# 1. PARAMETERS ACROSS GROUPS  (Eq. 1, one parameter set per subject)
# ==================================================================
print('\n' + '=' * 78)
print('1. DO THE FITTED PARAMETERS DIFFER ACROSS PAIN GROUPS?')
print('=' * 78)

eq1 = full.copy()
eq1['group_label'] = eq1['group_label'].str.capitalize()
n_bounds = (eq1['at_bounds'].fillna('') != '').sum()
print(f'{len(eq1)} subject fits ({SITE}); {n_bounds} have at least one '
      f'parameter sitting on a bound and are kept, not excluded')
print(eq1['group_label'].value_counts(dropna=False).to_dict())


def compare_params(df, params, group_col, order, label):
    """Kruskal-Wallis per parameter, Holm-corrected across the parameters
    because five looks at one question is five chances to find something."""
    sub = df[df[group_col].isin(order)]
    present = [g for g in order if (sub[group_col] == g).sum() >= 10]
    sub = sub[sub[group_col].isin(present)]
    if len(present) < 2:
        print(f'\n  {label}: too few groups')
        return
    print(f'\n  {label}  (n per group: '
          f'{ {g: int((sub[group_col] == g).sum()) for g in present} })')
    tab = sub.groupby(group_col)[params].median().reindex(present).round(3)
    raw = [stats.kruskal(*[sub.loc[sub[group_col] == g, p].dropna()
                           for g in present])[1] for p in params]
    tab.loc['p'] = np.round(raw, 4)
    tab.loc['p_holm'] = np.round(multipletests(raw, method='holm')[1], 4)
    print(tab.to_string())


compare_params(eq1, EQ1_PARAMS, 'group_label', GROUP_ORDER,
               'by pain group, all datasets pooled (confounded)')
compare_params(eq1[eq1['dataset'] == 'kneeOA'], EQ1_PARAMS, 'group_label',
               GROUP_ORDER, 'by pain group within kneeOA only (unconfounded)')

# Across datasets rather than across groups. Worth having as the contrast to
# the two tests above: the parameters do NOT separate pain groups, but they do
# separate studies, which says the model is picking up protocol rather than
# pathology. theta differing is the least surprising part of that -- the
# studies reach different peak temperatures.
#
# The lambda row of this table is NOT interpretable: lambda sits on its upper
# bound of 50 for 60% of plosONE subjects, 25% of kneeOA and 6% of cLBP, so a
# "difference in lambda" here is a difference in how often the optimizer ran
# out of room. The same caveat applies wherever lambda is compared.
compare_params(eq1, EQ1_PARAMS, 'dataset', ['plosONE', 'kneeOA', 'cLBP'],
               'by dataset (NOT a group comparison -- see note on lambda)')
print('   lambda pinned at its upper bound, by dataset: '
      + str((eq1.assign(_b=eq1['lam'] >= 49.99)
             .groupby('dataset')['_b'].mean().mul(100).round(1).to_dict())))

fig, axes = plt.subplots(1, len(PLOT_PARAMS), figsize=(16.5, 3.9))
sub = eq1[eq1['group_label'].isin(GROUP_ORDER)]
for ax, (p, label, log) in zip(axes, PLOT_PARAMS):
    present = [g for g in GROUP_ORDER if (sub['group_label'] == g).any()]
    bp = ax.boxplot([sub.loc[sub['group_label'] == g, p].dropna()
                     for g in present],
                    labels=present, patch_artist=True, widths=.62,
                    medianprops=dict(color='black', lw=1.6),
                    flierprops=dict(marker='.', ms=3, mfc='0.4', mec='none'))
    for patch, g in zip(bp['boxes'], present):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(.9)
        patch.set_edgecolor('0.25')
    if log:
        ax.set_yscale('log')
    ax.set_title(label, fontsize=11)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('fitted value', fontsize=9)
fig.suptitle('Eq. 1 parameters by pain group (one fit per subject)',
             fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig(FIG_PATH + f'{datetime.now():%Y%m%d}_eq1_params_by_group.png',
            dpi=200, bbox_inches='tight')
plt.show()

# ---- the same parameters against the habituator / sensitizer classes ----
# The classification files key subjects as (dataset offset + within-study
# number): plosONE + 0, kneeOA + 1000, cLBP + 2000. cLBP used to be dropped
# here because the classifications predated the DPOP/MBPR collision fix and
# could not be matched; they were regenerated on 28 Oct 2026 and all 71 cLBP
# subjects now join, so all three datasets are included.
OFFSETS = {'plosONE': 0, 'kneeOA': 1000, 'cLBP': 2000}


def join_classifications(fits):
    f = fits[fits['dataset'].isin(OFFSETS)].copy()
    f['class_subject'] = (f['subject_uid'].str.split('_').str[-1].astype(int)
                          + f['dataset'].map(OFFSETS))
    for name, col in [('holdslope', 'classification'),
                      ('trajectory', 'trajectory_classification')]:
        c = pd.read_csv(DATA_PATH + f'{name}_classifications.csv')
        keep = ['subject', 'dataset', col] + (
            ['observed_mean_slope'] if 'observed_mean_slope' in c.columns else [])
        c = c[keep].rename(columns={'subject': 'class_subject', col: f'{name}_class'})
        f = f.merge(c, on=['class_subject', 'dataset'], how='left')
    return f


eq1 = join_classifications(eq1)
print(f'\n{eq1["holdslope_class"].notna().sum()} of {len(eq1)} subjects '
      f'carry a hold-slope classification')
for scheme in ['holdslope_class', 'trajectory_class']:
    print(f'  {scheme}: {eq1[scheme].value_counts(dropna=False).to_dict()}')

for scheme, lab in [('holdslope_class', 'by habituator/sensitizer (hold slope)'),
                    ('trajectory_class', 'by habituator/sensitizer (trajectory)')]:
    order = [g for g, n in eq1[scheme].value_counts().items() if n >= 10]
    compare_params(eq1, EQ1_PARAMS, scheme, order, lab)

# The classes are cut from a continuous measure, so testing against that
# measure directly is more powerful than testing against the cut version.
print('\n--- parameters vs the continuous hold slope '
      '(negative slope = habituator) ---')
s = eq1.dropna(subset=['observed_mean_slope'])
raw, names = [], []
for p in EQ1_PARAMS:
    v = s.dropna(subset=[p])
    if len(v) < 20:
        continue
    rho, pv = stats.spearmanr(v['observed_mean_slope'], v[p])
    raw.append(pv)
    names.append((p, rho, len(v)))
for (p, rho, n), pv, pa in zip(names, raw, multipletests(raw, method='holm')[1]):
    print(f'    {p:8s} n={n:4d}  rho = {rho:+.3f}  p = {pv:.4f}  '
          f'(Holm {pa:.4f})')


#%%
# ==================================================================
# 2. TRIAL-TO-TRIAL PARAMETER VARIANCE, BY TRIAL TYPE
# ==================================================================
# The prediction: fit each trial on its own and the stepped trials should give
# wildly different parameters from trial to trial, while the holds should be
# stable. If that holds, the parameters are soaking up the step rather than
# describing it.
#
# Compared as a coefficient of variation (SD / |mean|) rather than a raw SD,
# because alpha_bar and gamma_bar sit on different scales and a subject with a
# large alpha_bar would otherwise look more variable by construction.
print('\n' + '=' * 78)
print('1. HOW MUCH DO PARAMETERS VARY TRIAL TO TRIAL, WITHIN A TRIAL TYPE?')
print('=' * 78)

t = trials[trials['trial_type'].isin(MAIN_TYPES)].copy()
cell = (t.groupby(['subject_uid', 'dataset', 'trial_type'])
        .agg(n=('alpha_bar', 'size'),
             alpha_sd=('alpha_bar', 'std'), alpha_mean=('alpha_bar', 'mean'),
             gamma_sd=('gamma_bar', 'std'), gamma_mean=('gamma_bar', 'mean'),
             r_mean=('r', 'mean'), points=('n_points', 'mean'))
        .reset_index())
cell = cell[cell['n'] >= MIN_TRIALS_PER_CELL]
cell['alpha_cv'] = cell['alpha_sd'] / cell['alpha_mean'].abs()
cell['gamma_cv'] = cell['gamma_sd'] / cell['gamma_mean'].abs()
print(f'{len(cell):,} subject x trial-type cells with >= {MIN_TRIALS_PER_CELL} trials '
      f'({cell["subject_uid"].nunique()} subjects)\n')
print(cell.groupby('trial_type')[['n', 'alpha_cv', 'gamma_cv', 'r_mean']]
      .median().round(3).to_string())

# Within subject, so a subject who is variable everywhere cannot drive it.
print('\nWithin subject, stepped vs hold (Wilcoxon on paired cells):')
for param in ['alpha_cv', 'gamma_cv']:
    wide = cell.pivot_table(index='subject_uid', columns='trial_type', values=param)
    for stepped, hold in [('offset', 't1_hold'), ('onset', 't2_hold')]:
        pair = wide[[stepped, hold]].dropna()
        if len(pair) < 20:
            continue
        stat, p = stats.wilcoxon(pair[stepped], pair[hold])
        direction = 'MORE' if pair[stepped].median() > pair[hold].median() else 'LESS'
        print(f'  {param:9s} {stepped:7s} {pair[stepped].median():6.3f} vs '
              f'{hold:8s} {pair[hold].median():6.3f}   n={len(pair):3d}  '
              f'p={p:8.2g}   stepped is {direction} variable')

# A trial the model fits badly can give an extreme parameter, so variance and
# fit quality are not independent. Reported so the reader can judge.
print('\n(mean per-trial r by type, for context on the above)')
print(t.groupby('trial_type')['r'].agg(n='size', median='median').round(3).to_string())


#%%
# ==================================================================
# 3. PERIOD C ACROSS A SESSION
# ==================================================================
# Whole-trial fit quality improves across trials. Period C is the window the
# offset effect lives in, so whether it behaves the same way is a different
# question -- and the one that bears on a learning account.
print('\n' + '=' * 78)
print('3. WITHIN PERIOD C, DOES FIT QUALITY CHANGE ACROSS A SESSION?')
print('=' * 78)

# No flatness filter: a trial where the rating barely moves is a real
# response, and excluding those took out hold windows three times as often as
# offset windows, which shrinks the trial-type contrast being measured.
# SENSITIVITY: add  & (periods['pain_sd'] >= 2.0)  to the mask below.
#
# Scored on RMSE, not r. r is blind to level -- a prediction with the right
# shape at the wrong height scores near 1 -- and period C is where the model
# gets the shape right and the level wrong, which is exactly what r cannot
# see. RMSE is the square root of the mse the fit was optimised on, in VAS.
c = periods[(periods['period'] == 'C') & periods['mse_eq1'].notna()
            & periods['trial_type'].isin(MAIN_TYPES)].copy()
c['rmse'] = np.sqrt(c['mse_eq1'])
c['signed'] = c['mean_obs'] - c['mean_eq1']   # + = model too low
print(f'{len(c):,} scorable period-C rows, {c["subject_uid"].nunique()} subjects\n')

print('period C by trial type (RMSE and signed error, VAS):')
print(c.groupby('trial_type')[['rmse', 'signed']]
      .agg(['size', 'median']).round(2).to_string())

# Trial order is only meaningful within a subject and visit, so the slope is
# taken per subject and then tested across subjects -- a subject with a high
# baseline fit cannot create a trend this way.
print('\nchange in period-C error across trials, per subject (VAS per trial):')
rows = []
for tt, g in c.groupby('trial_type'):
    slopes = []
    for uid, s in g.groupby('subject_uid'):
        # a session holds only ~3 trials of each type, so a higher floor here
        # empties the table entirely
        if s['trial_order'].nunique() < 3:
            continue
        slopes.append(np.polyfit(s['trial_order'], s['rmse'], 1)[0])
    if len(slopes) < 20:
        continue
    stat, pv = stats.wilcoxon(slopes)
    rows.append({'trial_type': tt, 'n_subjects': len(slopes),
                 'median_slope_VAS_per_trial': np.median(slopes), 'p': pv})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print('\n(NEGATIVE = period C error SHRINKS later in the session, i.e. the '
      'model fits better.\n The sign flipped from the old r version, where '
      'positive meant better.)')


#%%
# ------------------------------------------------------------------
# Figure for section 2 (trial-to-trial variance). Eq. 2 parameters, because
# that is the only model that was ever fitted per trial.
# ------------------------------------------------------------------
# The third panel here used to plot Eq. 2's gamma_bar against the hold slope.
# That question now lives in section 1 as a Spearman over all five Eq. 1
# parameters, so it is not repeated.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
data = [cell[cell['trial_type'] == tt]['gamma_cv'].dropna() for tt in MAIN_TYPES]
ax.boxplot(data, labels=MAIN_TYPES, showfliers=False)
ax.set(ylabel='CV of gamma_bar across trials',
       title='Trial-to-trial parameter variability\n(within subject, within trial type)')

ax = axes[1]
data = [cell[cell['trial_type'] == tt]['alpha_cv'].dropna() for tt in MAIN_TYPES]
ax.boxplot(data, labels=MAIN_TYPES, showfliers=False)
ax.set(ylabel='CV of alpha_bar across trials', title='Same, for alpha_bar')

plt.tight_layout()
plt.savefig(FIG_PATH + f'{datetime.now():%Y%m%d}_parameter_comparisons.png',
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# 4. THE SAME HISTOGRAMS CECCHI 2012 PUBLISHED
# ==================================================================
# Their parameter figure is five panels in this order: alpha, 1/beta (sec),
# T_0 (degC), gamma, lambda. T_0 is their name for theta, and beta is shown as
# the time constant 1/beta rather than the rate, so both are converted here
# instead of being plotted raw.
#
# The plosONE row is the one that matters: that study IS their dataset, so any
# difference there is a difference in method, not in subjects.
#
# The reference numbers below were read off the published histogram (their
# Figure S4B), so they are approximate -- good to roughly one significant
# figure, no better. They are drawn as a line and a span, never as bars, so
# nothing here can be mistaken for their actual data.
#
# ------------------------------------------------------------------
# WHAT THIS COMPARISON IS AND IS NOT WORTH (worked out 1 Nov 2026
# against the paper and its supplement, which are in references/)
# ------------------------------------------------------------------
# Short version: the raw parameters look wildly different, almost all of that
# is parameterisation rather than disagreement, and the one number that does
# disagree is the damping ratio.
#
# 1. THEIR ALPHA IS NOT USABLE. Figure S1 prints both models' parameters for
#    each of their subjects, so their own algebra can be checked against
#    their own numbers. Using alpha_bar = alpha/beta and gamma_bar =
#    gamma*lam/beta (their Methods):
#        gamma_bar reproduces their first-order fit, median ratio 0.83
#        alpha_bar does NOT, median ratio 0.063 -- i.e. 16x too small, and
#        all 10 subjects fall below 0.13, so it is systematic, not scatter.
#    gamma_bar is scale-free while alpha_bar carries the pain units, so a
#    clean deficit in alpha_bar alone is the signature of the two models
#    having been fitted to differently-normalised ratings. Whatever the
#    cause, their published alpha contradicts their own alpha_bar by more
#    than an order of magnitude. The alpha panel below is kept for
#    completeness; it should not be read as a comparison.
#
# 2. THEIR RATINGS ARE ON A ~0-1 SCALE, which is why CECCHI_PAIN_SCALE is
#    1.0 and not 100.0. Their Methods describe a finger-span device anchored
#    0-100, but their first-order alpha_bar/gamma_bar is ~1.0 pain units per
#    degC and their stimulus sits 2-4 degC above threshold, so the model
#    predicts a plateau of 2-4: saturated on a 0-1 axis, absurd on 0-100.
#    HOW they normalised (divide by 100, or per-subject min-max) is not
#    stated anywhere in the paper. It does not matter here, because of (1).
#
# 3. WHAT ACTUALLY AGREES. Per-subject medians from their Figure S1 (simple
#    stimulus, n = 10) against the fits in this file (n = 251):
#        gamma_bar = gamma*lam/beta   theirs 0.27-0.35   ours 0.33
#        T_0 / theta                  theirs 44.7        ours 43.9
#    Those are the two things the slow envelope of a 1 Hz trace determines,
#    and they match.
#
# 4. WHAT DISAGREES, AND IT IS ONE RIDGE PLUS ONE REAL DIFFERENCE.
#        beta     theirs 0.90 [0.25-5.0]    ours 29.5
#        lam      theirs 2.91 [0.29-9.5]    ours 35.9
#        gamma    theirs 0.075              ours 0.235
#        gamma*lam theirs 0.224             ours 7.43
#    beta and gamma*lam are both ~33x apart while their RATIO (gamma_bar) is
#    not, which is the definition of sliding along a flat ridge: in the
#    overdamped limit the trajectory depends on alpha/beta and gamma*lam/beta
#    and not on beta itself, so the optimiser can move beta freely at almost
#    no cost in MSE. That is the ridge, not a result.
#
#    The real difference is where on that ridge each fit sits:
#        zeta^2 = beta^2 / (4*gamma*lam)   theirs 0.77   ours 18.9
#        fraction underdamped (zeta^2 < 1) theirs 50%    ours 10%
#    During a hold (T' = 0) Eq. 1 is p'' + beta*p' + gamma*lam*p = alpha*F,
#    a damped oscillator with zeta = beta/(2*sqrt(gamma*lam)). Below 1 it
#    overshoots; above 1 it cannot. Half their fits can overshoot a hold
#    trajectory. A tenth of ours can. That is the one place the two
#    parameter sets describe genuinely different dynamics, and it is also
#    the mechanism offset analgesia needs, which is consistent with the
#    period-C offset failure in section 3 above.
#
# 5. THE LAMBDA CEILING OF 50 WAS NEVER THE CONSTRAINT. Their whole fitted
#    range, 0.29 to 9.54, sits inside our bounds. Raising the ceiling cannot
#    help; our optimiser is choosing a lambda 12x larger than theirs, and
#    large lambda is exactly what switches the rate term off (their
#    assumption (b) for collapsing Eq. 1 to Eq. 2 is |T'| << lambda). The
#    ratio that decides whether the rate term is alive, |T'|/lambda, is
#    about 2.8 at their 8 degC/s ramps and about 0.03 at our 1.5 degC/s
#    step -- roughly 90x apart.
#
# 6. NEXT TEST, when this is picked up again: profile over lambda, not over
#    its ceiling. Fix lambda at 0.1, 0.3, 1, 3, 10, 30, 50; refit alpha,
#    beta, gamma, theta at each; plot RMSE against lambda. Flat from 3 to 50
#    means lambda was never identified and the fix is to fit gamma*lam and
#    gamma directly so the rate term cannot be switched off for free. A
#    steep climb below 10 means the data really do reject their regime, and
#    the next question is whether that survives refitting at the native
#    sampling rate (the 1 Hz traces are a downsample, and a 1 degC step at
#    1.5 degC/s lasts 0.67 s -- less than one sample).
#
# SCALE CAVEAT, and it only touches alpha: alpha carries the units of the pain
# axis ([pain] / degC / s^2), so it moves by exactly the ratio of the two
# rating scales. Everything else in the model -- beta, gamma, lambda, theta --
# is scale-free. See (2) above for why CECCHI_PAIN_SCALE is 1.0.
print('\n' + '=' * 78)
print('4. PARAMETER DISTRIBUTIONS vs CECCHI 2012')
print('=' * 78)

CECCHI_PAIN_SCALE = 1.0     # their ratings assumed 0-1; yours are 0-100
VAS_SCALE = 100.0
ALPHA_RESCALE = VAS_SCALE / CECCHI_PAIN_SCALE

# (key, x label, value getter, Cecchi median, Cecchi range, display x range)
#
# The display range is set by hand rather than from the data, because for
# alpha and lam the interesting feature IS the pile-up against the fitting
# bound and it has to stay visible. Where no bound is involved the range is
# left as None and cut at the 98th percentile instead, so that one subject
# with beta pinned at its 0.05 floor (1/beta = 20 s) cannot squeeze every
# other subject into a single bar. Anything past the right edge is counted in
# the panel rather than dropped silently.
CECCHI_PANELS = [
    ('alpha', r'$\alpha$', lambda d: d['alpha'],
     0.015 * ALPHA_RESCALE, (0.0 * ALPHA_RESCALE, 0.125 * ALPHA_RESCALE),
     (0, 500)),
    ('inv_beta', r'1/$\beta$  (sec)', lambda d: 1.0 / d['beta'],
     2.1, (0.5, 5.0), None),
    ('theta', r'T$_0$  ($^\circ$C)', lambda d: d['theta'],
     45.5, (36.0, 47.0), (30, 48)),
    # gamma is cut at 1.0, just past the 0.7 their widest subject reached, so
    # the bulk is legible. The tail past it is real and sizeable (~18% of
    # subjects, out to the bound at 50) and is counted in the panel -- that
    # heavy tail is itself one of the differences from their fits.
    ('gamma', r'$\gamma$', lambda d: d['gamma'],
     0.07, (0.0, 0.7), (0, 1.0)),
    ('lam', r'$\lambda$', lambda d: d['lam'],
     3.0, (0.0, 22.0), (0, 50)),
]

eq1_f = full.copy()                      # already filtered to SITE above
ROWS = [('plosONE  (Cecchi 2012 dataset)', eq1_f[eq1_f['dataset'] == 'plosONE']),
        ('all datasets', eq1_f)]

fig, axes = plt.subplots(len(ROWS), len(CECCHI_PANELS),
                         figsize=(15.5, 3.3 * len(ROWS)))
axes = np.atleast_2d(axes)
for row, (row_label, sub) in enumerate(ROWS):
    for col, (key, xlabel, get, ref_med, ref_rng, xlim) in enumerate(CECCHI_PANELS):
        ax = axes[row, col]
        v = get(sub).dropna()
        lo, hi = xlim if xlim is not None else (v.min(), v.quantile(.95))
        n_over = int((v > hi).sum())
        # Their bars are plain grey with white gaps and a full box frame, so
        # the two figures can sit side by side on a slide without the styling
        # itself being a difference.
        counts, _, _ = ax.hist(v, bins=np.linspace(lo, hi, 13),
                               color='0.5', edgecolor='white', linewidth=.6)
        # Their median as a line, their full range as a faint span.
        ax.axvspan(*ref_rng, color='#B5651D', alpha=.10, zorder=0)
        ax.axvline(ref_med, color='#B5651D', lw=1.4, ls='--', zorder=3)
        ax.set_xlim(lo, hi)
        # Put the caption in whichever half of the panel holds fewer subjects,
        # so it never lands on top of the bars or the Cecchi line.
        right_heavy = counts[len(counts) // 2:].sum() > counts[:len(counts) // 2].sum()
        xt, ha = (.03, 'left') if right_heavy else (.97, 'right')
        note = (f'\n{n_over} above axis ({100 * n_over / len(v):.0f}%)'
                if n_over else '')
        off = ' (off scale)' if not (lo <= ref_med <= hi) else ''
        ax.text(xt, .95, f'Cecchi $\\approx${ref_med:.3g}{off}\n'
                         f'yours = {v.median():.3g}{note}',
                transform=ax.transAxes, ha=ha, va='top', fontsize=7.5,
                color='#8A4A14')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.tick_params(labelsize=8)
        if row == 0:
            ax.set_title('2nd-order', fontsize=10)
    axes[row, 0].set_ylabel(f'{row_label}\n(n = {len(sub)})', fontsize=9)
fig.suptitle('Eq. 1 parameters, in the order Cecchi 2012 plots them '
             '(dashed = their median, shaded = their range)', fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(FIG_PATH + f'{datetime.now():%Y%m%d}_cecchi_parameter_histograms.png',
            dpi=200, bbox_inches='tight')
plt.show()


#%%
# ------------------------------------------------------------------
# 4b. The combinations that are actually identifiable
# ------------------------------------------------------------------
# Comparing alpha to alpha is the wrong comparison. In the overdamped limit
# (which is where 90% of these fits live -- see the damping panel) Eq. 1
# collapses to
#       p' = (alpha/beta) F  -  (gamma*lam/beta) p
# so the data constrain alpha/beta and gamma*lam/beta, and beta itself is free
# to slide along a flat ridge. Two fits can disagree by four orders of
# magnitude in alpha and still describe the same trajectory.
#
# The last panel is the one that bears on offset analgesia: Eq. 1 can only
# undershoot a hold trajectory if it is underdamped, beta^2 < 4*gamma*lam.
# Values above 1 are overdamped and structurally cannot produce OA through
# this route.
DERIVED = [
    ('alpha/beta', r'$\bar\alpha = \alpha/\beta$' + '\n(rise rate)',
     lambda d: d['alpha'] / d['beta'], 0.015 / 0.476 * ALPHA_RESCALE),
    ('gammalam/beta', r'$\bar\gamma = \gamma\lambda/\beta$' + '\n(decay rate, s$^{-1}$)',
     lambda d: d['gamma'] * d['lam'] / d['beta'], 0.21 / 0.476),
    ('tau', r'$\beta/\gamma\lambda$' + '\n(slow time constant, s)',
     lambda d: d['beta'] / (d['gamma'] * d['lam']), 0.476 / 0.21),
    ('gain', r'$\alpha/\gamma\lambda$' + '\n' + r'(steady gain, per $^\circ$C)',
     lambda d: d['alpha'] / (d['gamma'] * d['lam']),
     0.015 / 0.21 * ALPHA_RESCALE),
    ('damping', r'$\beta^2/4\gamma\lambda$' + '\n(>1 = overdamped)',
     lambda d: d['beta'] ** 2 / (4 * d['gamma'] * d['lam']),
     0.476 ** 2 / (4 * 0.21)),
]

fig, axes = plt.subplots(1, len(DERIVED), figsize=(16.5, 3.6))
for ax, (key, label, get, ref) in zip(axes, DERIVED):
    v = get(eq1_f).replace([np.inf, -np.inf], np.nan).dropna()
    v = v[v > 0]
    # Log only where the data demand it. These are all ratios, and a ratio
    # whose 5th-95th span exceeds 50x cannot show its own shape on a linear
    # axis -- the top few percent set the scale and flatten everything else.
    # Below that threshold the linear axis is left alone.
    use_log = v.quantile(.95) / v.quantile(.05) > 50
    if use_log:
        lo, hi = v.quantile(.01), v.quantile(.99)
        bins = np.logspace(np.log10(lo), np.log10(hi), 20)
    else:
        # Cut the top 5% off the axis rather than the data. These all have one
        # long right tail, and on a full-range linear axis it puts every
        # subject in the leftmost bar.
        lo, hi = max(v.min(), 0), v.quantile(.95)
        bins = np.linspace(lo, hi, 20)
    counts, _, _ = ax.hist(v, bins=bins, color='0.5', edgecolor='white',
                           linewidth=.6)
    if use_log:
        ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.axvline(ref, color='#B5651D', lw=1.6, ls='--')
    if key == 'damping':
        ax.axvline(1.0, color='#2B6CA3', lw=1.4)
        # low in the panel: the caption above it moves side to side depending
        # on where the bars sit, and these two would otherwise collide
        ax.text(.03, .50, f'{(get(eq1_f) < 1).mean() * 100:.0f}% underdamped\n'
                          f'(left of the blue line)',
                transform=ax.transAxes, fontsize=7.5, color='#2B6CA3')
    n_over = int((v > hi).sum())
    note = f'\n{n_over} above axis ({100 * n_over / len(v):.0f}%)' if n_over else ''
    right_heavy = counts[len(counts) // 2:].sum() > counts[:len(counts) // 2].sum()
    xt, ha = (.03, 'left') if right_heavy else (.97, 'right')
    ax.text(xt, .95, f'Cecchi $\\approx${ref:.3g}\n'
                     f'yours = {v.median():.3g}{note}',
            transform=ax.transAxes, ha=ha, va='top', fontsize=7.5,
            color='#8A4A14')
    ax.set_xlabel(label, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('subjects', fontsize=9)
fig.suptitle('What is actually comparable: the identifiable combinations '
             '(all datasets, dashed = Cecchi 2012)', fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig(FIG_PATH + f'{datetime.now():%Y%m%d}_cecchi_identifiable_combos.png',
            dpi=200, bbox_inches='tight')
plt.show()

print('\nmedians, yours vs Cecchi 2012 (alpha-bearing rows assume their pain '
      f'axis spans 0-{CECCHI_PAIN_SCALE:g} and yours 0-{VAS_SCALE:g}):')
rows = []
for name, sub in [('plosONE', eq1_f[eq1_f['dataset'] == 'plosONE']),
                  ('all', eq1_f)]:
    for key, _, get, ref in DERIVED:
        med = get(sub).replace([np.inf, -np.inf], np.nan).median()
        rows.append({'subset': name, 'quantity': key, 'yours': med,
                     'cecchi': ref, 'fold_off': med / ref})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print('\nFor contrast, the raw parameters are off by:')
for key, _, get, ref_med, _, _ in CECCHI_PANELS:
    med = get(eq1_f).median()
    print(f'  {key:9s} yours {med:10.4g}  Cecchi {ref_med:8.4g}  '
          f'{med / ref_med:>10.1f}x')

# %%
