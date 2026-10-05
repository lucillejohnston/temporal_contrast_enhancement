# -*- coding: utf-8 -*-
"""
Where does the simplified Cecchi 2012 model fail?

Reads the per-trial fit table written by Psychophysical_Modeling.py and asks
three questions:

  Q1  Is the model systematically worse on particular trial types?
      (hypothesis: worse on 'onset' / OH trials)
  Q2  Is it worse in pain groups than in controls?
  Q3  Does it degrade across repeated trials within a session?

Throughout, fit quality is the zero-lag Pearson correlation between observed
and predicted pain, computed per trial -- the measure Cecchi 2012 report
(0.92 simple / 0.88 complex stimuli). Correlations are Fisher z-transformed
before averaging or modelling, because r is bounded at 1 and piles up near the
ceiling, so means and variances on the raw scale are misleading.

Author: Lucille Johnston
Updated: 9/20/26
"""
#%%
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import plot_subject_trials

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
TRACES_FILE = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
               'temporal_contrast_enhancement/data/alter_collab_data/'
               'combined_traces_1Hz.pkl')

# Trial types present in all three datasets. plosONE also carries
# offset_AV_conditioning / offset_AV_test (an aversive conditioning paradigm),
# kneeOA has innocuous, and plosONE has calibration / inv / stepdown -- none of
# which have a counterpart elsewhere, so cross-dataset comparisons use these.
COMMON_TRIAL_TYPES = ['offset', 'onset', 't1_hold', 't2_hold']

# Fits where a parameter landed on a bound were chosen by the bound rather
# than by the data. Every result below is reported with and without them.
DROP_AT_BOUNDS = True


def fisher_z(r):
    """r -> z. Clipped because r = +/-1 maps to infinity."""
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def inverse_fisher_z(z):
    return np.tanh(z)


#%%
# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
subject_csv = sorted(glob.glob(DATA_PATH + 'model_fits_subject_*.csv'))[-1]
trial_csv = sorted(glob.glob(DATA_PATH + 'model_fits_trial_*.csv'))[-1]
print(f'subjects: {subject_csv.split("/")[-1]}')
print(f'trials  : {trial_csv.split("/")[-1]}')

subjects = pd.read_csv(subject_csv)
trials = pd.read_csv(trial_csv)
subjects['at_bounds'] = subjects['at_bounds'].fillna('')

trials['r_z'] = fisher_z(trials['r'])
trials = trials.dropna(subset=['r_z'])

flagged = set(subjects.loc[subjects['at_bounds'] != '', 'subject_uid'])
print(f'\n{len(trials):,} trials from {trials["subject_uid"].nunique()} subjects')
print(f'{len(flagged)} subjects have a parameter on a bound')

if DROP_AT_BOUNDS:
    trials_main = trials[~trials['subject_uid'].isin(flagged)].copy()
    print(f'-> analysing {len(trials_main):,} trials from '
          f'{trials_main["subject_uid"].nunique()} subjects')
else:
    trials_main = trials.copy()

print(f'\nOverall median r = {trials_main["r"].median():.3f} '
      f'(Cecchi 2012: 0.92 / 0.88)')


#%%
# ==================================================================
# Q1. Is the model worse on particular trial types?
# ==================================================================
print('\n' + '=' * 70)
print('Q1. FIT QUALITY BY TRIAL TYPE')
print('=' * 70)

by_type = (trials_main.groupby(['dataset', 'trial_type'])
           .agg(n_trials=('r', 'size'), n_subjects=('subject_uid', 'nunique'),
                median_r=('r', 'median'), mean_z=('r_z', 'mean'))
           .assign(mean_r=lambda d: inverse_fisher_z(d['mean_z']))
           .drop(columns='mean_z').round(3))
print(by_type.to_string())

common = trials_main[trials_main['trial_type'].isin(COMMON_TRIAL_TYPES)].copy()
print(f'\nAcross the {len(COMMON_TRIAL_TYPES)} shared trial types '
      f'({len(common):,} trials):')
print(common.groupby('trial_type')
      .agg(n=('r', 'size'), median_r=('r', 'median'),
           mean_r=('r_z', lambda z: inverse_fisher_z(z.mean())))
      .round(3).to_string())

# Mixed model: trial type is within-subject, so a random intercept per subject
# keeps a few well-fitted subjects from driving the effect. 'offset' is the
# reference level, so each coefficient reads as "relative to offset trials".
common['trial_type'] = pd.Categorical(common['trial_type'],
                                      categories=COMMON_TRIAL_TYPES)
m1 = smf.mixedlm('r_z ~ C(trial_type) + C(dataset)', common,
                 groups=common['subject_uid']).fit()
print('\nMixed model  r_z ~ trial_type + dataset,  random intercept per subject')
print('(coefficients are relative to offset trials, in Fisher z units)')
print(m1.summary().tables[1])

# The hypothesis was specifically that onset trials fit worse. Test it
# within-subject, on subjects who have both kinds of trial.
# .astype(str) matters: trial_type is Categorical with four levels, and a
# groupby on it keeps all four, so the unstacked frame would carry all-NaN
# t1_hold/t2_hold columns and dropna() would discard every subject.
pair_src = common[common['trial_type'].isin(['offset', 'onset'])].copy()
pair_src['trial_type'] = pair_src['trial_type'].astype(str)
paired = (pair_src.groupby(['subject_uid', 'trial_type'])['r_z']
          .mean().unstack()[['offset', 'onset']].dropna())
t_stat, p_val = stats.ttest_rel(paired['onset'], paired['offset'])
w_stat, p_w = stats.wilcoxon(paired['onset'], paired['offset'])
print(f'\nPaired test, onset vs offset ({len(paired)} subjects with both):')
print(f'   mean r: onset {inverse_fisher_z(paired["onset"].mean()):.3f}  '
      f'offset {inverse_fisher_z(paired["offset"].mean()):.3f}')
print(f'   paired t = {t_stat:.2f}, p = {p_val:.2e}')
print(f'   Wilcoxon  = {w_stat:.0f}, p = {p_w:.2e}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = (common.groupby('trial_type')['r'].median().sort_values().index.tolist())
sns.boxplot(data=common, x='trial_type', y='r', order=order, ax=axes[0],
            color='lightsteelblue', fliersize=2)
axes[0].axhline(0.88, color='crimson', ls='--', lw=1,
                label='Cecchi 2012 (complex)')
axes[0].set(xlabel='', ylabel='per-trial r', title='Fit quality by trial type')
axes[0].legend(fontsize=8)
sns.boxplot(data=common, x='trial_type', y='r', hue='dataset', order=order,
            ax=axes[1], fliersize=2)
axes[1].axhline(0.88, color='crimson', ls='--', lw=1)
axes[1].set(xlabel='', ylabel='per-trial r', title='...split by dataset')
axes[1].legend(fontsize=8, title='')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_by_trial_type.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# Q2. Is the model worse in pain groups than controls?
# ==================================================================
print('\n' + '=' * 70)
print('Q2. FIT QUALITY BY PAIN GROUP')
print('=' * 70)

# Group is a between-subject variable, so collapse to one value per subject.
per_subject = (trials_main.groupby(['subject_uid', 'dataset', 'group_label'])
               .agg(mean_z=('r_z', 'mean'), median_r=('r', 'median'),
                    n_trials=('r', 'size'))
               .reset_index())
per_subject['mean_r'] = inverse_fisher_z(per_subject['mean_z'])

print(per_subject.groupby(['dataset', 'group_label'])
      .agg(n_subjects=('subject_uid', 'size'), mean_r=('mean_r', 'mean'),
           median_r=('median_r', 'median')).round(3).to_string())

# Group and dataset are confounded: plosONE contributes 137 controls and no
# patients, cLBP contributes patients and no controls. Pooling them would
# compare studies, not groups. kneeOA is the only dataset holding all three
# groups under one protocol, so that is the interpretable comparison.
knee = per_subject[per_subject['dataset'] == 'kneeOA']
print(f'\n--- kneeOA only: the one dataset with Control, Low and High '
      f'({len(knee)} subjects) ---')
groups = [g['mean_z'].values for _, g in knee.groupby('group_label')]
labels = sorted(knee['group_label'].unique())
if len(groups) > 1:
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kw = stats.kruskal(*groups)
    print(f'   groups: {labels}')
    print(f'   one-way ANOVA on z: F = {f_stat:.2f}, p = {p_anova:.3f}')
    print(f'   Kruskal-Wallis    : H = {h_stat:.2f}, p = {p_kw:.3f}')

clbp = per_subject[per_subject['dataset'] == 'cLBP'].dropna(subset=['group_label'])
if clbp['group_label'].nunique() == 2:
    hi = clbp[clbp['group_label'] == 'High']['mean_z']
    lo = clbp[clbp['group_label'] == 'Low']['mean_z']
    u, p_u = stats.mannwhitneyu(hi, lo)
    print(f'\n--- cLBP only: High (n={len(hi)}) vs Low (n={len(lo)}) ---')
    print(f'   mean r: High {inverse_fisher_z(hi.mean()):.3f}, '
          f'Low {inverse_fisher_z(lo.mean()):.3f}')
    print(f'   Mann-Whitney U = {u:.0f}, p = {p_u:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=per_subject, x='dataset', y='mean_r', hue='group_label',
            ax=axes[0], fliersize=2)
axes[0].set(xlabel='', ylabel='mean per-trial r', title='Fit quality by group')
axes[0].legend(fontsize=8, title='')
sns.stripplot(data=knee, x='group_label', y='mean_r', ax=axes[1],
              order=['Control', 'Low', 'High'], size=5, alpha=0.7)
sns.boxplot(data=knee, x='group_label', y='mean_r', ax=axes[1],
            order=['Control', 'Low', 'High'], color='white', fliersize=0)
axes[1].set(xlabel='', ylabel='mean per-trial r',
            title='kneeOA only (same protocol, all 3 groups)')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_by_group.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# Q3. Does the fit degrade across repeated trials?
# ==================================================================
print('\n' + '=' * 70)
print('Q3. FIT QUALITY ACROSS REPEATED TRIALS')
print('=' * 70)
print("""
CAVEAT. Each subject has ONE parameter set fitted across ALL their trials, so
the optimiser has already spread its error to minimise total MSE. That biases
this test toward a U shape (good in the middle, worse at both ends) rather than
a clean trend, and it cannot distinguish "the model gets worse" from "the
compromise fit suits middle trials best". The strong version -- fit on trial 1,
predict trials 2..n with frozen parameters -- needs a separate fitting run.
Read what follows as a first look.
""")

# trial_order ranks trials within a session. cLBP encodes a second visit as
# trial_num + 100, and those visits are months apart, so order restarts at 1
# for visit 2 rather than continuing to 101.
ordered = trials_main.dropna(subset=['trial_order']).copy()
ordered['session'] = (ordered['subject_uid'] + '_v'
                      + ordered['visit'].astype(int).astype(str))

print('median r by position in session (first 12):')
pos = (ordered[ordered['trial_order'] <= 12]
       .groupby(['dataset', 'trial_order'])['r'].median().unstack().round(3))
print(pos.to_string())

# Random slope per subject: subjects may drift at different rates, and a
# random-intercept-only model would treat that as noise. trial_type is a
# covariate because trial order and type are not fully crossed.
ordered['trial_type_c'] = ordered['trial_type'].astype(str)
m3 = smf.mixedlm('r_z ~ trial_order + C(dataset) + C(trial_type_c)', ordered,
                 groups=ordered['subject_uid'],
                 re_formula='~trial_order').fit()
print('\nMixed model  r_z ~ trial_order + dataset + trial_type,')
print('             random intercept AND slope per subject')
print(m3.summary().tables[1])

slope = m3.params.get('trial_order', np.nan)
pval = m3.pvalues.get('trial_order', np.nan)
print(f'\ntrial_order slope = {slope:+.5f} z-units per trial, p = {pval:.3g}')
if np.isfinite(pval) and pval < 0.05:
    direction = 'DEGRADES' if slope < 0 else 'IMPROVES'
    r_start = inverse_fisher_z(m3.params['Intercept'])
    r_10 = inverse_fisher_z(m3.params['Intercept'] + 10 * slope)
    print(f'-> fit {direction} across trials; over 10 trials that is '
          f'r {r_start:.3f} -> {r_10:.3f}')
else:
    print('-> no reliable linear change across trials')

# Subjects with two visits are a built-in replication: if the model degrades
# within a session, a second session months later should show the same pattern
# starting fresh rather than continuing to decline.
two_visit = (ordered.groupby('subject_uid')['visit'].nunique()
             .loc[lambda s: s > 1].index)
print(f'\n--- {len(two_visit)} subjects have two visits (within-subject replication) ---')
if len(two_visit):
    tv = ordered[ordered['subject_uid'].isin(two_visit)]
    print(tv.groupby(['visit', 'trial_order'])['r'].median().unstack(0).round(3).to_string())
    for v, g in tv.groupby('visit'):
        if g['subject_uid'].nunique() > 2:
            mv = smf.mixedlm('r_z ~ trial_order', g,
                             groups=g['subject_uid']).fit()
            print(f'   visit {int(v)}: slope = {mv.params["trial_order"]:+.5f}, '
                  f'p = {mv.pvalues["trial_order"]:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ds, g in ordered[ordered['trial_order'] <= 24].groupby('dataset'):
    s = g.groupby('trial_order')['r'].agg(['median', 'size'])
    s = s[s['size'] >= 20]
    axes[0].plot(s.index, s['median'], marker='o', ms=4, label=ds)
axes[0].set(xlabel='trial position within session', ylabel='median per-trial r',
            title='Does fit quality change across a session?')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

if len(two_visit):
    tv = ordered[ordered['subject_uid'].isin(two_visit)]
    for v, g in tv.groupby('visit'):
        s = g.groupby('trial_order')['r'].median()
        axes[1].plot(s.index, s.values, marker='o', ms=4,
                     label=f'visit {int(v)}')
    axes[1].set(xlabel='trial position within session', ylabel='median per-trial r',
                title=f'Two-visit subjects (n={len(two_visit)}): does it reset?')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_across_trials.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# Sensitivity: do the answers hold if bound-flagged subjects are kept?
# ==================================================================
print('\n' + '=' * 70)
print('SENSITIVITY: including the subjects whose parameters hit a bound')
print('=' * 70)
alt = trials.copy()
alt['trial_type_c'] = alt['trial_type'].astype(str)
for label, d in [('excluding flagged', trials_main.assign(
                      trial_type_c=trials_main['trial_type'].astype(str))),
                 ('including flagged', alt)]:
    c = d[d['trial_type'].isin(COMMON_TRIAL_TYPES)]
    on = c[c['trial_type'] == 'onset']['r'].median()
    off = c[c['trial_type'] == 'offset']['r'].median()
    o = d.dropna(subset=['trial_order'])
    mm = smf.mixedlm('r_z ~ trial_order + C(dataset) + C(trial_type_c)', o,
                     groups=o['subject_uid']).fit()
    print(f'{label:20s} n={d["subject_uid"].nunique():3d} subjects | '
          f'median r {d["r"].median():.3f} | onset {on:.3f} vs offset {off:.3f} | '
          f'trial_order slope {mm.params["trial_order"]:+.5f} '
          f'(p={mm.pvalues["trial_order"]:.3g})')


#%%
# ==================================================================
# How variable is fit quality, within and between subjects?
# ==================================================================
# Everything here uses r, the zero-lag correlation Cecchi 2012 report, rather
# than r2. r2 is the coefficient of determination -- it asks whether the model
# beats a flat line at that trial's own mean -- so it divides by the variance
# of the observed ratings. On trials where the subject barely moved the slider
# that denominator is near zero and r2 explodes: the worst trial here scores
# -298,065 with an ordinary MSE of 802, purely because the subject's ratings
# had an SD of 0.58 VAS. Median MSE is flat across the whole r2 range, so the
# model's actual errors are not what r2 is tracking. r is bounded at +/-1 and
# has no such failure mode.
#
# Two distinct things vary, and they mean different things: how well the model
# describes a person on average, and how consistently it does so across their
# trials. Fitting well but erratically says some trials contain dynamics the
# model misses; fitting poorly but uniformly says the model is wrong for that
# person.
print('\n' + '=' * 70)
print('DISTRIBUTION OF FIT QUALITY (r)')
print('=' * 70)

MIN_TRIALS_FOR_SPREAD = 8   # a spread estimated from 3 trials is meaningless
MIN_PAIN_SD = 2.0           # VAS; below this a trial is too flat to score

data_df = pd.read_pickle(TRACES_FILE)
pain_sd = (data_df.groupby(['subject_uid', 'trial_num'])['pain']
           .std().rename('pain_sd').reset_index())
scored = trials_main.merge(pain_sd, on=['subject_uid', 'trial_num'], how='left')
flat = scored['pain_sd'] < MIN_PAIN_SD
print(f'{flat.sum()} of {len(scored)} trials have rating SD < {MIN_PAIN_SD} VAS '
      f'and are excluded as too flat to score')
scored = scored[~flat]

# Spread is measured on the Fisher z scale. On the raw scale a subject whose
# trials run 0.95-0.99 looks far more consistent than one running 0.55-0.75,
# but that is the ceiling at 1 compressing the top of the range, not a real
# difference in consistency.
spread = (scored.groupby(['subject_uid', 'dataset'])
          .agg(median_r=('r', 'median'), mean_z=('r_z', 'mean'),
               sd_z=('r_z', 'std'),
               iqr_r=('r', lambda x: x.quantile(.75) - x.quantile(.25)),
               min_r=('r', 'min'), max_r=('r', 'max'),
               n_trials=('r', 'size'))
          .reset_index())
spread['mean_r'] = inverse_fisher_z(spread['mean_z'])
eligible = spread[spread['n_trials'] >= MIN_TRIALS_FOR_SPREAD].copy()

print(f'\n{len(spread)} subjects, {len(eligible)} with >= {MIN_TRIALS_FOR_SPREAD} trials')
print('\nper-subject median r:')
print(spread['median_r'].describe().round(3).to_string())
print('\nper-subject spread across trials (SD on Fisher z scale):')
print(eligible['sd_z'].describe().round(3).to_string())
print('\nby dataset:')
print(spread.groupby('dataset')[['median_r', 'sd_z', 'iqr_r', 'n_trials']]
      .median().round(3).to_string())

# Exemplars. Level and spread are not independent -- a subject who fits
# uniformly badly has low spread for an uninteresting reason -- so each pick's
# level is printed alongside its spread, and the scatter below shows where
# each one sits in the joint distribution.
picks = {
    'good fit': eligible.loc[eligible['median_r'].idxmax()],
    'poor fit': eligible.loc[eligible['median_r'].idxmin()],
    'high variance across trials': eligible.loc[eligible['sd_z'].idxmax()],
    'low variance across trials': eligible.loc[eligible['sd_z'].idxmin()],
}
print('\nexemplar subjects:')
print(f'{"":30s}{"subject":<14}{"median r":>9}{"SD(z)":>8}{"IQR r":>8}'
      f'{"min r":>8}{"max r":>8}{"trials":>8}')
for label, row in picks.items():
    print(f'{label:<30s}{row["subject_uid"]:<14}{row["median_r"]:>9.3f}'
          f'{row["sd_z"]:>8.3f}{row["iqr_r"]:>8.3f}{row["min_r"]:>8.3f}'
          f'{row["max_r"]:>8.3f}{int(row["n_trials"]):>8d}')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0][0]
for ds, g in scored.groupby('dataset'):
    ax.hist(g['r'], bins=50, range=(-1, 1), alpha=0.5, label=ds)
ax.axvline(0.88, color='crimson', ls='--', lw=1, label='Cecchi 2012 (complex)')
ax.set(xlabel='per-trial r', ylabel='trials',
       title=f'All {len(scored):,} scored trials\n'
             f'median r = {scored["r"].median():.3f}')
ax.legend(fontsize=8)

ax = axes[0][1]
for ds, g in spread.groupby('dataset'):
    ax.hist(g['median_r'], bins=25, range=(0, 1), alpha=0.5, label=ds)
ax.axvline(0.88, color='crimson', ls='--', lw=1)
ax.set(xlabel="median r across that subject's trials", ylabel='subjects',
       title=f'Per-subject median r (n={len(spread)})')
ax.legend(fontsize=8)

# If this cloud slopes, level and consistency are related, and the two
# 'variance' picks are partly just picking on level.
ax = axes[1][0]
for ds, g in eligible.groupby('dataset'):
    ax.scatter(g['median_r'], g['sd_z'], s=18, alpha=0.6, label=ds)
for label, row in picks.items():
    ax.scatter(row['median_r'], row['sd_z'], s=140, facecolors='none',
               edgecolors='k', linewidths=1.5, zorder=5)
    ax.annotate(row['subject_uid'], (row['median_r'], row['sd_z']),
                textcoords='offset points', xytext=(6, 6), fontsize=7)
rho, p_rho = stats.spearmanr(eligible['median_r'], eligible['sd_z'])
ax.set(xlabel='median r  (how well)',
       ylabel='SD of Fisher z across trials  (how consistently)',
       title=f'Level vs consistency (Spearman ρ={rho:.2f}, p={p_rho:.2g})\n'
             'circled = exemplars plotted below')
ax.legend(fontsize=8)

# Every subject as a vertical range, sorted -- shows at a glance how much of
# the variability is between subjects and how much is within them.
ax = axes[1][1]
srt = eligible.sort_values('median_r').reset_index(drop=True)
ax.vlines(srt.index, srt['min_r'], srt['max_r'], color='0.8', lw=0.8)
ax.scatter(srt.index, srt['median_r'], s=10, color='crimson', zorder=3)
ax.axhline(0.88, color='crimson', ls='--', lw=1)
ax.set(xlabel='subject (sorted by median r)', ylabel='r', ylim=(-1.05, 1.05),
       title='Each subject: median (red) and full range across trials (grey)')

plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_quality_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# Trial-by-trial plots for the four exemplars
# ==================================================================
params_by_uid = subjects.set_index('subject_uid')

for label, row in picks.items():
    uid = row['subject_uid']
    p = params_by_uid.loc[uid]
    tag = label.replace(' ', '_')
    print(f'\n{label}: {uid}  '
          f'(ᾱ={p["alpha_bar"]:.2f}, γ̄={p["gamma_bar"]:.3f}, θ={p["theta"]:.1f}°C, '
          f'median r={row["median_r"]:.3f}, r range {row["min_r"]:.2f}-{row["max_r"]:.2f})')
    plot_subject_trials(
        data_df[data_df['subject_uid'] == uid],
        {'alpha_bar': p['alpha_bar'], 'gamma_bar': p['gamma_bar'],
         'theta': p['theta']},
        subject_uid=f'{uid} — {label}',
        save_path=f'{FIG_PATH}{datetime.now():%Y%m%d}_exemplar_{tag}_{uid}.png')


#%%
# ==================================================================
# Eq. 1 (full, second-order) vs Eq. 2 (simplified, first-order)
# ==================================================================
# The motivating question: Eq. 2 drops the gamma*(T' - lambda)*p term, which is
# the mechanism Cecchi describe as producing offset analgesia, and Eq. 2 does
# fit offset trials worst. Does restoring that term close the gap?
#
# Eq. 1 NESTS Eq. 2 (large beta and lambda recover it), so the full model can
# never have a higher SSE at a true optimum, and "is it better?" is not a
# meaningful question on its own -- more parameters always fit tighter. The
# question AIC asks is whether it is better by more than three extra
# parameters buy, which is the comparison Cecchi report.
FULL_SUBJECT_CSV = sorted(glob.glob(DATA_PATH + '*_model_fits_subject_full.csv'))
FULL_TRIAL_CSV = sorted(glob.glob(DATA_PATH + '*_model_fits_trial_full.csv'))

if not FULL_SUBJECT_CSV:
    print('\n(no full-model fits found; skipping the model comparison)')
else:
    full = pd.read_csv(FULL_SUBJECT_CSV[-1])
    full_trials = pd.read_csv(FULL_TRIAL_CSV[-1])
    full['at_bounds'] = full['at_bounds'].fillna('')
    print('\n' + '=' * 70)
    print('MODEL COMPARISON: Eq. 1 (second-order) vs Eq. 2 (first-order)')
    print('=' * 70)
    print(f'full-model fits: {os.path.basename(FULL_SUBJECT_CSV[-1])} ({len(full)} subjects)')

    # --- did the search actually explore? -------------------------------
    # Eq.1 was seeded at the Eq.2-equivalent point (beta = lambda = 30). If most
    # subjects never left it, a null result would mean "the optimiser did not
    # look", not "the rate term does not help", and nothing below is
    # interpretable. This check comes first for that reason.
    SEED_SCALE = 30.0
    at_seed = (((full['beta'] - SEED_SCALE).abs() < 0.01) &
               ((full['lam'] - SEED_SCALE).abs() < 0.01))
    print(f'\nOptimiser diagnostic: {at_seed.sum()}/{len(full)} ({at_seed.mean():.0%}) '
          f'never moved off the seed.')
    print('  A null result is only meaningful because this is small.')

    # --- what regime did each subject land in? --------------------------
    # These ramps move at roughly 1-1.5 degC/s, so lambda well above that makes
    # (T' - lambda) effectively constant and the rate term inert -- the full
    # model has chosen to behave like the simplified one.
    LAMBDA_INERT = 12.0
    full['regime'] = np.where(full['lam'] > LAMBDA_INERT,
                              'rate term inert (~Eq. 2)', 'rate term live')
    print(f'\nWhere the rate term ended up (lambda vs the ~1.5 degC/s ramp rate):')
    print(full.groupby('regime')
          .agg(n_subjects=('r', 'size'),
               r_full=('r', 'median'), r_simplified=('r_simplified', 'median'),
               beta=('beta', 'median'), lam=('lam', 'median')).round(3).to_string())

    # --- AIC ------------------------------------------------------------
    # AIC = n*ln(SSE/n) + 2k for Gaussian residuals. k counts fitted parameters:
    # 3 for Eq. 2 (alpha_bar, gamma_bar, theta) and 5 for Eq. 1. The variance
    # parameter is common to both and cancels in the difference.
    #
    # CAVEAT worth carrying into any write-up: AIC assumes independent
    # residuals, and consecutive samples in a pain trace are strongly
    # autocorrelated. The effective sample size is therefore far below
    # n_points, so these values overstate the evidence for BOTH models. Petre
    # 2017 use ARIMA errors for exactly this reason. Read the direction and the
    # subject counts, not the absolute magnitudes.
    K_SIMPLE, K_FULL = 3, 5
    cmp = full.merge(subjects[['subject_uid', 'sse', 'n_points']]
                     .rename(columns={'sse': 'sse_simplified',
                                      'n_points': 'n_points_simplified'}),
                     on='subject_uid', how='inner')
    n = cmp['n_points']
    cmp['aic_full'] = n * np.log(cmp['sse'] / n) + 2 * K_FULL
    cmp['aic_simplified'] = n * np.log(cmp['sse_simplified'] / n) + 2 * K_SIMPLE
    cmp['delta_aic'] = cmp['aic_full'] - cmp['aic_simplified']   # negative favours Eq. 1

    print(f'\nAIC across {len(cmp)} subjects (negative ΔAIC favours the full model):')
    print(f'  median ΔAIC            : {cmp["delta_aic"].median():+.1f}')
    for lo, hi, label in [(-np.inf, -10, 'Eq. 1 strongly favoured (ΔAIC < -10)'),
                          (-10, -2, 'Eq. 1 favoured (-10 to -2)'),
                          (-2, 2, 'indistinguishable (-2 to +2)'),
                          (2, 10, 'Eq. 2 favoured (+2 to +10)'),
                          (10, np.inf, 'Eq. 2 strongly favoured (> +10)')]:
        m = (cmp['delta_aic'] > lo) & (cmp['delta_aic'] <= hi)
        print(f'    {label:<40s} {m.sum():3d} ({m.mean():.0%})')
    print('\n  by regime:')
    print(cmp.groupby('regime')['delta_aic']
          .agg(n='size', median='median',
               pct_favouring_full=lambda x: (x < -2).mean() * 100).round(1).to_string())

    # --- the question the run existed for -------------------------------
    # Eq. 2 fits offset trials worst. If the rate term is what offset trials
    # need, the full model should improve them specifically -- not uniformly.
    print(f'\nPer-trial fit by trial type, both models (median r):')
    a = (trials.groupby('trial_type')['r'].median().rename('Eq. 2'))
    b = (full_trials.groupby('trial_type')['r'].median().rename('Eq. 1'))
    bytype = pd.concat([a, b], axis=1)
    bytype['gain'] = bytype['Eq. 1'] - bytype['Eq. 2']
    print(bytype.round(3).to_string())

    paired = (trials[['subject_uid', 'trial_num', 'trial_type', 'r']]
              .rename(columns={'r': 'r_simple'})
              .merge(full_trials[['subject_uid', 'trial_num', 'r']]
                     .rename(columns={'r': 'r_full'}),
                     on=['subject_uid', 'trial_num'], how='inner'))
    paired['dz'] = fisher_z(paired['r_full']) - fisher_z(paired['r_simple'])
    paired = paired.dropna(subset=['dz'])
    print(f'\nPaired per-trial gain (Fisher z), {len(paired):,} matched trials:')
    for tt, g in paired.groupby('trial_type'):
        if len(g) < 30:
            continue
        t_stat, p_val = stats.ttest_1samp(g['dz'], 0)
        print(f'  {tt:<12s} n={len(g):5d}  mean Δz={g["dz"].mean():+.4f}  '
              f't={t_stat:+6.2f}  p={p_val:.2g}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    lim = [0, 1]
    for reg, g in cmp.groupby('regime'):
        ax.scatter(g['r_simplified'], g['r'], s=20, alpha=0.65, label=reg)
    ax.plot(lim, lim, 'k--', lw=1, alpha=.5)
    ax.set(xlim=lim, ylim=lim, xlabel='r — Eq. 2 (first-order)',
           ylabel='r — Eq. 1 (second-order)',
           title='Per-subject fit, both models\n(points above the line favour Eq. 1)')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(cmp['delta_aic'].clip(-60, 60), bins=45, color='#7f9bb5',
            edgecolor='#33475b')
    ax.axvline(0, color='k', lw=1)
    ax.axvline(-2, color='crimson', ls='--', lw=1, label='ΔAIC = −2')
    ax.set(xlabel='ΔAIC (clipped at ±60) — negative favours Eq. 1', ylabel='subjects',
           title=f'AIC penalises the 2 extra parameters\nmedian ΔAIC = {cmp["delta_aic"].median():+.1f}')
    ax.legend(fontsize=8)

    ax = axes[2]
    order = bytype.index.tolist()
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, bytype['Eq. 2'], w, label='Eq. 2', color='#3d6b8f')
    ax.bar(x + w/2, bytype['Eq. 1'], w, label='Eq. 1', color='#b3202c')
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set(ylabel='median per-trial r', ylim=(0, 1),
           title='Does the rate term rescue offset trials?')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

# %%
