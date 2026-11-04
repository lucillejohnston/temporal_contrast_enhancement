# -*- coding: utf-8 -*-
"""
Looking into the results of the model fit
Specifically the fit one set of parameters per subject model fit
Check per_trial_fits to look at fits per trial 

Author: Lucille Johnston
Updated: 10/08/26
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
from statsmodels.stats.multitest import multipletests
from datetime import datetime
from itertools import combinations

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (plot_subject_trials,
                                              prepare_trials_for_optimization,
                                              simulate_trials_analytic)

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
TRACES_FILE = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
               'temporal_contrast_enhancement/data/alter_collab_data/'
               'combined_traces_1Hz.pkl')

# Just in case there's any confusion but I think I already fixed the naming inconsistencies
COMMON_TRIAL_TYPES = ['offset', 'onset', 't1_hold', 't2_hold']

# Each stepped trial sits next to the hold it should be compared against, so
# the adjacent pairs in the figure are the contrasts that matter. Note this is
# the PLOT order only -- COMMON_TRIAL_TYPES still sets the model's reference
# level (offset), so changing this does not change any statistic.
PLOT_ORDER = ['t1_hold', 'offset', 't2_hold', 'onset']

# offset probes offset analgesia, onset probes onset hyperalgesia -- the two
# conditions of interest, so they get the only colours. The holds are their
# controls and are greyed out so the eye goes to OA and OH; light vs medium
# distinguishes the two temperature levels without adding a third hue.
TRIAL_COLORS = {'offset': '#026A07',   # OA
                'onset': '#DA1C1C',    # OH
                't1_hold': '#CFCFCF',
                't2_hold': '#8A8A8A'}
TRIAL_LABELS = {'offset': 'Offset\n(OA)', 'onset': 'Onset\n(OH)',
                't1_hold': 'Hold\n(T1)', 't2_hold': 'Hold\n(T2)'}

# Pain group: light -> dark with increasing pain. The database spells it both
# 'Control' and 'control' depending on the study, so labels are capitalised
# before anything groups or plots them -- otherwise those are two categories.
GROUP_ORDER = ['Control', 'Low', 'High']
GROUP_COLORS = {'Control': '#DBBBAF', 'Low': '#91472A', 'High': '#491400'}

# Dataset is a nuisance factor here, not the point, so these are deliberately
# muted and carry no ordering.
DATASET_ORDER = ['plosONE', 'kneeOA', 'cLBP']
DATASET_COLORS = {'plosONE': '#8FA8B8', 'kneeOA': '#A7B29A', 'cLBP': '#AFA3B5'}

# One value per subject per trial type, not one per trial. A subject
# contributes ~3 trials of each type, so a trial-level distribution lets the
# heavily-sampled subjects shape the violin. It also matches the mixed model
# below, which gives each subject one intercept. Set False for trial-level.
PLOT_PER_SUBJECT = True

# Fits where a parameter landed on a bound were chosen by the bound rather
# than by the data. Every result below is reported with and without them.
DROP_AT_BOUNDS = False

# kneeOA has forearm and knee and is fitted per site.
# and the model comparison use forearm only, so kneeOA stays comparable to the
# single-site plosONE and cLBP; the knee fits get their own paired section at
# the end of this file.
SITE = 'forearm'

def fisher_z(r):
    """r -> z. Clipped because r = +/-1 maps to infinity."""
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def inverse_fisher_z(z):
    return np.tanh(z)

#%%
# ------------------------------------------------------------------
# Load the data 
# ------------------------------------------------------------------
subject_csv = sorted(glob.glob(DATA_PATH + 'model_fits_subject_*.csv'))[-1]
trial_csv = sorted(glob.glob(DATA_PATH + 'model_fits_trial_*.csv'))[-1]
print(f'subjects: {subject_csv.split("/")[-1]}')
print(f'trials  : {trial_csv.split("/")[-1]}')

subjects = pd.read_csv(subject_csv)
trials = pd.read_csv(trial_csv)
subjects['at_bounds'] = subjects['at_bounds'].fillna('')

# separte the sites
for frame in (subjects, trials):
    if 'site' not in frame.columns:
        frame['site'] = SITE
    frame['site'] = frame['site'].fillna(SITE)
    if 'subject' not in frame.columns:
        frame['subject'] = frame['subject_uid']
    frame['subject'] = frame['subject'].fillna(frame['subject_uid'])

# Keep the full table for the paired site comparison at the end, then restrict
# everything else to one site.
subjects_all_sites, trials_all_sites = subjects.copy(), trials.copy()
n_before = len(subjects)
subjects = subjects[subjects['site'] == SITE]
trials = trials[trials['site'] == SITE].copy()
print(f'site filter ({SITE}): {n_before} -> {len(subjects)} subject fits')

trials['r_z'] = fisher_z(trials['r'])

# RMSE, in VAS, is the headline measure everywhere below. r is kept for one
# thing only: the replication line against Cecchi 2012's 0.88. The reason is
# that r is blind to level -- a prediction with the right shape at the wrong
# height scores near 1 -- and every question here is about level. RMSE is the
# square root of the mse the fits were actually optimised on, so it reports
# the quantity the optimizer minimised, in the units of the rating scale.
trials['rmse'] = np.sqrt(trials['mse'])

# Trials with a flat rating have no r, but they do have an RMSE, so they are
# kept now and only excluded from the r column. Previously they were dropped
# outright, which quietly removed the flattest hold trials from everything.
unscorable = trials['r_z'].isna()
if unscorable.any():
    unscorable_pain_trials = unscorable & (trials['trial_type'] != 'innocuous')
    print(f'\n{unscorable.sum()} trials have no r (flat rating or prediction); '
          f'{unscorable_pain_trials.sum()} of them supra-threshold. '
          f'They are KEPT -- RMSE is still defined:')
    print(trials[unscorable].groupby(['dataset', 'trial_type'])
          .size().to_string())

# RMSE scales with how wide a range a subject uses on the VAS (Spearman rho
# = 0.78 against each subject's own 5-95 percentile range). That is harmless
# for a within-subject comparison such as trial type, where each person is
# their own control, but it would dominate any comparison BETWEEN subjects --
# a pain-group difference in raw RMSE would largely be a difference in rating
# style. nrmse divides it out and is what the group and dataset plots use.
_traces_for_range = pd.read_pickle(TRACES_FILE)
vas_range = (_traces_for_range.groupby('subject_uid')['pain']
             .agg(lambda x: x.quantile(.95) - x.quantile(.05))
             .rename('vas_range'))
trials = trials.merge(vas_range, on='subject_uid', how='left')
trials['nrmse'] = trials['rmse'] / trials['vas_range'].replace(0, np.nan)
print(f"\nsubject VAS range (p95-p05): median "
      f"{vas_range.median():.1f}, IQR {vas_range.quantile(.25):.1f}"
      f"-{vas_range.quantile(.75):.1f}")

flagged = set(subjects.loc[subjects['at_bounds'] != '', 'subject_uid'])
print(f'\n{len(trials):,} trials from {trials["subject_uid"].nunique()} subjects')
print(f'{len(flagged)} subjects have a parameter on a bound')

if DROP_AT_BOUNDS:
    trials_main = trials[~trials['subject_uid'].isin(flagged)].copy()
    print(f'-> analysing {len(trials_main):,} trials from '
          f'{trials_main["subject_uid"].nunique()} subjects')
else:
    trials_main = trials.copy()

print(f'\nOverall median r = {trials["r"].median():.3f}  (all {trials["subject_uid"].nunique()} subjects)')
print(f'                   {trials_main["r"].median():.3f}  '
      f'(excluding {len(flagged)} bound-flagged)')
print(f'                   Cecchi 2012: 0.92 simple / 0.88 complex stimuli')

# # %%
# # ==================================================================
# # Explore the flagged subjects with parameters at the bounds
# # ==================================================================
# print('\n' + '=' * 70)
# print('SUBJECTS WITH PARAMETER ON A BOUND')
# print('=' * 70)

# # bounds match optimize_cecchi_simplified in psychophysics_modeling_functions
# BOUNDS = {'alpha_bar':(0.001, 50.0),
#           'gamma_bar':(0.001,3.0),
#           'theta':(30.0, 52.0)}

# at_bound = subjects[subjects['at_bounds']!= ''].copy()
# print(f'{len(at_bound)}/{len(subjects)} subjects ({len(at_bound)/len(subjects):.0%})\n')

# # Which end of which bound, since low and high mean different things
# hits = []
# for _, row in at_bound.iterrows():
#     for param in str(row['at_bounds']).split(','):
#         param = param.strip()
#         if param in BOUNDS:
#             lo, hi = BOUNDS[param]
#             end = 'lower' if abs(row[param] - lo) < abs(row[param] - hi) else 'upper'
#             hits.append({'subject_uid': row['subject_uid'],
#                          'dataset': row['dataset'], 'param': param,
#                          'end': end, 'value': row[param], 'r': row['r'],
#                          'n_trials': row['n_trials']})
# hits = pd.DataFrame(hits)
# print('which bound, and which end:')
# print(hits.groupby(['param', 'end']).agg(
#     n=('r', 'size'), median_r=('r', 'median')).round(3).to_string())
# print('\nby dataset:')
# print(hits.groupby(['dataset', 'param', 'end']).size().to_string())

# # Are these bad fits, or just extreme subjects? If their fit quality matches
# # everyone else's, the bound is a constraint on the parameter rather than a
# # symptom of failure, and excluding them is throwing away usable data.
# clean = subjects[subjects['at_bounds'] == '']
# print(f'\nfit quality, flagged vs not:')
# print(f'  on a bound : median r {at_bound["r"].median():.3f}  '
#       f'(n={len(at_bound)}, median {at_bound["n_trials"].median():.0f} trials)')
# print(f'  clean      : median r {clean["r"].median():.3f}  '
#       f'(n={len(clean)}, median {clean["n_trials"].median():.0f} trials)')
# u, p = stats.mannwhitneyu(at_bound['r'].dropna(), clean['r'].dropna())
# print(f'  Mann-Whitney p = {p:.3g}')

# print('\nFlagged subjects:')
# print(at_bound[['subject_uid', 'dataset', 'group_label', 'at_bounds',
#             'alpha_bar', 'gamma_bar', 'theta', 'r', 'n_trials']]
#       .sort_values('r').round(3).to_string(index=False))

# # Refit the bound-flagged subjects with much wider limits 
# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath('.')))
# from psychophysics_modeling_functions import optimize_cecchi_simplified

# WIDE = [(1e-4, 5000.0), (1e-5, 50.0), (25.0, 55.0)]
# data_df = pd.read_pickle(TRACES_FILE)

# rows = []
# for uid in sorted(flagged):
#     sd = data_df[data_df['subject_uid'] == uid]
#     old = subjects.set_index('subject_uid').loc[uid]
#     new, _ = optimize_cecchi_simplified(sd, threshold=None, optimizer='de',
#                                         bounds=WIDE, verbose=False)
#     if new is None:
#         continue
#     rows.append({'subject_uid': uid, 'r_old': old['r'], 'r_new': new['r'],
#                  'alpha_old': old['alpha_bar'], 'alpha_new': new['alpha_bar'],
#                  'gamma_old': old['gamma_bar'], 'gamma_new': new['gamma_bar'],
#                  'theta_old': old['theta'], 'theta_new': new['theta'],
#                  'still_at_bound': ','.join(new['at_bounds'])})
# w = pd.DataFrame(rows)
# w['gain'] = w['r_new'] - w['r_old']
# print(w.round(3).to_string(index=False))
# print(f'\nimproved by >0.02: {(w.gain > 0.02).sum()}/{len(w)}')
# print(f'still on a bound : {(w.still_at_bound != "").sum()}/{len(w)}')
#%%
# ==================================================================
# Is the model worse on particular trial types?
# ==================================================================
print('\n' + '=' * 70)
print('FIT QUALITY BY TRIAL TYPE')
print('=' * 70)

by_type = (trials_main.groupby(['dataset', 'trial_type'])
           .agg(n_trials=('rmse', 'size'), n_subjects=('subject_uid', 'nunique'),
                median_rmse=('rmse', 'median'), median_r=('r', 'median'))
           .round(3))
print(by_type.to_string())

common = trials_main[trials_main['trial_type'].isin(COMMON_TRIAL_TYPES)].copy()
print(f'\nAcross the {len(COMMON_TRIAL_TYPES)} shared trial types '
      f'({len(common):,} trials):')
print(common.groupby('trial_type')
      .agg(n=('rmse', 'size'), median_rmse=('rmse', 'median'),
           median_r=('r', 'median'))
      .round(3).to_string())

# THE one place r is still the headline: Cecchi 2012 report r, so this is the
# number that says whether the replication worked. Everything after it is a
# question about level, which r cannot answer, and uses RMSE instead.
print(f'\nREPLICATION: median per-trial r = {trials["r"].median():.3f}   '
      f'(Cecchi 2012: 0.92 simple / 0.88 complex)')

# Mixed model: trial type is within-subject, so a random intercept per subject
# keeps a few well-fitted subjects from driving the effect. 'offset' is the
# reference level, so each coefficient reads as "relative to offset trials".
# Fitted on RMSE rather than Fisher z: RMSE is unbounded so it needs no
# variance-stabilising transform, and its units (VAS) are readable, so a
# coefficient of +1.5 means "1.5 VAS worse than offset trials".
common['trial_type'] = pd.Categorical(common['trial_type'],
                                      categories=COMMON_TRIAL_TYPES)
m1 = smf.mixedlm('rmse ~ C(trial_type) + C(dataset)', common,
                 groups=common['subject_uid']).fit()
print('\nMixed model  rmse ~ trial_type + dataset,  random intercept per subject')
print('(coefficients are relative to offset trials, in VAS)')
print(m1.summary().tables[1])

# The hypothesis was specifically that onset trials fit worse. Test it
# within-subject, on subjects who have both kinds of trial.
pair_src = common[common['trial_type'].isin(['offset', 'onset'])].copy()
pair_src['trial_type'] = pair_src['trial_type'].astype(str)
paired = (pair_src.groupby(['subject_uid', 'trial_type'])['rmse']
          .mean().unstack()[['offset', 'onset']].dropna())
t_stat, p_val = stats.ttest_rel(paired['onset'], paired['offset'])
w_stat, p_w = stats.wilcoxon(paired['onset'], paired['offset'])
print(f'\nPaired test, onset vs offset ({len(paired)} subjects with both):')
print(f'   mean RMSE: onset {paired["onset"].mean():.2f} VAS  '
      f'offset {paired["offset"].mean():.2f} VAS  '
      f'(offset is {paired["offset"].mean() - paired["onset"].mean():+.2f} worse)')
print(f'   paired t = {t_stat:.2f}, p = {p_val:.2e}')
print(f'   Wilcoxon  = {w_stat:.0f}, p = {p_w:.2e}')

# ------------------------------------------------------------------
# FIGURE: fit quality by trial type
# ------------------------------------------------------------------
# Raw RMSE, not normalised: trial type is a WITHIN-subject contrast, so each
# person is their own control and their rating range cancels out.
if PLOT_PER_SUBJECT:
    plot_src = (common.groupby(['subject_uid', 'trial_type'], observed=True)['rmse']
                .mean().reset_index())
    unit = 'subjects'
else:
    plot_src = common.copy()
    unit = 'trials'

counts = plot_src['trial_type'].value_counts()

fig, ax = plt.subplots(figsize=(8.6, 5.8))
# cut=0 stops the KDE tails running past the observed range.
sns.violinplot(data=plot_src, x='trial_type', y='rmse', order=PLOT_ORDER,
               hue='trial_type', palette=TRIAL_COLORS, legend=False,
               cut=0, inner='box', linewidth=1.0, saturation=1.0, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.88)

ax.set_xticks(range(len(PLOT_ORDER)))
ax.set_xticklabels([TRIAL_LABELS[tt] for tt in PLOT_ORDER])
ax.set(xlabel='', ylabel=f'model error (RMSE, VAS) per {unit[:-1]}',
       title='Fit per trial type')
ax.spines[['top', 'right']].set_visible(False)
ax.invert_yaxis()   # lower RMSE is a better fit, so better reads as higher

# Median sits above its own violin.
low = plot_src['rmse'].min()
span = plot_src['rmse'].max() - low
for i, tt in enumerate(PLOT_ORDER):
    vals = plot_src.loc[plot_src['trial_type'] == tt, 'rmse']
    ax.text(i, vals.min() - .012 * span, f'{vals.median():.1f}', va='bottom',
            ha='center', fontsize=9.5, color='0.15')

# No significance brackets. With 251 subjects every pairwise contrast came out
# at p < .001 after correction, so the stars separated nothing. The mixed
# model printed above reports the effect sizes in VAS, which is where the
# contrast between these trial types actually lives.
ax.set_ylim(plot_src['rmse'].max() + .04 * span, low - .07 * span)

plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_by_trial_type.png',
            dpi=200, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# Example fits: one OA trial and one OH trial, for the same slide
# ==================================================================
# Picked at the MEDIAN r for their trial type, so the example is typical of
# the violin next to it rather than a flattering case. Run the contact sheet
# first, pick the rank you like out of it, set EXAMPLE_PICK, re-run.
EXAMPLE_TYPES = ['offset', 'onset']
N_CANDIDATES = 6                           # panels per row in the contact sheet
EXAMPLE_PICK = {'offset': 2, 'onset': 4}   # 0 = closest to the median r

traces = pd.read_pickle(TRACES_FILE)
traces['site'] = traces['site'].fillna(SITE)
traces = traces[traces['site'] == SITE]
fitted_params = subjects.set_index('subject_uid')[['alpha_bar', 'gamma_bar',
                                                   'theta']]


def median_candidates(trial_type, n):
    """Trials of this type, ordered by how close their RMSE is to the median."""
    pool = trials_main[trials_main['trial_type'] == trial_type]
    med = pool['rmse'].median()
    return (pool.assign(_dist=(pool['rmse'] - med).abs())
            .sort_values('_dist').head(n).reset_index(drop=True))


def plot_example_fit(row, ax, show_temp=True):
    """Observed pain in black, model in the trial type's colour."""
    uid, tnum = row['subject_uid'], int(row['trial_num'])
    subject_rows = traces[traces['subject_uid'] == uid]
    trial_list = prepare_trials_for_optimization(subject_rows)
    idx = next((i for i, tr in enumerate(trial_list)
                if tr['trial_num'] == tnum), None)
    if idx is None:
        ax.set_axis_off()
        return
    params = fitted_params.loc[uid].to_dict()
    pred = simulate_trials_analytic(params, trial_list)[idx]
    tr = trial_list[idx]

    if show_temp:
        axT = ax.twinx()
        axT.plot(tr['time'], tr['temperature'], color='0.75', lw=1.0, zorder=0)
        axT.set_ylabel('°C', fontsize=7, color='0.6')
        axT.tick_params(labelsize=6, colors='0.6')
        axT.spines[['top']].set_visible(False)

    ax.plot(tr['time'], tr['pain'], color='black', lw=1.6, label='rating',
            zorder=3)
    ax.plot(tr['time'], pred, color=TRIAL_COLORS[row['trial_type']], lw=1.8,
            label='model', zorder=4)

    # The stored RMSE came from the subject-level fit; recomputing it here is
    # a check that this panel really is the trial the table is describing.
    check = float(np.sqrt(np.mean((tr['pain'] - pred) ** 2)))
    warn = '' if abs(check - row['rmse']) < 0.5 else '  !!'
    bias = float(np.mean(tr['pain'] - pred))
    ax.set_title(f"{uid}  trial {tnum}\nRMSE = {row['rmse']:.1f} VAS   "
                 f"bias = {bias:+.1f}{warn}", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)


# ---- contact sheet: flip through these ----
fig, axes = plt.subplots(len(EXAMPLE_TYPES), N_CANDIDATES,
                         figsize=(3.0 * N_CANDIDATES, 2.6 * len(EXAMPLE_TYPES)),
                         squeeze=False)
for r_i, tt in enumerate(EXAMPLE_TYPES):
    cands = median_candidates(tt, N_CANDIDATES)
    print(f'\n{TRIAL_LABELS[tt].replace(chr(10), " ")}  median RMSE = '
          f'{trials_main.loc[trials_main["trial_type"] == tt, "rmse"].median():.2f} VAS')
    print(cands[['subject_uid', 'trial_num', 'dataset', 'rmse', 'r']]
          .round(3).to_string(index=True))
    for c_i in range(N_CANDIDATES):
        ax = axes[r_i][c_i]
        if c_i < len(cands):
            plot_example_fit(cands.loc[c_i], ax, show_temp=False)
            ax.text(.02, .97, f'rank {c_i}', transform=ax.transAxes,
                    fontsize=7, va='top', color='0.45')
        else:
            ax.set_axis_off()
    axes[r_i][0].set_ylabel(f'{TRIAL_LABELS[tt]}\npain (VAS)', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_example_fit_candidates.png',
            dpi=150, bbox_inches='tight')
plt.show()


# ---- the picked pair, slide-sized ----
# Stacked and sharing x: the two trials run on the same clock, so the stimulus
# phases line up vertically. y is NOT shared -- these are different people, and
# forcing a common VAS range flattens whichever of them rated lower.
fig, axes = plt.subplots(len(EXAMPLE_TYPES), 1, figsize=(5, 6.6),
                         sharex=True)
axes = np.atleast_1d(axes)
for ax, tt in zip(axes, EXAMPLE_TYPES):
    cands = median_candidates(tt, max(N_CANDIDATES, EXAMPLE_PICK[tt] + 1))
    plot_example_fit(cands.loc[EXAMPLE_PICK[tt]], ax)
    ax.set_ylabel(f'{TRIAL_LABELS[tt].replace(chr(10), " ")}\npain (VAS)',
                  fontsize=9)
axes[-1].set_xlabel('time (s)', fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_example_fit_OA_OH.png',
            dpi=200, bbox_inches='tight')
plt.show()



#%%
# ==================================================================
# Is the model worse in pain groups than controls?
# ==================================================================
print('\n' + '=' * 70)
print('FIT QUALITY BY PAIN GROUP')
print('=' * 70)

# Group is a between-subject variable, so collapse to one value per subject.
# This is also why there is no mixed model here as there was for trial type:
# with one value per subject there are no repeated measures left to nest, and
# a random intercept per subject would have nothing to pool.
# Everything in this section uses nrmse, RMSE divided by that subject's own
# VAS range, NOT raw RMSE. Raw RMSE tracks how wide a range a person uses on
# the scale at rho = 0.78, so a raw-RMSE group difference would mostly be a
# difference in rating style between the studies. Dividing it out leaves
# "how far off the model is, as a fraction of the range this person uses".
scored_groups = trials_main.copy()
scored_groups['group_label'] = scored_groups['group_label'].str.capitalize()
per_subject = (scored_groups.groupby(['subject_uid', 'dataset', 'group_label'])
               .agg(nrmse=('nrmse', 'mean'), rmse=('rmse', 'mean'),
                    n_trials=('rmse', 'size'))
               .reset_index())

# groupby drops rows with a null key, so subjects with no group in the
# database are already gone by this point. Said out loud rather than implied.
n_nogroup = scored_groups.loc[scored_groups['group_label'].isna(), 'subject_uid'].nunique()
if n_nogroup:
    print(f'({n_nogroup} subjects have no group in the database and are '
          f'excluded from this section)')

print(per_subject.groupby(['dataset', 'group_label'])
      .agg(n_subjects=('subject_uid', 'size'), mean_nrmse=('nrmse', 'mean'),
           mean_rmse_vas=('rmse', 'mean')).round(3).to_string())

# Group and dataset are confounded: plosONE contributes controls and no
# patients, cLBP contributes patients and no controls. Pooling them would
# compare studies, not groups. kneeOA is the only dataset holding all three
# groups under one protocol, so that is the interpretable comparison.
knee = per_subject[per_subject['dataset'] == 'kneeOA']
print(f'\n--- kneeOA only: the one dataset with Control, Low and High '
      f'({len(knee)} subjects) ---')
groups = [g['nrmse'].values for _, g in knee.groupby('group_label')]
labels = sorted(knee['group_label'].unique())
if len(groups) > 1:
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kw = stats.kruskal(*groups)
    print(f'   groups: {labels}')
    print(f'   one-way ANOVA on nrmse: F = {f_stat:.2f}, p = {p_anova:.3f}')
    print(f'   Kruskal-Wallis        : H = {h_stat:.2f}, p = {p_kw:.3f}')

clbp = per_subject[per_subject['dataset'] == 'cLBP'].dropna(subset=['group_label'])
if clbp['group_label'].nunique() == 2:
    hi = clbp[clbp['group_label'] == 'High']['nrmse']
    lo = clbp[clbp['group_label'] == 'Low']['nrmse']
    u, p_u = stats.mannwhitneyu(hi, lo)
    print(f'\n--- cLBP only: High (n={len(hi)}) vs Low (n={len(lo)}) ---')
    print(f'   mean nrmse: High {hi.mean():.3f}, Low {lo.mean():.3f}')
    print(f'   Mann-Whitney U = {u:.0f}, p = {p_u:.3f}')


def between_subject_tests(df, col, order, label):
    """Omnibus then every pair, Holm-corrected -- the same shape as the
    trial-type section, but on one value per subject because both of these
    factors are between-subject."""
    sub = df[df[col].isin(order)]
    present = [g for g in order if (sub[col] == g).sum() >= 5]
    print(f'\n--- {label} ---')
    print(sub.groupby(col)['nrmse'].agg(n='size', median='median')
          .reindex(present).round(3).to_string())
    if len(present) < 2:
        return
    h, p = stats.kruskal(*[sub.loc[sub[col] == g, 'nrmse'] for g in present])
    print(f'  Kruskal-Wallis: H = {h:.2f}, p = {p:.4f}')
    pairs = list(combinations(present, 2))
    raw = [stats.mannwhitneyu(sub.loc[sub[col] == a, 'nrmse'],
                              sub.loc[sub[col] == b, 'nrmse'])[1]
           for a, b in pairs]
    for (a, b), pr, pa in zip(pairs, raw, multipletests(raw, method='holm')[1]):
        print(f'    {a + " vs " + b:22s} p = {pr:.4f}   (Holm {pa:.4f})')


between_subject_tests(per_subject, 'group_label', GROUP_ORDER,
                      'pain group, all datasets pooled')
between_subject_tests(per_subject, 'dataset', DATASET_ORDER, 'dataset')

# Kept even though the kneeOA panel is gone: pooling groups across datasets
# compares studies as much as groups, because plosONE supplies controls and no
# patients and cLBP supplies patients and no controls. kneeOA is the only
# dataset holding all three groups under one protocol, so this is the number
# to quote if the pooled comparison is challenged.
between_subject_tests(per_subject[per_subject['dataset'] == 'kneeOA'],
                      'group_label', GROUP_ORDER,
                      'pain group within kneeOA only (unconfounded)')


def violin_panel(ax, df, col, order, colors, title):
    sub = df[df[col].isin(order)]
    present = [g for g in order if (sub[col] == g).any()]
    sns.violinplot(data=sub, x=col, y='nrmse', order=present, hue=col,
                   palette=colors, legend=False, cut=0, inner='box',
                   linewidth=1.0, saturation=1.0, ax=ax)
    for coll in ax.collections:
        coll.set_alpha(.88)
    for i, g in enumerate(present):
        vals = sub.loc[sub[col] == g, 'nrmse']
        ax.text(i, vals.max() + .012, f'{vals.median():.2f}', va='bottom',
                ha='center', fontsize=9.5, color='0.15')
        ax.text(i, vals.min() - .035, f'n={len(vals)}', va='top', ha='center',
                fontsize=8, color='0.45')
    ax.set(xlabel='', ylabel='model error (nRMSE)', title=title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(sub['nrmse'].min() - .09, sub['nrmse'].max() + .07)


def swarm_panel(ax, df, col, order, colors, title):
    """One dot per subject. Used where n is small enough to show every person
    -- kneeOA has ~30 per group, so a violin would be smoothing 30 points."""
    sub = df[df[col].isin(order)]
    present = [g for g in order if (sub[col] == g).any()]
    sns.swarmplot(data=sub, x=col, y='nrmse', order=present, hue=col,
                  palette=colors, legend=False, size=4.6, alpha=.92,
                  linewidth=.4, edgecolor='white', ax=ax)
    for i, g in enumerate(present):
        vals = sub.loc[sub[col] == g, 'nrmse']
        ax.hlines(vals.median(), i - .3, i + .3, color='black', lw=2.2,
                  zorder=10)
        ax.text(i, vals.max() + .012, f'{vals.median():.2f}', va='bottom',
                ha='center', fontsize=9.5, color='0.15')
        ax.text(i, vals.min() - .035, f'n={len(vals)}', va='top', ha='center',
                fontsize=8, color='0.45')
    ax.set(xlabel='', ylabel='model error (nRMSE)', title=title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(sub['nrmse'].min() - .09, sub['nrmse'].max() + .07)


# ---- FIGURE: pain group, pooled and then with dataset held fixed ----
# The right panel is the one that survives the confound: pooled, "group" is
# partly "which study", because plosONE supplies controls and no patients and
# cLBP supplies patients and no controls.
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
violin_panel(axes[0], per_subject, 'group_label', GROUP_ORDER, GROUP_COLORS,
             'Fit by pain group\n(all datasets)')
swarm_panel(axes[1], per_subject[per_subject['dataset'] == 'kneeOA'],
            'group_label', GROUP_ORDER, GROUP_COLORS,
            'Pain group within kneeOA\n(one protocol, all three groups)')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_by_paingroup.png',
            dpi=200, bbox_inches='tight')
plt.show()

# ---- FIGURE: dataset on its own ----
fig, ax = plt.subplots(figsize=(5.2, 5.0))
violin_panel(ax, per_subject, 'dataset', DATASET_ORDER, DATASET_COLORS,
             'Fit by dataset')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_by_dataset.png',
            dpi=200, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# Does the group null hold inside every trial type?
# ==================================================================
# The pooled group comparison is null, but it averages over trial types. If
# the model failed specifically on, say, offset trials in high-pain subjects,
# that would be invisible there. So: group compared WITHIN each trial type.
print('\n' + '=' * 70)
print('FIT QUALITY BY TRIAL TYPE x PAIN GROUP')
print('=' * 70)

# The comparison inside each panel is BETWEEN subjects (group), so this uses
# nrmse for the same reason the section above does.
per_subj_type = (scored_groups[scored_groups['trial_type'].isin(COMMON_TRIAL_TYPES)]
                 .groupby(['subject_uid', 'dataset', 'group_label',
                           'trial_type'], observed=True)
                 .agg(nrmse=('nrmse', 'mean'), rmse=('rmse', 'mean'),
                      n_trials=('rmse', 'size'))
                 .reset_index())


def group_within_trial_type(df, scope):
    """Kruskal across pain groups separately inside each trial type, then Holm
    across the four tests -- four looks at the same question is four chances."""
    rows, raw = [], []
    for tt in PLOT_ORDER:
        sub = df[(df['trial_type'] == tt) & df['group_label'].isin(GROUP_ORDER)]
        present = [g for g in GROUP_ORDER if (sub['group_label'] == g).sum() >= 5]
        if len(present) < 2:
            continue
        h, p = stats.kruskal(*[sub.loc[sub['group_label'] == g, 'nrmse']
                               for g in present])
        med = {g: sub.loc[sub['group_label'] == g, 'nrmse'].median()
               for g in present}
        rows.append({'trial_type': tt, 'groups': '/'.join(present),
                     **{f'med_{g}': round(v, 3) for g, v in med.items()},
                     'H': round(h, 2)})
        raw.append(p)
    if not rows:
        return
    out = pd.DataFrame(rows)
    out['p'] = np.round(raw, 4)
    out['p_holm'] = np.round(multipletests(raw, method='holm')[1], 4)
    print(f'\n--- {scope} ---')
    print(out.to_string(index=False))


group_within_trial_type(per_subj_type, 'all datasets pooled (confounded)')
group_within_trial_type(per_subj_type[per_subj_type['dataset'] == 'kneeOA'],
                        'kneeOA only (unconfounded)')

fig, ax = plt.subplots(figsize=(9.5, 5.2))
sub = per_subj_type[per_subj_type['group_label'].isin(GROUP_ORDER)]
sns.violinplot(data=sub, x='trial_type', y='nrmse', order=PLOT_ORDER,
               hue='group_label', hue_order=GROUP_ORDER, palette=GROUP_COLORS,
               cut=0, inner='box', linewidth=.9, saturation=1.0, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.88)
ax.set_xticks(range(len(PLOT_ORDER)))
ax.set_xticklabels([TRIAL_LABELS[tt] for tt in PLOT_ORDER])
ax.set(xlabel='', ylabel='model error (nRMSE)', title='Fit by trial type and pain group')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=8, title='', frameon=False, ncol=3, loc='lower right')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_by_trialtype_group.png',
            dpi=200, bbox_inches='tight')
plt.show()


#%%
# ==================================================================
# How variable is fit quality, within and between subjects?
# ==================================================================
# Everything here uses r, the zero-lag correlation Cecchi 2012 report, rather
# than r2. 
print('\n' + '=' * 70)
print('DISTRIBUTION OF FIT QUALITY (RMSE, VAS)')
print('=' * 70)

# No flatness filter. A trial where the rating barely moves is a real
# response, not an artifact, and dropping those took out hold trials three
# times as often as offset trials -- which shrinks the very trial-type
# contrast this file is measuring. pain_sd is still attached so it can be
# used as a sensitivity analysis, but nothing is excluded on it.
# SENSITIVITY: scored = scored[scored['pain_sd'] >= 2.0]
data_df = pd.read_pickle(TRACES_FILE)
pain_sd = (data_df.groupby(['subject_uid', 'trial_num'])['pain']
           .std().rename('pain_sd').reset_index())
scored = trials_main.merge(pain_sd, on=['subject_uid', 'trial_num'], how='left')
print(f'{len(scored):,} scored trials, no flatness exclusion '
      f"({(scored['pain_sd'] < 2.0).sum():,} have rating SD < 2 VAS and are kept)")

# Spread across a subject's trials, in VAS. RMSE needs no variance-stabilising
# transform -- the Fisher z this used to run on existed only because r is
# bounded at 1 and its spread is compressed near the ceiling.
spread = (scored.groupby(['subject_uid', 'dataset'])
          .agg(median_rmse=('rmse', 'median'), mean_rmse=('rmse', 'mean'),
               sd_rmse=('rmse', 'std'),
               iqr_rmse=('rmse', lambda x: x.quantile(.75) - x.quantile(.25)),
               min_rmse=('rmse', 'min'), max_rmse=('rmse', 'max'),
               median_nrmse=('nrmse', 'median'), median_r=('r', 'median'),
               n_trials=('rmse', 'size'))
          .reset_index())

print(f'\n{len(spread)} subjects')
print('\nper-subject median RMSE (VAS):')
print(spread['median_rmse'].describe().round(2).to_string())
print('\nper-subject spread across trials (SD of RMSE, VAS):')
print(spread['sd_rmse'].describe().round(2).to_string())
print('\nby dataset:')
print(spread.groupby('dataset')[['median_rmse', 'sd_rmse', 'iqr_rmse',
                                 'median_nrmse', 'n_trials']]
      .median().round(3).to_string())

# Exemplars for the trial-by-trial panels further down. Level and spread are
# picked separately because a subject who fits uniformly badly would have low
# spread for an uninteresting reason, so each pick's level is printed next to
# its spread and the reader can judge.
picks = {
    'good fit': spread.loc[spread['median_rmse'].idxmin()],
    'poor fit': spread.loc[spread['median_rmse'].idxmax()],
    'high variance across trials': spread.loc[spread['sd_rmse'].idxmax()],
    'low variance across trials': spread.loc[spread['sd_rmse'].idxmin()],
}
print('\nexemplar subjects (RMSE in VAS):')
print(f'{"":30s}{"subject":<14}{"median":>9}{"SD":>8}{"IQR":>8}'
      f'{"min":>8}{"max":>8}{"trials":>8}')
for label, row in picks.items():
    print(f'{label:<30s}{row["subject_uid"]:<14}{row["median_rmse"]:>9.2f}'
          f'{row["sd_rmse"]:>8.2f}{row["iqr_rmse"]:>8.2f}{row["min_rmse"]:>8.2f}'
          f'{row["max_rmse"]:>8.2f}{int(row["n_trials"]):>8d}')

# Cumulative, not a histogram. RMSE is heavily right-skewed -- most trials
# cluster in a narrow band with a long bad tail -- so on any density axis the
# peak sets the scale and the tail vanishes. The ECDF has no bins and reads
# directly: height at any x is the share of trials with error at or below x,
# and three curves lying on top of each other is the claim being made.
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for ds in DATASET_ORDER:
    g = scored[scored['dataset'] == ds]
    if g.empty:
        continue
    sns.ecdfplot(data=g, x='rmse', ax=ax, lw=2.1, color=DATASET_COLORS[ds],
                 label=f'{ds}  (n={len(g):,}, median {g["rmse"].median():.1f})')
    ax.plot(g['rmse'].median(), .5, 'o', ms=5, color=DATASET_COLORS[ds],
            mec='white', mew=.8, zorder=5)

ax.axhline(.5, color='0.8', lw=.8, zorder=0)
ax.set(xlabel='model error (RMSE, VAS) per trial',
       ylabel='proportion of trials at or below',
       xlim=(0, np.nanpercentile(scored['rmse'], 99)), ylim=(0, 1),
       title=f'Per-trial fit, all {len(scored):,} trials  '
             f'(median RMSE = {scored["rmse"].median():.1f} VAS)')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_fit_quality_distribution.png',
            dpi=200, bbox_inches='tight')
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
          f'median RMSE={row["median_rmse"]:.1f} VAS, '
          f'range {row["min_rmse"]:.1f}-{row["max_rmse"]:.1f})')
    plot_subject_trials(
        data_df[data_df['subject_uid'] == uid],
        {'alpha_bar': p['alpha_bar'], 'gamma_bar': p['gamma_bar'],
         'theta': p['theta']},
        subject_uid=f'{uid} — {label}',
        save_path=f'{FIG_PATH}{datetime.now():%Y%m%d}_exemplar_{tag}_{uid}.png')

#%%
# ==================================================================
# kneeOA only: the same person at the forearm and at the affected knee
# ==================================================================
# Every kneeOA subject was stimulated at both sites with the same protocol, so
# this is a within-subject comparison -- each person is their own control, and
# differences cannot be explained by who the subjects are. Earlier fits pooled
# the two sites into one parameter set per subject, which both cost fit quality
# (median r 0.849 pooled vs 0.879 forearm) and hid this contrast entirely.
print('\n' + '=' * 70)
print('kneeOA: FOREARM vs KNEE, WITHIN SUBJECT')
print('=' * 70)

knee_fits = subjects_all_sites[subjects_all_sites['dataset'] == 'kneeOA'].copy()
if knee_fits['site'].nunique() < 2:
    print('(only one site present in this fit table; skipping)')
else:
    paired = knee_fits.pivot_table(
        index='subject', columns='site',
        values=['r', 'r2', 'alpha_bar', 'gamma_bar', 'theta', 'mse']).dropna()
    print(f'{len(paired)} subjects with a fit at both sites\n')

    print(f'{"":12s}{"forearm":>10}{"knee":>10}{"median Δ":>11}'
          f'{"Wilcoxon p":>13}{"n better at knee":>18}')
    rows = []
    for param, label in [('r', 'fit r'), ('alpha_bar', 'ᾱ drive'),
                         ('gamma_bar', 'γ̄ decay'), ('theta', 'θ threshold')]:
        a, b = paired[(param, 'forearm')], paired[(param, 'knee')]
        stat, pval = stats.wilcoxon(a, b)
        rows.append({'param': param, 'forearm': a.median(), 'knee': b.median(),
                     'delta': (b - a).median(), 'p': pval})
        print(f'{label:12s}{a.median():>10.3f}{b.median():>10.3f}'
              f'{(b - a).median():>+11.3f}{pval:>13.2g}{(b > a).sum():>18d}')

    # gamma_bar is a rate, so 1/gamma_bar is the time pain takes to fade --
    # the interpretable form of whatever difference shows up above.
    tc_f = 1.0 / paired[('gamma_bar', 'forearm')]
    tc_k = 1.0 / paired[('gamma_bar', 'knee')]
    print(f'\ndecay time constant 1/γ̄ (seconds for pain to fade):')
    print(f'  forearm {tc_f.median():.2f}s   knee {tc_k.median():.2f}s   '
          f'ratio {(tc_k / tc_f).median():.2f}x')

    # Fit quality differs between sites, so a parameter difference could partly
    # reflect the model describing the knee less well. Repeating the test on
    # subjects fitted well at BOTH sites shows whether it survives.
    WELL_FIT = 0.8
    good = paired[(paired[('r', 'forearm')] >= WELL_FIT) &
                  (paired[('r', 'knee')] >= WELL_FIT)]
    print(f'\nRestricted to subjects with r >= {WELL_FIT} at both sites '
          f'(n={len(good)}):')
    for param, label in [('alpha_bar', 'ᾱ drive'), ('gamma_bar', 'γ̄ decay'),
                         ('theta', 'θ threshold')]:
        a, b = good[(param, 'forearm')], good[(param, 'knee')]
        if len(good) < 10:
            continue
        _, pval = stats.wilcoxon(a, b)
        print(f'  {label:12s} forearm {a.median():8.3f}   knee {b.median():8.3f}'
              f'   p={pval:.2g}')

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (param, label) in zip(axes, [('r', 'fit quality (r)'),
                                         ('alpha_bar', 'ᾱ  drive'),
                                         ('gamma_bar', 'γ̄  decay rate'),
                                         ('theta', 'θ  threshold (°C)')]):
        a, b = paired[(param, 'forearm')], paired[(param, 'knee')]
        for xa, xb in zip(a, b):
            ax.plot([0, 1], [xa, xb], color='0.75', lw=0.6, zorder=1)
        ax.scatter(np.zeros(len(a)), a, s=14, color='#3d6b8f', zorder=3, label='forearm')
        ax.scatter(np.ones(len(b)), b, s=14, color='#b3202c', zorder=3, label='knee')
        ax.plot([0, 1], [a.median(), b.median()], color='k', lw=2.2, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['forearm', 'knee'])
        ax.set_xlim(-0.35, 1.35); ax.set_title(label)
        if param in ('alpha_bar', 'gamma_bar'):
            ax.set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'kneeOA, same subject at both sites (n={len(paired)}) — '
                 'thick line joins the medians', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_kneeOA_site_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()

#%%
# ==================================================================
# Example fits: the same subject at each site
# ==================================================================
# Picked by how much gamma_bar moves between sites, so the plots show the
# effect the test above reports rather than an arbitrary subject.
if 'paired' in dir() and len(paired) > 2:
    data_df_sites = pd.read_pickle(TRACES_FILE)
    shift = ((paired[('gamma_bar', 'forearm')] - paired[('gamma_bar', 'knee')])
             / paired[('gamma_bar', 'forearm')])
    examples = {'largest site difference': shift.idxmax(),
                'smallest site difference': shift.abs().idxmin()}
    fits_by_unit = subjects_all_sites.set_index('subject_uid')

    for label, subj in examples.items():
        for site in ['forearm', 'knee']:
            # forearm fits keep the plain subject id; knee carries the suffix
            unit = subj if site == 'forearm' else f'{subj}@knee'
            if unit not in fits_by_unit.index:
                continue
            f = fits_by_unit.loc[unit]
            sd = data_df_sites[(data_df_sites['subject_uid'] == subj) &
                               (data_df_sites['site'] == site)]
            print(f'{label}: {subj} @ {site} — γ̄={f["gamma_bar"]:.3f}, '
                  f'r={f["r"]:.3f}')
            plot_subject_trials(
                sd, {'alpha_bar': f['alpha_bar'], 'gamma_bar': f['gamma_bar'],
                     'theta': f['theta']},
                subject_uid=f'{subj} — {site}',
                save_path=f'{FIG_PATH}{datetime.now():%Y%m%d}_site_example_'
                          f'{subj}_{site}.png')

# %%
