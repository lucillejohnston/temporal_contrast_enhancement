# -*- coding: utf-8 -*-
"""
Score both models inside the stimulus periods, not just over the whole trial.

Periods are MEASURED from each trial's own temperature trace by trial_windows.py:

    A   plateau at T1   (the baseline plateau)
    B   the perturbation -- up for offset trials, down for onset
    C   back at T1      -- ~20s

In hold trials there is no step, so A and B remain 5s each by convention and
only C is measured.

Author: Lucille Johnston
Updated: 11/02/26
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
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import (prepare_trials_for_optimization,
                                              simulate_trials_analytic,
                                              pack_trials, simulate_trials_full,
                                              fit_metrics)
from trial_windows import measure_all_trials

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
TRACES_FILE = DATA_PATH + 'combined_traces_1Hz.pkl'

MIN_POINTS = 5     # r on fewer than this is not worth computing

MIN_PAIN_SD = None  # VAS; None keeps every scorable window - keep for later as a sensitivity analysis

# kneeOA was stimulated at forearm and knee; everything here is forearm, which
# is what extract_metrics.py (and so the period boundaries) assumes, and what
# makes kneeOA comparable to the single-site plosONE and cLBP.
SITE = 'forearm'

STAMP = f'{datetime.now():%Y%m%d}'
OUT_CSV = RESULTS_PATH + f'{STAMP}_model_fits_per_period.csv'
WINDOWS_CSV = RESULTS_PATH + f'{STAMP}_measured_trial_windows.csv'
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

simple = simple.set_index('subject_uid')
full = full.set_index('subject_uid')

# One body site, before anything else: kneeOA was stimulated at forearm and
# knee, and a fit unit straddling both is not comparable to the single-site
# plosONE and cLBP.
data_df['site'] = data_df['site'].fillna(SITE)
data_df = data_df[data_df['site'] == SITE]

# Measure each trial's A/B/C windows from its own temperature trace.
windows = measure_all_trials(data_df)
windows.to_csv(WINDOWS_CSV, index=False)
measured = windows['status'] == 'ok'
print(f'\nperiod windows measured for {measured.sum():,}/{len(windows):,} trials '
      f'({measured.mean():.1%}) -> {os.path.basename(WINDOWS_CSV)}')
if (~measured).any():
    print('  could not be measured:',
          windows.loc[~measured, 'status'].value_counts().to_dict())
print(windows[measured].groupby(['dataset', 'trial_type'])
      [['A_dur', 'B_dur', 'C_dur']].median().round(1).to_string())

# Windows are in aligned_time, the same clock the traces use. Within one site
# (subject_uid, trial_num) is unique, which is the key the scoring loop holds.
boundary_lookup = windows[measured].set_index(['subject_uid', 'trial_num'])

PERIODS = ['A', 'B', 'C']


#%%
# ------------------------------------------------------------------
# Recompute predictions and score each window
# ------------------------------------------------------------------
window_rows = []
n_trials_missing_boundaries = 0

fit_units = sorted(simple.index)
units_with_eq1 = set(full.index)
print(f'\nscoring {len(fit_units)} {SITE} fits '
      f'({len(set(fit_units) & units_with_eq1)} also have a usable Eq. 1 fit)...')

for i, fit_unit in enumerate(fit_units):
    subject_traces = data_df[data_df['subject_uid'] == fit_unit]
    trials = prepare_trials_for_optimization(subject_traces)
    if not trials:
        continue

    eq2_params = simple.loc[fit_unit]
    pred_eq2_per_trial = simulate_trials_analytic(
        {'alpha_bar': eq2_params['alpha_bar'],
         'gamma_bar': eq2_params['gamma_bar'],
         'theta': eq2_params['theta']}, trials)

    pred_eq1_padded = None
    if fit_unit in units_with_eq1:
        eq1_params = full.loc[fit_unit]
        pred_eq1_padded = simulate_trials_full(
            {'alpha': eq1_params['alpha'], 'beta': eq1_params['beta'],
             'gamma': eq1_params['gamma'], 'lam': eq1_params['lam'],
             'theta': eq1_params['theta']}, pack_trials(trials))

    dataset = subject_traces['dataset'].iloc[0]
    subject_number = int(subject_traces['subject_orig'].iloc[0])

    for trial_idx, trial in enumerate(trials):
        trial_time = trial['time']
        observed = trial['pain']
        pred_eq2 = pred_eq2_per_trial[trial_idx]
        # simulate_trials_full returns a padded matrix, so trim to this trial
        pred_eq1 = (pred_eq1_padded[trial_idx, :len(trial_time)]
                    if pred_eq1_padded is not None else None)

        trial_info = {
            'subject_uid': fit_unit, 'dataset': dataset,
            'study': subject_traces['study'].iloc[0],
            'group_label': subject_traces['group_label'].iloc[0],
            'subject_orig': subject_number, 'trial_num': trial['trial_num'],
            'trial_type': trial['trial_type']}

        boundary_key = (fit_unit, trial['trial_num'])
        boundaries = (boundary_lookup.loc[boundary_key]
                      if boundary_key in boundary_lookup.index else None)
        if boundaries is None:
            n_trials_missing_boundaries += 1

        # Each window is a boolean mask over this trial's samples.
        window_masks = {'full': np.ones(len(trial_time), dtype=bool)}
        if boundaries is not None:
            for period in PERIODS:
                start, end = boundaries[f'{period}_start'], boundaries[f'{period}_end']
                if pd.notna(start) and pd.notna(end):
                    window_masks[period] = ((trial_time >= start)
                                            & (trial_time <= end))

        for window_name, in_window in window_masks.items():
            n_samples = int(in_window.sum())
            obs_window = observed[in_window]

            row = {**trial_info, 'period': window_name, 'n_points': n_samples,
                   'pain_sd': float(np.std(obs_window)) if n_samples else np.nan,
                   't_start': float(trial_time[in_window][0]) if n_samples else np.nan,
                   't_end': float(trial_time[in_window][-1]) if n_samples else np.nan,
                   'mean_obs': float(np.mean(obs_window)) if n_samples else np.nan,
                   'mean_eq2': float(np.mean(pred_eq2[in_window])) if n_samples else np.nan,
                   'mean_eq1': (float(np.mean(pred_eq1[in_window]))
                                if (n_samples and pred_eq1 is not None) else np.nan),
                   'auc_obs': (float(np.trapezoid(obs_window, trial_time[in_window]))
                               if n_samples > 1 else np.nan),
                   'auc_eq2': (float(np.trapezoid(pred_eq2[in_window], trial_time[in_window]))
                               if n_samples > 1 else np.nan)}

            # FIT QUALITY in the window
            if n_samples >= MIN_POINTS:
                metrics_eq2 = fit_metrics(obs_window, pred_eq2[in_window])
                row.update({'r_eq2': metrics_eq2['r'], 'r2_eq2': metrics_eq2['r2'],
                            'mse_eq2': metrics_eq2['mse'],
                            'obs_range': float(obs_window.max() - obs_window.min())})
                if pred_eq1 is not None:
                    metrics_eq1 = fit_metrics(obs_window, pred_eq1[in_window])
                    row.update({'r_eq1': metrics_eq1['r'], 'r2_eq1': metrics_eq1['r2'],
                                'mse_eq1': metrics_eq1['mse']})
            window_rows.append(row)

    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(fit_units)}')

window_fits = pd.DataFrame(window_rows)
window_fits['visit'] = window_fits['trial_num'] // 100
window_fits['trial_in_visit'] = window_fits['trial_num'] % 100
window_fits['trial_order'] = (window_fits.groupby(['subject_uid', 'visit'])
                              ['trial_in_visit'].rank(method='dense').astype(int))
window_fits.to_csv(OUT_CSV, index=False)


#%%
# ==================================================================
# WHY WINDOW AT ALL: example OA and OH trials with A/B/C drawn on
# ==================================================================
# This is the motivating figure and belongs before any of the numbers. A
# single whole-trial r hides that the model tracks the rating well through the
# baseline plateau and the step, then comes apart in period C. Nothing below
# makes sense until that is on screen.
#
# It cannot sit any higher in the file than this: it needs window_fits, which
# is what the cell above builds.
TRIAL_COLORS = {'offset': '#026A07', 'onset': '#DA1C1C',
                't1_hold': '#CFCFCF', 't2_hold': '#8A8A8A'}
TRIAL_LABELS = {'offset': 'Offset (OA)', 'onset': 'Onset (OH)',
                't1_hold': 'Hold (T1)', 't2_hold': 'Hold (T2)'}
# Each stepped trial sits next to its temperature-matched hold: the hold is at
# the same plateau the stepped trial returns to (cLBP offset and t1_hold are
# both 44 degC, onset and t2_hold both 45), so the only thing differing within
# an adjacent pair is whether a step happened. Same order as
# analyze_model_fits.py, kept in sync by hand.
PLOT_ORDER = ['t1_hold', 'offset', 't2_hold', 'onset']

# Same palette as analyze_model_fits.py: light -> dark with increasing pain.
# The database spells it both 'Control' and 'control' depending on the study,
# so labels are capitalised before anything groups or plots them.
GROUP_ORDER = ['Control', 'Low', 'High']
GROUP_COLORS = {'Control': '#DBBBAF', 'Low': '#91472A', 'High': '#491400'}
# A and B are grey because they are the windows where the model behaves; C is
# tinted so the eye lands on the one that does not.
PERIOD_SHADE = {'A': '#EDEDED', 'B': '#D6D6D6', 'C': '#F7D9A0'}

N_CANDIDATES = 6
EXAMPLE_PICK = {'offset': 4, 'onset': 2}   # rank from the contact sheet


def plot_trial_windows(fit_unit, trial_num, axis=None, show_eq1=False,
                       compact=False):
    """
    One trial: stimulus, ratings, model prediction, with the A/B/C windows
    shaded and each window's r printed on it.

    Drawn because a per-window number on its own does not show whether a low r
    means the model has the wrong shape or merely the wrong level -- and those
    call for different conclusions.
    """
    subject_rows = data_df[data_df['subject_uid'] == fit_unit]
    trials = prepare_trials_for_optimization(subject_rows)
    trial = next((x for x in trials if x['trial_num'] == trial_num), None)
    if trial is None:
        if axis is not None:
            axis.set_axis_off()
        return None

    eq2 = simple.loc[fit_unit]
    pred2 = simulate_trials_analytic(
        {'alpha_bar': eq2['alpha_bar'], 'gamma_bar': eq2['gamma_bar'],
         'theta': eq2['theta']}, [trial])[0]
    pred1 = None
    if show_eq1 and fit_unit in set(full.index):
        e1 = full.loc[fit_unit]
        pred1 = simulate_trials_full(
            {'alpha': e1['alpha'], 'beta': e1['beta'], 'gamma': e1['gamma'],
             'lam': e1['lam'], 'theta': e1['theta']},
            pack_trials([trial]))[0][:len(trial['time'])]

    ax = axis or plt.subplots(figsize=(9, 4))[1]
    t, obs = trial['time'], trial['pain']
    ttype = trial['trial_type']

    rows = window_fits[(window_fits['subject_uid'] == fit_unit)
                       & (window_fits['trial_num'] == trial_num)]
    for period in PERIODS:
        r = rows[rows['period'] == period]
        if r.empty or pd.isna(r['t_start'].iloc[0]):
            continue
        lo, hi = r['t_start'].iloc[0], r['t_end'].iloc[0]
        ax.axvspan(lo, hi, color=PERIOD_SHADE[period], lw=0, zorder=0)
        # Labelled with the LEVEL error, not r. A and B are only ~6 samples
        # at 1 Hz, and an r between two smooth monotone series on six points
        # is near 1 almost by construction -- 42% of A windows exceed 0.95.
        # So an r comparison across these windows is largely comparing a
        # 6-point correlation with a 21-point one. The signed error in VAS is
        # immune to window length and answers the actual question: is the
        # model too high or too low here?
        bias = r['mean_obs'].iloc[0] - r['mean_eq2'].iloc[0]
        rm = np.sqrt(r['mse_eq2'].iloc[0])
        ax.text((lo + hi) / 2, 103,
                f'{period}\n{bias:+.0f} VAS' if pd.notna(bias) else period,
                ha='center', va='bottom', fontsize=7 if compact else 9,
                fontweight='bold' if period == 'C' else 'normal')

    ax.plot(t, obs, color='k', lw=2, label='rating', zorder=3)
    ax.plot(t, pred2, color=TRIAL_COLORS.get(ttype, 'crimson'), lw=2,
            label='model', zorder=3)
    if pred1 is not None:
        ax.plot(t, pred1, color='#3d6b8f', lw=1.4, ls='--', label='Eq. 1',
                zorder=3)

    ax2 = ax.twinx()
    ax2.plot(t, trial['temperature'], color='0.45', lw=1.1, ls=':', zorder=1)
    ax2.set_ylim(30, 52)
    ax2.set_ylabel('°C', color='0.45', fontsize=8)
    ax2.tick_params(labelsize=7, colors='0.45')
    ax2.spines[['top']].set_visible(False)

    whole = rows[rows['period'] == 'full']
    rmse_whole = (np.sqrt(whole['mse_eq2'].iloc[0]) if len(whole) else np.nan)
    ax.set(ylabel='VAS', ylim=(-5, 118))
    ax.set_title(f'{TRIAL_LABELS.get(ttype, ttype)}   {fit_unit} trial '
                 f'{trial_num}   ·   whole-trial RMSE = {rmse_whole:.1f} VAS',
                 fontsize=8.5 if compact else 10)
    ax.tick_params(labelsize=8)
    ax.spines[['top']].set_visible(False)
    return ax


def c_deficit(trial_type):
    """Trials of this type ranked by how much worse period C is than A and B,
    in VAS.

    The point of the figure is that the model does not fail uniformly across a
    trial, so candidates are ordered by exactly that gap rather than by overall
    fit. The median gap is printed alongside so an example can be described as
    typical or not, instead of being quietly cherry-picked.

    Measured on RMSE rather than r because A and B hold only ~6 samples each,
    where r saturates near 1 regardless of how far off the model is.
    """
    w = window_fits[(window_fits['trial_type'] == trial_type)
                    & window_fits['period'].isin(PERIODS)].copy()
    w['rmse'] = np.sqrt(w['mse_eq2'])
    wide = (w.pivot_table(index=['subject_uid', 'trial_num'],
                          columns='period', values='rmse')
            .dropna(subset=PERIODS))
    wide['gap'] = wide['C'] - wide[['A', 'B']].mean(axis=1)
    return wide.sort_values('gap', ascending=False)


# ---- contact sheet: flip through these, then set EXAMPLE_PICK ----
fig, axes = plt.subplots(2, N_CANDIDATES, figsize=(4.0 * N_CANDIDATES, 6.4),
                         squeeze=False)
for r_i, tt in enumerate(['offset', 'onset']):
    ranked = c_deficit(tt)
    print(f'\n{TRIAL_LABELS[tt]}: {len(ranked)} scorable trials, '
          f'median (A,B) - C gap = {ranked["gap"].median():+.3f}')
    print(ranked.head(N_CANDIDATES).round(3).to_string())
    for c_i in range(N_CANDIDATES):
        ax = axes[r_i][c_i]
        if c_i < len(ranked):
            uid, tnum = ranked.index[c_i]
            plot_trial_windows(uid, tnum, axis=ax, compact=True)
            ax.text(.02, .02, f'rank {c_i}  gap {ranked["gap"].iloc[c_i]:+.2f}',
                    transform=ax.transAxes, fontsize=7, color='0.4')
        else:
            ax.set_axis_off()
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{STAMP}_period_example_candidates.png', dpi=130,
            bbox_inches='tight')
plt.show()

# ---- the picked pair, slide-sized ----
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
for ax, tt in zip(axes, ['offset', 'onset']):
    ranked = c_deficit(tt)
    uid, tnum = ranked.index[EXAMPLE_PICK[tt]]
    plot_trial_windows(uid, tnum, axis=ax)
axes[0].legend(fontsize=8, loc='lower left', frameon=False)
axes[-1].set_xlabel('time (s)', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{STAMP}_period_examples_OA_OH.png', dpi=200,
            bbox_inches='tight')
plt.show()


#%%
# ------------------------------------------------------------------
# Does windowing change the picture?
# ------------------------------------------------------------------
print('\n' + '=' * 70)
print('FIT BY PERIOD -- whole trial vs the windows that matter')
print('=' * 70)

# Every window with an mse is scorable. Previously this required r to exist,
# which silently dropped windows where the rating was flat -- those have a
# perfectly well-defined error, they just have no correlation.
ok = window_fits[window_fits['mse_eq2'].notna()].copy()
ok['rmse'] = np.sqrt(ok['mse_eq2'])
ok['signed'] = ok['mean_obs'] - ok['mean_eq2']    # + = model too LOW
# SENSITIVITY: uncomment to drop windows where the rating hardly moved
# ok = ok[ok['pain_sd'] >= MIN_PAIN_SD]
print(f'{len(ok):,} scorable rows (every window kept, flat ratings included)')

MAIN_TYPES = [t for t in ['offset', 'onset', 't1_hold', 't2_hold']
              if t in set(ok['trial_type'])]

print('\nmedian per window, VAS:')
print(ok[ok['period'].isin(['full'] + PERIODS)]
      .pivot_table(index='period', columns='trial_type',
                   values=['rmse', 'signed'], aggfunc='median')
      .round(2).to_string())

# ------------------------------------------------------------------
# WHEN does the model break? A vs B vs C, pooled over trial types
# ------------------------------------------------------------------
# Before asking WHICH trials fail, establish WHEN within a trial it fails.
# Pooled over the four main trial types so this is a statement about the
# stimulus phase rather than about any one condition -- the trial-specific
# version is the next cell.
#
# Paired within subject: every subject contributes one value per period, so a
# subject the model fits badly throughout cannot create the effect. RMSE is a
# per-sample mean, so it is comparable across windows of different length --
# which r was not, A and B holding only ~6 samples against C's ~21.
def stars(p):
    return ('***' if p < .001 else '**' if p < .01 else
            '*' if p < .05 else 'n.s.')


# Bare letters. 'B = the step' was wrong on half of these figures: hold
# trials have no step, B is simply the matching stretch of time in them, and
# that is the whole point of using them as controls.
PERIOD_LABEL = {'A': 'A', 'B': 'B', 'C': 'C'}

# ---- FIGURE 1: when does it break? stepped vs hold, per period ----
# Stepped and hold are pooled WITHIN each side of the split, which works here
# because this panel is about magnitude: offset and onset are both badly
# wrong in period C, so the error does not cancel. (It would cancel on a
# signed measure, which is why figure 2 keeps them apart.)
print('\n--- when does it break? stepped vs hold, paired within subject ---')
abc = ok[ok['period'].isin(PERIODS) & ok['trial_type'].isin(PLOT_ORDER)].copy()
abc['kind'] = np.where(abc['trial_type'].isin(['offset', 'onset']),
                       'stepped', 'hold')
subj_kind = (abc.groupby(['subject_uid', 'period', 'kind'])[['rmse', 'signed']]
             .mean().reset_index())

kind_p = {}
for per in PERIODS:
    sub = subj_kind[subj_kind['period'] == per]
    pr = sub.pivot(index='subject_uid', columns='kind', values='rmse').dropna()
    _, p = stats.wilcoxon(pr['stepped'], pr['hold'])
    kind_p[per] = p
    print(f'  {per}:  stepped {pr["stepped"].median():5.2f} vs hold '
          f'{pr["hold"].median():5.2f}   diff '
          f'{(pr["stepped"] - pr["hold"]).median():+5.2f} VAS   '
          f'p = {p:.2e} {stars(p)}   n={len(pr)}')

# Blue for the pooled stepped trials: it has to read as its own condition,
# but it cannot be green or red, since those are reserved for OA and OH
# individually in the figure below and this violin is the two of them pooled.
# The holds stay grey -- they are the control.
KIND_COLORS = {'hold': '#CFCFCF', 'stepped': '#2B6CA3'}

fig, ax = plt.subplots(figsize=(6.6, 5.2))
sns.violinplot(data=subj_kind, x='period', y='rmse', order=PERIODS,
               hue='kind', hue_order=['hold', 'stepped'], split=True,
               palette=KIND_COLORS, cut=0, inner='quart', linewidth=1.0,
               saturation=1.0, gap=.22, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.9)

ktop = subj_kind['rmse'].max()
kspan = ktop - subj_kind['rmse'].min()
for i, per in enumerate(PERIODS):
    sub = subj_kind[subj_kind['period'] == per]
    for off, kind in [(-.2, 'hold'), (.2, 'stepped')]:
        med = sub.loc[sub['kind'] == kind, 'rmse'].median()
        ax.text(i + off, -.03 * kspan, f'{med:.1f}', ha='center', va='top',
                fontsize=8.5, color='0.2')
    y = ktop + .04 * kspan
    ax.plot([i - .2, i - .2, i + .2, i + .2],
            [y, y + .02 * kspan, y + .02 * kspan, y], lw=1.1, color='0.25')
    ax.text(i, y + .028 * kspan, stars(kind_p[per]), ha='center', va='bottom',
            fontsize=12 if kind_p[per] < .05 else 9, color='0.15')
ax.set_ylim(-.09 * kspan, ktop + .14 * kspan)
ax.set_xticks(range(len(PERIODS)))
ax.set_xticklabels([PERIOD_LABEL[p] for p in PERIODS], fontsize=9)
ax.set(xlabel='', ylabel='model error (RMSE, VAS) per subject',
       title='Model Performance By Period')
ax.spines[['top', 'right']].set_visible(False)
# To the right of the axes: inside it sat on the period-A bracket, and above
# it ran into the title.
ax.legend(fontsize=9, frameon=False, title='', loc='upper left',
          bbox_to_anchor=(1.01, 1.0))
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{STAMP}_period_abc_stepped_vs_hold.png', dpi=200,
            bbox_inches='tight')
plt.show()

# ---- FIGURE 2: which way does it miss? ----
# Holds are dropped: the model tracks them, so there is no miss to describe.
# Offset and onset are kept APART rather than pooled, because their misses
# point in opposite directions in period C (-4.4 vs +1.2 VAS) and averaging
# them would cancel the effect into nothing.
subj_tt = (ok[ok['period'].isin(PERIODS)
              & ok['trial_type'].isin(['offset', 'onset'])]
           .groupby(['subject_uid', 'period', 'trial_type'])['signed']
           .mean().reset_index())

print('\n--- Signed Error, stepped trials only ---')
tt_p = {}
for tt in ['offset', 'onset']:
    wide = (subj_tt[subj_tt['trial_type'] == tt]
            .pivot(index='subject_uid', columns='period', values='signed'))
    for a, b in [('A', 'C'), ('B', 'C')]:
        pr = wide[[a, b]].dropna()
        _, p = stats.wilcoxon(pr[a], pr[b])
        tt_p[(tt, a, b)] = p
    meds = wide.median()
    print(f'  {TRIAL_LABELS[tt]}:  A {meds["A"]:+5.2f}  B {meds["B"]:+5.2f}  '
          f'C {meds["C"]:+5.2f}   A-vs-C p = {tt_p[(tt, "A", "C")]:.2e} '
          f'{stars(tt_p[(tt, "A", "C")])}   B-vs-C p = '
          f'{tt_p[(tt, "B", "C")]:.2e} {stars(tt_p[(tt, "B", "C")])}')

fig, ax = plt.subplots(figsize=(6.8, 5.2))
sns.violinplot(data=subj_tt, x='period', y='signed', order=PERIODS,
               hue='trial_type', hue_order=['offset', 'onset'],
               palette=TRIAL_COLORS, cut=0, inner='quart', linewidth=1.0,
               saturation=1.0, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.85)
ax.axhline(0, color='0.3', lw=1.3, ls=(0, (4, 3)), zorder=0)

stop, sbot = subj_tt['signed'].max(), subj_tt['signed'].min()
sspan = stop - sbot
for i, per in enumerate(PERIODS):
    for off, tt in [(-.2, 'offset'), (.2, 'onset')]:
        v = subj_tt[(subj_tt['period'] == per)
                    & (subj_tt['trial_type'] == tt)]['signed']
        ax.text(i + off, sbot - .02 * sspan, f'{v.median():+.1f}',
                ha='center', va='top', fontsize=8.5, color='0.2')
ax.set_ylim(sbot - .1 * sspan, stop + .06 * sspan)
ax.set_xticks(range(len(PERIODS)))
ax.set_xticklabels([PERIOD_LABEL[p] for p in PERIODS], fontsize=9)
ax.set(xlabel='', ylabel='rating − model (VAS) per subject',
       title='Signed Error, Stepped Trials Only')
ax.spines[['top', 'right']].set_visible(False)
h, lab = ax.get_legend_handles_labels()
ax.legend(h, [TRIAL_LABELS[t] for t in lab], fontsize=9, frameon=False,
          loc='upper right', title='')
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{STAMP}_period_abc_signed.png', dpi=200,
            bbox_inches='tight')
plt.show()
#%%
# ------------------------------------------------------------------
# PERIOD C: is the model worse on OA and OH than on the holds?
# ------------------------------------------------------------------
# Each stepped trial is tested against its TEMPERATURE-MATCHED hold rather
# than against the holds pooled. The hold trials sit at the same plateau the
# stepped trial returns to -- cLBP offset and t1_hold are both 44 degC,
# onset and t2_hold both 45 -- so the only thing differing within a pair is
# whether a step happened. Pooling the two holds would blur that, since they
# are a degree apart and the model does not fit them equally.
#
# Paired within subject, so a subject the model fits badly everywhere cannot
# create the effect; only subjects with both trial types contribute.
PAIRS = [('offset', 't1_hold'), ('onset', 't2_hold')]

pc = ok[ok['period'] == 'C']
subj_c = (pc.groupby(['subject_uid', 'trial_type'])[['rmse', 'signed']]
          .mean().reset_index())
wide_rmse = subj_c.pivot(index='subject_uid', columns='trial_type',
                         values='rmse')
wide_signed = subj_c.pivot(index='subject_uid', columns='trial_type',
                           values='signed')

print('\n--- period C, stepped vs its temperature-matched hold '
      '(paired, per subject) ---')
diffs = {}
for stepped, hold in PAIRS:
    pair = wide_rmse[[stepped, hold]].dropna()
    d = pair[stepped] - pair[hold]
    diffs[stepped] = d
    stat, p = stats.wilcoxon(pair[stepped], pair[hold])
    sp = wide_signed[[stepped, hold]].dropna()
    print(f'  {TRIAL_LABELS[stepped]} vs {TRIAL_LABELS[hold]}   n={len(pair)}')
    print(f'     RMSE   {pair[stepped].median():6.2f} vs {pair[hold].median():6.2f}'
          f'   diff {d.median():+6.2f} VAS   Wilcoxon p = {p:.2e}')
    print(f'     signed {sp[stepped].median():+6.2f} vs {sp[hold].median():+6.2f}'
          f'   (+ = model too low, - = model too high)')


# Both figures carry the SAME stars, from the paired Wilcoxon above. The
# left-hand figure draws the four conditions unpaired, but the subjects in
# them are the same people, so the paired test is the correct one either way.
# (Unpaired would also clear ***, at p = 3e-06 rather than 5e-16 -- ignoring
# the pairing throws away ten orders of magnitude.)
pair_p = {s: stats.wilcoxon(diffs[s])[1] for s, _ in PAIRS}

# ---- FIGURE 1: the four conditions ----
fig, ax = plt.subplots(figsize=(6.4, 5.2))
pc_long = subj_c[subj_c['trial_type'].isin(PLOT_ORDER)]
sns.violinplot(data=pc_long, x='trial_type', y='rmse', order=PLOT_ORDER,
               hue='trial_type', palette=TRIAL_COLORS, legend=False,
               cut=0, inner='box', linewidth=1.0, saturation=1.0, ax=ax)
for coll in ax.collections:
    coll.set_alpha(.88)
for i, tt in enumerate(PLOT_ORDER):
    v = pc_long.loc[pc_long['trial_type'] == tt, 'rmse']
    ax.text(i, v.max() + .5, f'{v.median():.1f}', ha='center', va='bottom',
            fontsize=9.5, color='0.15')
ax.set_xticks(range(len(PLOT_ORDER)))
ax.set_xticklabels([TRIAL_LABELS[t].replace(' ', '\n') for t in PLOT_ORDER])
ax.set(xlabel='', ylabel='period C error (RMSE, VAS) per subject',
       title='Period C fit by trial type')
ax.spines[['top', 'right']].set_visible(False)

# Brackets joining each stepped trial to its matched hold -- they are adjacent
# in PLOT_ORDER precisely so this reads without a legend.
top = pc_long['rmse'].max()
for level, (s, h) in enumerate(PAIRS):
    x1, x2 = sorted((PLOT_ORDER.index(s), PLOT_ORDER.index(h)))
    y = top + 3.0 + level * 0.0
    ax.plot([x1, x1, x2, x2], [y, y + .9, y + .9, y], lw=1.1, color='0.25')
    ax.text((x1 + x2) / 2, y + 1.2, stars(pair_p[s]), ha='center',
            va='bottom', fontsize=12 if pair_p[s] < .05 else 9, color='0.15')
ax.set_ylim(None, top + 7)

plt.tight_layout()
plt.savefig(f'{FIG_PATH}{STAMP}_period_c_by_trial_type.png', dpi=200,
            bbox_inches='tight')
plt.show()

#%%
# ==================================================================
# Fit quality per window, as a table
# ==================================================================
print('\n' + '=' * 70)
print('r AND r2 BY WINDOW AND TRIAL TYPE')
print('=' * 70)
print('r judges shape only; r2 also penalises level and scale, so the two')
print('disagree exactly where the model has the right shape at the wrong')
print('level -- which is what offset analgesia looks like.\n')

scorable_rows = window_fits[window_fits['mse_eq2'].notna()].copy()
scorable_rows['rmse_eq2'] = np.sqrt(scorable_rows['mse_eq2'])
scorable_rows['rmse_eq1'] = np.sqrt(scorable_rows['mse_eq1'])
scorable_rows['signed'] = scorable_rows['mean_obs'] - scorable_rows['mean_eq2']
# SENSITIVITY: uncomment to drop windows where the rating hardly moved
# scorable_rows = scorable_rows[scorable_rows['pain_sd'] >= MIN_PAIN_SD]
# r and r2 are kept in this table only so the disagreement described above is
# visible side by side; rmse and signed are the numbers to read.
summary = (scorable_rows.groupby(['trial_type', 'period'])
           .agg(n=('mse_eq2', 'size'),
                rmse_eq2=('rmse_eq2', 'median'), signed=('signed', 'median'),
                rmse_eq1=('rmse_eq1', 'median'),
                r_eq2=('r_eq2', 'median'), r2_eq2=('r2_eq2', 'median'))
           .round(3))
print(summary.to_string())
summary.to_csv(RESULTS_PATH + f'{STAMP}_window_fit_summary.csv')
print(f'\n-> {STAMP}_window_fit_summary.csv')

#%%
# ==================================================================
# Period C fit quality by pain group and by hold-trial classification
# ==================================================================
# Period C is where offset analgesia lives, so how well the model does THERE is
# the question -- not how well it does over a whole trial, which is dominated
# by the rise phase everything fits.
#
# Two groupings:
#   pain group        Control / Low / High, carried through from the traces
#   hold slope        habituator / sensitizer / no trend, from
#                     holdslope_classifications.csv -- the slope of pain across
#                     period C of the hold trials. Regenerated 28 Oct against
#                     the measured A/B/C windows, so it is scored in the same
#                     window as the fits below.
#
# Scored per subject, not per trial: trials within a subject are not
# independent, and a subject with many trials would otherwise count for more.
# r is averaged on the Fisher z scale because r is bounded at 1.
print('\n' + '=' * 70)
print('PERIOD C FIT BY PAIN GROUP AND BY HOLD CLASSIFICATION')
print('=' * 70)

# Subject ids in the classification file are (within-dataset number + dataset
# offset), the convention combining_datasets.py uses. subject_uid carries the
# within-dataset number, which for cLBP is the canonical DPOP_ID.
CLASS_OFFSETS = {'plosONE': 0, 'kneeOA': 1000, 'cLBP': 2000}

holdslope = pd.read_csv(DATA_PATH + 'holdslope_classifications.csv')
holdslope = holdslope[['dataset', 'subject', 'classification', 'observed_mean_slope']]

period_c = window_fits[(window_fits['period'] == 'C')
                       & window_fits['mse_eq2'].notna()].copy()
# SENSITIVITY: uncomment to drop windows where the rating hardly moved
# period_c = period_c[period_c['pain_sd'] >= MIN_PAIN_SD]
period_c['rmse_eq2'] = np.sqrt(period_c['mse_eq2'])
period_c['rmse_eq1'] = np.sqrt(period_c['mse_eq1'])
period_c['signed'] = period_c['mean_obs'] - period_c['mean_eq2']
period_c['class_subject'] = (period_c['subject_uid'].str.split('_').str[-1].astype(int)
                             + period_c['dataset'].map(CLASS_OFFSETS))

# Pain group and habituator/sensitizer are BETWEEN-subject, and raw RMSE
# tracks how wide a range a subject uses on the VAS (rho = 0.78), so these
# comparisons run on nrmse -- RMSE as a fraction of that subject's own range.
# Without it a group difference would largely be a difference in rating style
# between the studies the groups come from.
_vas_range = (data_df.groupby('subject_uid')['pain']
              .agg(lambda x: x.quantile(.95) - x.quantile(.05))
              .rename('vas_range'))
period_c = period_c.merge(_vas_range, on='subject_uid', how='left')
period_c['nrmse'] = period_c['rmse_eq2'] / period_c['vas_range'].replace(0, np.nan)

# One row per subject, keeping the offset trials separate: offset is the trial
# type the effect is defined on, and its period C behaves very differently
# from the holds, so pooling them would hide any group difference.
def subject_level(df):
    return (df.groupby(['subject_uid', 'dataset', 'group_label', 'class_subject'])
            .agg(nrmse=('nrmse', 'mean'), rmse_eq2=('rmse_eq2', 'mean'),
                 rmse_eq1=('rmse_eq1', 'mean'), signed=('signed', 'mean'),
                 n_trials=('nrmse', 'size'))
            .reset_index())

subject_c = subject_level(period_c).merge(
    holdslope, left_on=['dataset', 'class_subject'],
    right_on=['dataset', 'subject'], how='left')
subject_c_offset = subject_level(period_c[period_c['trial_type'] == 'offset']).merge(
    holdslope, left_on=['dataset', 'class_subject'],
    right_on=['dataset', 'subject'], how='left')

unmatched = subject_c['classification'].isna().sum()
print(f'{len(subject_c)} subjects with a scorable period C; '
      f'{unmatched} without a hold classification')
print(f'{len(subject_c_offset)} of them have scorable period C on offset trials')


def compare_period_c(df, group_col, label):
    """Kruskal-Wallis across groups on the Fisher z, reported back as r."""
    sub = df.dropna(subset=[group_col, 'nrmse'])
    if sub[group_col].nunique() < 2:
        print(f'\n{label}: too few groups'); return
    summary = (sub.groupby(group_col)
               .agg(n=('nrmse', 'size'), median_nrmse=('nrmse', 'median'),
                    mean_nrmse=('nrmse', 'mean'))
               .round(3))
    print(f'\n{label}')
    print(summary.to_string())
    samples = [g['nrmse'].values for _, g in sub.groupby(group_col) if len(g) >= 5]
    if len(samples) >= 2:
        stat, pval = stats.kruskal(*samples)
        print(f'  Kruskal-Wallis p = {pval:.4f}'
              f'{"   <-- p < .05" if pval < 0.05 else ""}')


print('\n' + '-' * 70)
print('ALL TRIAL TYPES POOLED')
print('-' * 70)
compare_period_c(subject_c, 'group_label', 'by pain group')
compare_period_c(subject_c, 'classification', 'by hold-trial classification')

print('\n' + '-' * 70)
print('OFFSET TRIALS ONLY -- the window the effect is defined on')
print('-' * 70)
compare_period_c(subject_c_offset, 'group_label', 'by pain group')
compare_period_c(subject_c_offset, 'classification', 'by hold-trial classification')

# Pain group is badly confounded with dataset -- every plosONE subject is a
# control and cLBP has no controls at all -- so a pooled difference between
# Control and High could just be plosONE versus cLBP. kneeOA is the only
# dataset carrying all three groups, so it is the only clean within-dataset
# test, and it is reported separately rather than buried in the pooled number.
print('\n' + '-' * 70)
print('WITHIN DATASET (pain group is otherwise confounded with dataset)')
print('-' * 70)
print('subjects per dataset x pain group:')
print(subject_c.pivot_table(index='dataset', columns='group_label',
                            values='subject_uid', aggfunc='count',
                            fill_value=0).to_string())
# Run on the same subset as the pooled tests above -- offset trials as well as
# all types -- or the within-dataset numbers are not comparable to them.
for subset_label, table in [('all trial types', subject_c),
                            ('offset only', subject_c_offset)]:
    print(f'\n### {subset_label} ###')
    for ds, g in table.groupby('dataset'):
        if g['group_label'].nunique() >= 2:
            compare_period_c(g, 'group_label', f'{ds}: by pain group')
    for ds, g in table.groupby('dataset'):
        if g['classification'].nunique() >= 2:
            compare_period_c(g, 'classification', f'{ds}: by hold classification')

# Classification is cut from a continuous slope, so test against that too --
# it uses the whole range instead of three bins and is more powerful. Reported
# within dataset as well, for the same confound reason as above.
print('\n' + '-' * 70)
print('AGAINST THE CONTINUOUS HOLD SLOPE')
print('-' * 70)
for label, df in [('all trial types', subject_c), ('offset only', subject_c_offset)]:
    s = df.dropna(subset=['observed_mean_slope', 'nrmse'])
    rho, pval = stats.spearmanr(s['observed_mean_slope'], s['nrmse'])
    print(f'  {label:16s} pooled   n={len(s):3d}  rho = {rho:+.3f}  p = {pval:.4f}'
          f'{"   <-- p < .05" if pval < 0.05 else ""}')
    for ds, g in s.groupby('dataset'):
        if len(g) < 20:
            continue
        rho, pval = stats.spearmanr(g['observed_mean_slope'], g['nrmse'])
        print(f'  {"":16s} {ds:8s} n={len(g):3d}  rho = {rho:+.3f}  p = {pval:.4f}'
              f'{"   <-- p < .05" if pval < 0.05 else ""}')

subject_c.to_csv(RESULTS_PATH + f'{STAMP}_period_c_by_group.csv', index=False)
print(f'\n-> {STAMP}_period_c_by_group.csv')

#%%
# ==================================================================
# Does period-C failure track any of the fitted parameters?
# ==================================================================
# Exploratory. The model misses in period C, worst on offset trials -- so is
# there anything about a subject's own fitted dynamics that predicts how badly
# it misses them? Offset and onset are kept separate throughout, because their
# period-C errors point in opposite directions and a pooled correlation would
# average a real effect against its own mirror image.
#
# Two outcome measures, because they ask different things:
#   nrmse   HOW FAR off the model is, as a fraction of that subject's VAS
#           range (raw RMSE would partly measure rating style -- rho = 0.78
#           against range -- and alpha scales with rating magnitude too, so
#           raw RMSE vs alpha would be confounded on both sides).
#   signed  WHICH WAY it is off. This is the one the offset result lives in.
#
# Spearman rather than Pearson: the parameters are bounded, skewed and in
# alpha's case spread over four orders of magnitude.
#
# CAVEAT on lambda: it sits on its upper bound of 50 for ~30% of subjects, so
# a third of that column is not an estimate, it is the edge of the search
# space. Any lambda correlation here is reported but should not be trusted.
print('\n' + '=' * 70)
print('PERIOD C ERROR vs FITTED PARAMETERS (exploratory)')
print('=' * 70)

REG_PARAMS = [('alpha', r'$\alpha$', True), ('beta', r'$\beta$', False),
              ('gamma', r'$\gamma$', True), ('lam', r'$\lambda$', False),
              ('theta', r'$\theta$', False)]
REG_TYPES = ['offset', 'onset']

reg = (period_c[period_c['trial_type'].isin(REG_TYPES)]
       .groupby(['subject_uid', 'trial_type'])[['nrmse', 'signed']]
       .mean().reset_index()
       .merge(full[[p for p, _, _ in REG_PARAMS]],
              left_on='subject_uid', right_index=True, how='inner'))

results = {}
for tt in REG_TYPES:
    s = reg[reg['trial_type'] == tt]
    for measure in ['nrmse', 'signed']:
        rho = [stats.spearmanr(s[p], s[measure])[0] for p, _, _ in REG_PARAMS]
        raw = [stats.spearmanr(s[p], s[measure])[1] for p, _, _ in REG_PARAMS]
        adj = multipletests(raw, method='holm')[1]
        for (p, _, _), r, a in zip(REG_PARAMS, rho, adj):
            results[(tt, measure, p)] = (r, a)
        print(f'\n  {TRIAL_LABELS[tt]}  period-C {measure}  (n={len(s)}, '
              f'Holm across the {len(REG_PARAMS)} parameters):')
        print('    ' + '   '.join(
            f'{p}: rho={r:+.2f} p={a:.3f}{"*" if a < .05 else " "}'
            for (p, _, _), r, a in zip(REG_PARAMS, rho, adj)))

# Figure on the signed error only: it is the measure the offset result is
# stated in, and a 2x5 grid is already as much as one panel can carry.
fig, axes = plt.subplots(2, len(REG_PARAMS), figsize=(3.0 * len(REG_PARAMS), 6.2),
                         sharey='row')
for r_i, tt in enumerate(REG_TYPES):
    s = reg[reg['trial_type'] == tt]
    for c_i, (p, label, use_log) in enumerate(REG_PARAMS):
        ax = axes[r_i][c_i]
        ax.scatter(s[p], s['signed'], s=11, alpha=.5,
                   color=TRIAL_COLORS[tt], edgecolor='none')
        rho, adj = results[(tt, 'signed', p)]
        # Fitted on ranks, to match the statistic being reported -- a
        # least-squares line through raw values would be pulled by the few
        # subjects at alpha = 500.
        xr, yr = s[p].rank(), s['signed'].rank()
        b, a0 = np.polyfit(xr, yr, 1)
        order = np.argsort(xr.values)
        ax.plot(s[p].values[order],
                np.interp(a0 + b * xr.values[order], yr.sort_values(),
                          s['signed'].sort_values()),
                color='0.2', lw=1.4, zorder=4)
        ax.axhline(0, color='0.6', lw=.9, ls=(0, (4, 3)), zorder=0)
        if use_log:
            ax.set_xscale('log')
        ax.set_title(f'{label}   ρ={rho:+.2f}'
                     f'{"*" if adj < .05 else ""}', fontsize=10)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        if c_i == 0:
            ax.set_ylabel(f'{TRIAL_LABELS[tt]}\nperiod C: rating − model (VAS)',
                          fontsize=8.5)
fig.suptitle('Period C error against each fitted parameter   '
             '(* = Holm-corrected p < .05)', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_PATH}{STAMP}_period_c_vs_parameters.png', dpi=200,
            bbox_inches='tight')
plt.show()

# %%
