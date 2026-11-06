# -*- coding: utf-8 -*-
"""
Where do the three sEEG patients sit in the cohort's full-model distributions?

Every fit is per subject (fit_full_model.py) or per trial warm-started from that
subject's own fit (per_trial_fits.py), so adding the patients to the cohort
doesn't change anyone else's parameters. Overlaying the two result sets is the
same as refitting everyone together.

Reference group is forearm fits only, since the patients were tested on the
forearm. Percentile = % of cohort fits at or below the patient's value.

Author: Lucille Johnston
Updated: 11/4/26
"""
#%%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, 'model_fit_results') + '/'
FIG_PATH = os.path.join(HERE, 'figures', 'sEEG') + '/'
SITE = 'forearm'

def newest(pattern):
    return sorted(glob.glob(pattern))[-1]

cohort_subj = pd.read_csv(newest(RESULTS_PATH + '*model_fits_subject_full.csv'))
cohort_trial = pd.read_csv(newest(RESULTS_PATH + '*_per_trial_fits_full_fixedtheta.csv'))
seeg_subj = pd.read_csv(newest(RESULTS_PATH + 'sEEG/*model_fits_subject_full.csv'))
seeg_trial = pd.read_csv(newest(RESULTS_PATH + 'sEEG/*_per_trial_fits_full_fixedtheta.csv'))

cohort_subj = cohort_subj[cohort_subj['site'] == SITE]
cohort_trial = cohort_trial[cohort_trial['site'] == SITE]

PATIENT_COLORS = {'sEEG_PR08': '#2a78d6', 'sEEG_RCS08': '#eb6834', 'sEEG_RCS09': '#1baf7a'}
COHORT_COLOR = '#9a9993'


def add_derived(df):
    df = df.copy()
    df['alpha_over_beta'] = df['alpha'] / df['beta']
    df['gamma_lam_over_beta'] = df['gamma'] * df['lam'] / df['beta']
    df['rmse'] = np.sqrt(df['mse'])
    return df

cohort_subj, seeg_subj = add_derived(cohort_subj), add_derived(seeg_subj)


def per_subject_trial_summary(trials):
    """Median per-trial fit quality and share of trials with a parameter on a bound."""
    g = trials.groupby('subject_uid')
    return pd.DataFrame({
        'trial_r_median': g['r'].median(),
        'trial_rmse_median': g['rmse'].median(),
        'frac_trials_at_bound': g['at_bounds'].apply(lambda s: s.notna().mean()),
    })

cohort_subj = cohort_subj.merge(per_subject_trial_summary(cohort_trial),
                                left_on='subject_uid', right_index=True, how='left')
seeg_subj = seeg_subj.merge(per_subject_trial_summary(seeg_trial),
                            left_on='subject_uid', right_index=True, how='left')

# (column, label, log axis)
PANELS = [
    ('alpha', 'α', True),
    ('beta', 'β', True),
    ('gamma', 'γ', True),
    ('lam', 'λ', True),
    ('theta', 'θ (°C)', False),
    ('alpha_over_beta', 'α/β  (steady-state gain)', True),
    ('gamma_lam_over_beta', 'γλ/β  (rate-term weight)', True),
    ('r', 'per-subject fit r', False),
    ('rmse', 'per-subject fit RMSE (VAS)', False),
    ('trial_r_median', 'median per-trial r', False),
    ('trial_rmse_median', 'median per-trial RMSE (VAS)', False),
    ('frac_trials_at_bound', 'share of trials with a\nparameter on a bound', False),
]


#%%
# ------------------------------------------------------------------
# Percentile of each patient within the cohort
# ------------------------------------------------------------------
rows = []
for _, p in seeg_subj.iterrows():
    for col, label, _ in PANELS:
        ref = cohort_subj[col].dropna()
        rows.append({'subject_uid': p['subject_uid'], 'measure': col,
                     'value': p[col], 'cohort_median': ref.median(),
                     'cohort_percentile': 100 * (ref <= p[col]).mean()})
pct = pd.DataFrame(rows)
pct_wide = pct.pivot(index='measure', columns='subject_uid', values='cohort_percentile')
pct_wide = pct_wide.loc[[c for c, _, _ in PANELS]].round(0)
print(f'Cohort reference: {len(cohort_subj)} {SITE} fits\n')
print('Percentile within cohort:')
print(pct_wide.to_string())

stamp = pd.Timestamp.now().strftime('%Y%m%d')
pct.to_csv(f'{RESULTS_PATH}sEEG/{stamp}_sEEG_cohort_percentiles.csv', index=False)


#%%
# ------------------------------------------------------------------
# Violins of the two ratios: cohort as gray violin + jittered dots, patients on top.
# Both span orders of magnitude, so the violin is drawn on log10 values (a KDE on
# a log axis would be distorted) and the ticks are relabelled as powers of ten.
# ------------------------------------------------------------------
from matplotlib.ticker import FuncFormatter, MultipleLocator

VIOLIN_PANELS = [('alpha_over_beta', 'α/β  (steady-state gain)'),
                 ('gamma_lam_over_beta', 'γλ/β  (rate-term weight)')]

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(8, 5))
for ax, (col, label) in zip(axes, VIOLIN_PANELS):
    ref = np.log10(cohort_subj[col][cohort_subj[col] > 0])
    parts = ax.violinplot(ref, positions=[0], widths=0.8, showextrema=False,
                          showmedians=True)
    for body in parts['bodies']:
        body.set_facecolor('#e6e5e0')
        body.set_edgecolor('#9a9993')
        body.set_alpha(1)
    parts['cmedians'].set_color('#52514e')
    parts['cmedians'].set_linewidth(2)
    ax.scatter(rng.uniform(-0.15, 0.15, len(ref)), ref, s=10,
               color=COHORT_COLOR, alpha=0.45, linewidths=0, zorder=2)
    for i, (_, p) in enumerate(seeg_subj.iterrows()):
        ax.scatter(-0.08 + 0.08 * i, np.log10(p[col]), s=80,
                   color=PATIENT_COLORS[p['subject_uid']],
                   edgecolor='white', linewidth=1.5, zorder=4)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'$10^{{{v:.0f}}}$'))
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_title(label, fontsize=10)
    ax.grid(axis='y', color='#e6e5e0', linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

handles = [plt.Line2D([], [], marker='o', ls='', color=c, markersize=8,
                      label=uid.replace('sEEG_', '')) for uid, c in PATIENT_COLORS.items()]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0))
fig.suptitle('Full-model fits: sEEG patients against the cohort', y=1.05, fontsize=12)
fig.tight_layout()
out = f'{FIG_PATH}{stamp}_sEEG_vs_cohort_parameters.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved {out}')


#%%
# ------------------------------------------------------------------
# Per-trial fit quality by trial type
# ------------------------------------------------------------------
TYPES = ['onset', 'offset', 't1_hold', 't2_hold']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (col, label) in zip(axes, [('r', 'per-trial r'), ('rmse', 'per-trial RMSE (VAS)')]):
    for j, tt in enumerate(TYPES):
        subj_med = (cohort_trial[cohort_trial['trial_type'] == tt]
                    .groupby('subject_uid')[col].median().dropna())
        ax.boxplot(subj_med, positions=[j], widths=0.45, showfliers=False,
                   medianprops={'color': '#52514e', 'linewidth': 2},
                   boxprops={'color': '#52514e'}, whiskerprops={'color': '#52514e'},
                   capprops={'color': '#52514e'})
        ax.scatter(j + rng.uniform(-0.18, 0.18, len(subj_med)), subj_med, s=8,
                   color=COHORT_COLOR, alpha=0.4, linewidths=0)
        for i, (uid, c) in enumerate(PATIENT_COLORS.items()):
            v = seeg_trial.query('subject_uid == @uid and trial_type == @tt')[col].median()
            ax.scatter(j - 0.08 + 0.08 * i, v, s=70, color=c, edgecolor='white',
                       linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels(TYPES)
    ax.set_xlim(-0.5, len(TYPES) - 0.5)
    ax.set_title(f'{label}: subject medians by trial type', fontsize=10)
    ax.grid(axis='y', color='#e6e5e0', linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
out = f'{FIG_PATH}{stamp}_sEEG_vs_cohort_trial_fit_by_type.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

# %%
