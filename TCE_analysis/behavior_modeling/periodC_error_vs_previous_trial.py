# -*- coding: utf-8 -*-
"""
Does the per-trial model's period-C miss depend on the PREVIOUS trial?

The behavioural result (combining_datasets.py, section 2) is that OA on trial n
tracks the normalized pain change on trial n-1. If the model's period-C error
on trial n also tracks trial n-1, the model is missing information carried
from one trial to the next -- the room to improve it.

x: |abs_normalized_pain_change| of the previous trial (trial_num - 1, same
   subject, any trial type), from *_trial_metrics.json. "abs" in that column
   name means absolute time, not absolute value; the absolute value is taken
   here.
y: period-C signed error (rating - model) of the current trial, from
   *_per_trial_fits_per_period.csv. < 0 = model above the ratings. Drawn once
   for each fit:
     signed_trial   -- per-trial fit. It can re-tune its parameters on every
                       trial, so a history effect may be absorbed and never
                       show up as error.
     signed_subject -- one parameter set per subject. It cannot adapt from
                       trial to trial, so this is the cleaner test of whether
                       the model is missing something carried over.

Two correlations per panel:
  trial-level     : all trials as they are
  within-subject  : x and y each centred on the subject's own mean first, so
                    stable differences between people cannot produce it

Author: Lucille Johnston
Updated: 11/05/26
"""
#%%
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
RESULTS_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
                'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/'
                'model_fit_results/')
FIG_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
            'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/')
DATASETS = ['plosONE', 'kneeOA', 'cLBP']
STEPPED = ['offset', 'onset']
TRIAL_LABELS = {'offset': 'Offset (OA)', 'onset': 'Onset (OH)'}
TRIAL_COLORS = {'offset': '#026A07', 'onset': '#DA1C1C'}   # per_period_fits.py
NPC = 'abs_normalized_pain_change'
STAMP = f'{datetime.now():%Y%m%d}'

#%%
# ------------------------------------------------------------------
# Load: period-C error per trial, normalized pain change per trial
# ------------------------------------------------------------------
per_csv = sorted(glob.glob(RESULTS_PATH + '*_per_trial_fits_per_period.csv'))[-1]
per = pd.read_csv(per_csv)
c = per[(per['period'] == 'C') & per['trial_type'].isin(STEPPED)
        & per['signed_trial'].notna() & per['signed_subject'].notna()]
FITS = {'signed_trial': ('per-trial fit', 'per_trial'),
        'signed_subject': ('one set per subject', 'per_subject')}

recs = []
for ds in DATASETS:
    with open(f'{DATA_PATH}{ds}_trial_metrics.json') as f:
        m = json.load(f)
    # keys are canonical subject numbers (cLBP on DPOP_IDs), the same number
    # build_combined_traces.py zero-pads into subject_uid
    for subj, trials in m.items():
        for tn, t in trials.items():
            recs.append({'subject_uid': f'{ds}_{int(subj):04d}',
                         'trial_num': int(tn), NPC: t.get(NPC)})
met = pd.DataFrame(recs)

# previous trial = trial_num - 1, as in combining_datasets.py. cLBP repeat
# visits are trial_num + 100 * visit, so - 1 never crosses a visit.
prev = met.assign(trial_num=met['trial_num'] + 1).rename(
    columns={NPC: 'prev_npc'})
df = c.merge(prev, on=['subject_uid', 'trial_num'], how='inner')
df = df[df['prev_npc'].notna()].copy()
df['prev_abs'] = df['prev_npc'].abs()
print(f'{os.path.basename(per_csv)}')
for tt in STEPPED:
    print(f'  {tt}: {(df["trial_type"] == tt).sum()} of '
          f'{(c["trial_type"] == tt).sum()} period-C trials have a previous trial')

#%%
# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
for Y, (fit_label, fit_tag) in FITS.items():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, tt in zip(axes, STEPPED):
        s = df[df['trial_type'] == tt].copy()
        ax.scatter(s['prev_abs'], s[Y], s=9, alpha=.45,
                   color=TRIAL_COLORS[tt], linewidths=0)
        ax.axhline(0, color='0.3', lw=1.1, ls=(0, (4, 3)), zorder=0)

        rho, p = st.spearmanr(s['prev_abs'], s[Y])
        for col in ('prev_abs', Y):
            s[col + '_w'] = s[col] - s.groupby('subject_uid')[col].transform('mean')
        rho_w, p_w = st.spearmanr(s['prev_abs_w'], s[Y + '_w'])
        ax.text(.98, .03,
                f'trial-level ρ = {rho:+.2f}, p = {p:.2g}\n'
                f'within-subject ρ = {rho_w:+.2f}, p = {p_w:.2g}\n'
                f'{len(s)} trials, {s["subject_uid"].nunique()} subjects',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5)
        print(f'{fit_label:20s} {tt}: trial-level rho={rho:+.3f} p={p:.2g} | '
              f'within-subject rho={rho_w:+.3f} p={p_w:.2g}')

        ax.set_title(TRIAL_LABELS[tt], fontsize=11)
        ax.set_xlabel('previous trial: |normalized pain change|')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('this trial, period C: rating − model (VAS)')
    fig.suptitle(f'Model error in period C vs. the previous trial ({fit_label})',
                 fontsize=11)
    plt.tight_layout()
    out = FIG_PATH + f'{STAMP}_periodC_error_vs_previous_trial_{fit_tag}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'-> {out}')
    plt.show()
