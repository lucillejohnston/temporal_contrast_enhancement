# -*- coding: utf-8 -*-
"""
Build 1 Hz pain/temperature traces for the sEEG patients (RCS08, RCS09, PR08)
in the same format as combined_traces_1Hz.pkl, so the model fitting code can
run on them unchanged -- point its DATA_PATH at sEEG_traces_1Hz.pkl.

Written to its own file rather than appended to combined_traces_1Hz.pkl, so
the 310-subject results built on that table do not shift.

Source is each patient's *_all_events.csv, which is long format: one row per
(timestamp, event), with event = 'temperature' or 'pain_rating'. That is
pivoted to one row per timestamp, then each trial is put on a 1 Hz grid with
the same rule build_combined_traces.py uses.

Differences from the alter_collab datasets, and how they are handled:
  - Trial labels differ. step_down rises 1 C to T2 then returns to T1, which
    is 'offset'; step_up dips 1 C then returns, which is 'onset' (plosONE's
    'inv'). Checked by overlaying each type's temperature trace on the
    alter_collab type means.
  - There is no aligned_time. Time zero is the first sample of each trial.
  - RCS08 was run differently: samples are irregular (gaps up to ~1 s), its
    plateaus are hotter (temperature was calibrated per subject), and its
    slider was not reset between trials, so many trials start at VAS 50-90.
    Its timestamps also read 2025-12-11 against a 20250606 filename; only
    within-trial differences are used here, so that does not matter.
  - Baseline is ~29 C rather than 32-33 C. The model fits theta, so this
    needs no correction.

Author: Lucille Johnston
Updated: 11/3/26
"""
#%%
import glob
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

BASE_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/')
OUT_PATH = BASE_PATH + 'alter_collab_data/sEEG_traces_1Hz.pkl'

# subject numbers continue the SUBJECT_OFFSETS bands in build_combined_traces.py
# (plosONE 10k, kneeOA 20k, cLBP 30k) so the two tables can be concatenated.
# The sequential IDs match the manifest in extract_sEEG_data.py.
DATASET = 'sEEG'
SUBJECT_OFFSET = 40_000
PATIENTS = [
    {'pt': 'RCS08', 'subject': 1, 'study': 'sEEG_RCS'},
    {'pt': 'RCS09', 'subject': 2, 'study': 'sEEG_RCS'},
    {'pt': 'PR08',  'subject': 3, 'study': 'sEEG_Presidio'},
]

# Psychophysical_Modeling.py fits non-forearm sites as separate units, so the
# site label matters. All TCE runs here are assumed to be on the forearm.
SITE = 'forearm'

# Case differs between patients (hold_T1 in RCS08, hold_t1 elsewhere).
TRIAL_TYPE_MAP = {
    'step_down': 'offset',
    'step_up': 'onset',
    'hold_t1': 't1_hold',
    'hold_t2': 't2_hold',
}

# Same settings as build_combined_traces.py / preprocessing.py.
BASELINE_START = -5.0
CUTOFF_TIME = 60.0
TARGET_SAMPLE_RATE = 1.0

# Trials whose peak rating stays below this are flagged, not dropped -- a
# patient who never moved the slider is still data, but it pins theta above
# the stimulus for that trial.
LOW_RATING_FLAG = 10


#%%
# Copied from build_combined_traces.py rather than imported: that module runs
# its whole build at import time.
def resample_trial(trial_df):
    """
    Trim one trial to [BASELINE_START, CUTOFF_TIME] and resample onto a 1 Hz
    grid. Returns None if the trial has too few usable points.
    """
    clean = trial_df.dropna(subset=['aligned_time', 'temperature', 'pain'])
    clean = clean.sort_values('aligned_time')
    if len(clean) < 2:
        return None

    clean = clean[(clean['aligned_time'] >= BASELINE_START) &
                  (clean['aligned_time'] <= CUTOFF_TIME)]
    if len(clean) < 2:
        return None

    # interp1d needs strictly increasing x; duplicate timestamps do occur.
    clean = clean[~clean['aligned_time'].duplicated(keep='first')]
    if len(clean) < 2:
        return None

    t = clean['aligned_time'].values
    new_time = np.arange(t.min(), t.max(), 1.0 / TARGET_SAMPLE_RATE)
    if len(new_time) < 2:
        return None

    def interp(col):
        return interp1d(t, clean[col].values, kind='linear',
                        bounds_error=False, fill_value='extrapolate')(new_time)

    return pd.DataFrame({
        'aligned_time': new_time,
        'temperature': interp('temperature'),
        'pain': interp('pain'),
    })


#%%
# ------------------------------------------------------------------
# 1. Load, pivot to wide, and resample each patient
# ------------------------------------------------------------------
print('=== BUILDING sEEG 1 Hz TRACES ===')
per_patient = []
qc_rows = []

for info in PATIENTS:
    pt = info['pt']
    files = glob.glob(f'{BASE_PATH}{pt}_data/{pt}_*_all_events.csv')
    assert len(files) == 1, f'{pt}: expected one _all_events.csv, found {files}'
    print(f'\n--- {pt} ({os.path.basename(files[0])}) ---')

    long_df = pd.read_csv(files[0])
    long_df['timestamp'] = pd.to_datetime(long_df['timestamp'])
    long_df['trial_count'] = long_df['trial_count'].astype(int)  # float in RCS09

    # pivot_table averages any duplicated (timestamp, event) pairs.
    wide = (long_df.pivot_table(index=['trial_count', 'trial_type', 'timestamp'],
                                columns='event', values='value')
            .reset_index()
            .rename(columns={'pain_rating': 'pain'}))
    # The slider holds its last value, so a timestamp with temperature but no
    # rating takes the previous rating -- within the trial only.
    wide = wide.sort_values(['trial_count', 'timestamp'])
    wide['pain'] = wide.groupby('trial_count')['pain'].ffill()

    resampled = []
    for (trial_num, raw_type), trial_df in wide.groupby(['trial_count', 'trial_type']):
        trial_df = trial_df.copy()
        trial_df['aligned_time'] = (trial_df['timestamp']
                                    - trial_df['timestamp'].min()).dt.total_seconds()
        out = resample_trial(trial_df)
        if out is None:
            print(f'  trial {trial_num}: too few points, dropped')
            continue
        out['trial_num'] = trial_num
        out['trial_type'] = TRIAL_TYPE_MAP[raw_type.lower()]
        resampled.append(out)

        gaps = np.diff(np.sort(trial_df['aligned_time'].unique()))
        qc_rows.append({'pt': pt, 'trial_num': trial_num,
                        'trial_type': out['trial_type'].iloc[0],
                        'raw_rate_hz': 1 / np.median(gaps),
                        'max_gap_s': gaps.max(),
                        'pain_start': out['pain'].iloc[0],
                        'pain_max': out['pain'].max()})

    pt_df = pd.concat(resampled, ignore_index=True)
    pt_df['study'] = info['study']
    pt_df['subject_orig'] = info['subject']
    pt_df['subject'] = SUBJECT_OFFSET + info['subject']
    pt_df['subject_uid'] = f'{DATASET}_{pt}'
    per_patient.append(pt_df)
    print(f'  {wide["trial_count"].nunique()} trials in -> '
          f'{pt_df["trial_num"].nunique()} out, {len(pt_df):,} rows')

traces = pd.concat(per_patient, ignore_index=True)
qc = pd.DataFrame(qc_rows)


#%%
# ------------------------------------------------------------------
# 2. Remaining columns, matching combined_traces_1Hz.pkl
# ------------------------------------------------------------------
traces['dataset'] = DATASET
traces['site'] = SITE
traces['group_label'] = 'sEEG'
traces['threshold_limits'] = np.nan  # no method-of-limits data on file
# Single session per patient.
traces['visit'] = 0
traces['trial_in_visit'] = traces['trial_num']

assert traces.groupby(['subject', 'trial_num']).ngroups == len(qc), \
    'subject/trial key is not unique'

COLUMNS = ['dataset', 'study', 'site', 'subject', 'subject_uid', 'subject_orig',
           'group_label', 'threshold_limits',
           'trial_num', 'visit', 'trial_in_visit', 'trial_type',
           'aligned_time', 'temperature', 'pain']
traces = traces[COLUMNS].sort_values(['subject', 'trial_num', 'aligned_time'])
traces = traces.reset_index(drop=True)


#%%
# ------------------------------------------------------------------
# 3. QC summary
# ------------------------------------------------------------------
print('\n=== QC ===')
print(qc.groupby('pt').agg(trials=('trial_num', 'size'),
                           median_rate_hz=('raw_rate_hz', 'median'),
                           worst_gap_s=('max_gap_s', 'max'),
                           start_above_5=('pain_start', lambda s: (s > 5).sum()))
      .round(2).to_string())

low = qc[qc['pain_max'] < LOW_RATING_FLAG]
if len(low):
    print(f'\nTrials with peak rating below VAS {LOW_RATING_FLAG} (kept):')
    print(low[['pt', 'trial_num', 'trial_type', 'pain_max']].round(1).to_string(index=False))

print('\n=== SUMMARY ===')
print(f'rows      {len(traces):,}')
print(f'subjects  {traces["subject"].nunique()}')
print(f'trials    {traces.groupby(["subject", "trial_num"]).ngroups}')
print(traces.drop_duplicates(['subject_uid', 'trial_num'])
      .groupby(['subject_uid', 'trial_type']).size().unstack().to_string())

traces.to_pickle(OUT_PATH)
print(f'\nSaved -> {OUT_PATH}')

# %%
