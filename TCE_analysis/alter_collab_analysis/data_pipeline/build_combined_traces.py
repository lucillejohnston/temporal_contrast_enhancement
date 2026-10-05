# -*- coding: utf-8 -*-
"""
Build one analysis-ready table of 1 Hz pain/temperature traces across all three
datasets (plosONE, kneeOA, cLBP), with globally unique subject IDs, trial types
and group labels attached.

Rebuilds the 1 Hz traces from *_trial_data_cleaned_aligned.json rather than
reading the existing *_trial_data_trimmed_downsampled.json files, for two
reasons:

  1. The cLBP 1 Hz file is wrong. cLBP_trial_data_cleaned_aligned.json holds two
     substudies (cLBP_DPOP, cLBP_MBPR) whose subject IDs both start at 1, and 7
     IDs (1, 10, 22, 40, 43, 48, 53) belong to a different person in each. The
     downsampling loop in preprocessing.py groups on (subject, trial_num) with
     no 'study', so for those IDs it pulls both people's rows into one trial and
     interpolates them together. 1238 real trials collapse to 1154.
  2. preprocessing.py now guards that block with `if dataset != 'cLBP'`, so the
     cLBP file on disk is also stale relative to its own source.

Keying on (study, subject, trial_num) throughout fixes both.

Author: Lucille Johnston
Updated: 9/19/26
"""
#%%
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

DATA_PATH = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
             'temporal_contrast_enhancement/data/alter_collab_data/')
SQL_PATH = DATA_PATH + 'combined_data.sqlite'
OUT_STEM = DATA_PATH + 'combined_traces_1Hz'

DATASETS = ['plosONE', 'kneeOA', 'cLBP']

# kneeOA stimulated two sites, forearm and knee, and every subject has both
# (1352 and 1268 trials across all 102 subjects). cleaned_aligned.json carries
# a 'site' column; the 1 Hz file written by preprocessing.py drops it, which is
# how both sites ended up pooled in earlier fits. extract_metrics.py filters to
# forearm, so anything built on the metrics is forearm-only.
#
# Both sites are kept here and 'site' is carried through, so a fit can be run
# per site rather than straddling two body regions -- for a knee OA patient the
# knee is the affected joint, so the two are worth comparing rather than
# pooling. Set SITE_FILTER to 'forearm' to drop the knee trials at build time
# instead. Datasets without a 'site' column are unaffected either way.
SITE_FILTER = None
DEFAULT_SITE = 'forearm'   # used for datasets that have no site column

# Resampling settings -- these mirror preprocessing.py so the traces stay
# comparable to everything already built on the old 1 Hz files.
BASELINE_START = -5.0   # seconds before stimulus onset to keep
CUTOFF_TIME = 60.0      # seconds after onset to keep
TARGET_SAMPLE_RATE = 1.0  # Hz

# Offsets that make subject IDs unique across datasets. Bands are wide enough
# that cLBP's DPOP_ID namespace (which runs up to 2145) cannot collide.
SUBJECT_OFFSETS = {'plosONE': 10_000, 'kneeOA': 20_000, 'cLBP': 30_000}


#%%
def canonical_clbp_id(subject, study):
    """
    Map a within-substudy cLBP subject number onto the lab's DPOP_ID namespace.

    MYP_DPOP_MBPR_DATA_09062023.csv uses DPOP_ID as the master ID for both
    substudies. DPOP subjects keep their own number (1-57); MBPR subjects are
    re-coded by prefixing a '2' onto a zero-padded MBPR ID:

        MBPR 1 -> 201      MBPR 53  -> 253
        MBPR 10 -> 210     MBPR 101 -> 2101
        MBPR 22 -> 222     MBPR 145 -> 2145

    This holds for every MBPR row in that file, and it is what makes DPOP
    subject 1 and MBPR subject 1 distinguishable -- they are DPOP_ID 1 and 201.
    """
    if study == 'cLBP_DPOP':
        return int(subject)
    if study == 'cLBP_MBPR':
        return int('2' + str(int(subject)).zfill(2))
    raise ValueError(f'Unexpected cLBP study: {study!r}')


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
# 1. Rebuild the 1 Hz traces, one dataset at a time
# ------------------------------------------------------------------
print('=== REBUILDING 1 Hz TRACES ===')
per_dataset = []

for dataset in DATASETS:
    src = f'{DATA_PATH}{dataset}_trial_data_cleaned_aligned.json'
    print(f'\n--- {dataset} ---')
    print(f'  reading {os.path.basename(src)} '
          f'({os.path.getsize(src) / 1e6:.0f} MB)...')
    aligned = pd.read_json(src, orient='records')

    # kneeOA stimulated two sites, forearm and knee, and cleaned_aligned holds
    # both (1352 and 1268 trials, all 102 subjects). extract_metrics.py filters
    # to forearm, so everything built on the metrics already assumes forearm
    # only; without the same filter here a subject's fit would straddle two
    # body sites, one of which is their affected joint.
    if 'site' not in aligned.columns:
        aligned['site'] = DEFAULT_SITE
    elif SITE_FILTER is not None:
        before = aligned.groupby(['subject', 'trial_num']).ngroups
        aligned = aligned[aligned['site'] == SITE_FILTER].copy()
        after = aligned.groupby(['subject', 'trial_num']).ngroups
        print(f'  site filter ({SITE_FILTER}): {before} -> {after} trials')
    else:
        print(f'  sites kept: '
              + ', '.join(f'{s} ({n} trials)' for s, n in
                          aligned.groupby('site')
                          .apply(lambda g: g.groupby(["subject", "trial_num"]).ngroups,
                                 include_groups=False).items()))

    # plosONE/kneeOA carry a single study; make the column explicit either way
    # so the grouping key below is uniform across datasets.
    if 'study' not in aligned.columns:
        aligned['study'] = dataset

    n_trials = aligned.groupby(['study', 'site', 'subject', 'trial_num']).ngroups
    n_naive = aligned.groupby(['subject', 'trial_num']).ngroups
    print(f'  {len(aligned):,} rows, {n_trials:,} trials, '
          f'{aligned.groupby(["study", "subject"]).ngroups} subjects')
    if n_naive != n_trials:
        print(f'  NOTE: keying without "study" would lose '
              f'{n_trials - n_naive} trials -- this is the cLBP collision.')

    resampled = []
    # Grouping on study as well as subject is the whole point: it keeps
    # DPOP subject 1 and MBPR subject 1 apart.
    for (study, site, subject, trial_num), trial_df in aligned.groupby(
            ['study', 'site', 'subject', 'trial_num'], sort=True):
        out = resample_trial(trial_df)
        if out is None:
            continue
        out['dataset'] = dataset
        out['study'] = study
        out['site'] = site
        out['subject_orig'] = int(subject)
        out['trial_num'] = int(trial_num)
        out['trial_type'] = trial_df['trial_type'].iloc[0] if 'trial_type' in trial_df else np.nan
        resampled.append(out)

    ds_df = pd.concat(resampled, ignore_index=True)
    kept = ds_df.groupby(['study', 'site', 'subject_orig', 'trial_num']).ngroups
    print(f'  -> {len(ds_df):,} rows, {kept:,} trials kept '
          f'({n_trials - kept} dropped as too short)')
    per_dataset.append(ds_df)

    del aligned, resampled

traces = pd.concat(per_dataset, ignore_index=True)
del per_dataset
print(f'\nCombined: {len(traces):,} rows')


#%%
# ------------------------------------------------------------------
# 1b. Clean up trial types
# ------------------------------------------------------------------
# plosONE contains an audiovisual conditioning experiment, in which the
# thermal stimulus was paired with an audiovisual cue to associatively
# condition a placebo-like effect. That is a different experiment, so its
# trials do not belong in a psychophysical model fit.
#
# Those trials are over half of plosONE, and 59 of its 137 subjects have
# nothing else -- no offset, inv, stepdown, t1_hold or t2_hold at all. Those
# subjects therefore drop out entirely, leaving 78. That is exactly the set in
# plosONE_trial_metrics.json, so it is the sample the earlier pipeline used too.
DROP_TRIAL_TYPES = ['offset_AV_conditioning', 'offset_AV_test']

# Calibration trials are usable as ordinary T1 holds when they were delivered
# at the subject's own T1 temperature, and are otherwise a different stimulus.
# In practice most were run cooler -- a median of ~2C below T1 -- so only a
# minority convert.
CALIBRATION_TRIAL_TYPE = 'calibration'
CALIBRATION_MATCH_TOL = 0.5   # degrees C
CALIBRATION_TARGET = 't1_hold'

# The same stimulus carries different labels across datasets. cLBP's
# extraction maps the raw label 'Inv' to 'onset' (extract_cLBP_data.py), while
# plosONE keeps it as 'inv'. Without this, plosONE contributes nothing to any
# onset-vs-offset comparison.
RENAME_TRIAL_TYPES = {'inv': 'onset'}

print('\n=== CLEANING TRIAL TYPES ===')
before = traces.groupby(['dataset', 'subject_orig', 'study', 'trial_num']).ngroups

for old, new in RENAME_TRIAL_TYPES.items():
    n = traces.loc[traces['trial_type'] == old].groupby(
        ['study', 'subject_orig', 'trial_num']).ngroups
    if n:
        print(f'  relabelling {old!r} -> {new!r}: {n} trials')
        traces.loc[traces['trial_type'] == old, 'trial_type'] = new

dropped = traces['trial_type'].isin(DROP_TRIAL_TYPES)
print(f'  dropping {sorted(DROP_TRIAL_TYPES)}: '
      f'{traces.loc[dropped].groupby(["study", "subject_orig", "trial_num"]).ngroups} trials')
traces = traces[~dropped]

# Each trial's peak temperature, and each subject's T1 reference taken from
# their genuine t1_hold trials.
peaks = (traces.groupby(['study', 'subject_orig', 'trial_num'])
         .agg(peak=('temperature', 'max'),
              trial_type=('trial_type', 'first')).reset_index())
t1_ref = (peaks[peaks['trial_type'] == CALIBRATION_TARGET]
          .groupby(['study', 'subject_orig'])['peak'].median().rename('T1'))

cal = peaks[peaks['trial_type'] == CALIBRATION_TRIAL_TYPE].join(
    t1_ref, on=['study', 'subject_orig'])
cal['matches_T1'] = (cal['peak'] - cal['T1']).abs() <= CALIBRATION_MATCH_TOL

n_no_ref = cal['T1'].isna().sum()
keep_keys = set(map(tuple, cal.loc[cal['matches_T1'].fillna(False),
                                   ['study', 'subject_orig', 'trial_num']].values))
print(f'  {CALIBRATION_TRIAL_TYPE}: {len(cal)} trials — '
      f'{len(keep_keys)} within {CALIBRATION_MATCH_TOL}C of T1 '
      f'(relabelled {CALIBRATION_TARGET}), {len(cal) - len(keep_keys)} dropped '
      f'(of which {n_no_ref} had no t1_hold to compare against)')

is_cal = traces['trial_type'] == CALIBRATION_TRIAL_TYPE
key = list(zip(traces['study'], traces['subject_orig'], traces['trial_num']))
matches = pd.Series([k in keep_keys for k in key], index=traces.index)
traces.loc[is_cal & matches, 'trial_type'] = CALIBRATION_TARGET
traces = traces[~(is_cal & ~matches)]

after = traces.groupby(['dataset', 'subject_orig', 'study', 'trial_num']).ngroups
print(f'  trials: {before:,} -> {after:,}')
print(f'  subjects per dataset now:')
print(traces.groupby('dataset')
      .agg(subjects=('subject_orig', lambda s: len(set(zip(
          traces.loc[s.index, 'study'], s)))),
           trials=('trial_num', 'size')).to_string())


#%%
# ------------------------------------------------------------------
# 2. Globally unique subject IDs
# ------------------------------------------------------------------
print('\n=== ASSIGNING UNIQUE SUBJECT IDS ===')

# cLBP first: collapse the two substudies onto the shared DPOP_ID namespace.
is_clbp = traces['dataset'] == 'cLBP'
traces['subject_canonical'] = traces['subject_orig']
traces.loc[is_clbp, 'subject_canonical'] = [
    canonical_clbp_id(s, st)
    for s, st in zip(traces.loc[is_clbp, 'subject_orig'],
                     traces.loc[is_clbp, 'study'])
]

traces['subject'] = (traces['subject_canonical']
                     + traces['dataset'].map(SUBJECT_OFFSETS))
traces['subject_uid'] = (traces['dataset'] + '_'
                         + traces['subject_canonical'].astype(int).astype(str).str.zfill(4))

# cLBP encodes repeat visits as trial_num + 100 * visit.
traces['visit'] = traces['trial_num'] // 100
traces['trial_in_visit'] = traces['trial_num'] % 100

for dataset in DATASETS:
    sub = traces[traces['dataset'] == dataset]
    print(f'  {dataset:8s} {sub["subject"].nunique():3d} subjects, '
          f'{sub.groupby(["subject", "trial_num"]).ngroups:5d} trials, '
          f'ids {sub["subject"].min()}-{sub["subject"].max()}')

assert traces.groupby(['subject', 'trial_num']).ngroups == \
       traces.groupby(['study', 'subject_orig', 'trial_num']).ngroups, \
       'subject/trial key is not unique -- IDs are still colliding'
print('  subject/trial key verified unique')


#%%
# ------------------------------------------------------------------
# 3. Group labels from the SQL metadata
# ------------------------------------------------------------------
print('\n=== ATTACHING GROUP LABELS ===')
conn = sqlite3.connect(SQL_PATH)
groups = pd.read_sql_query(
    'SELECT DISTINCT study, subject AS subject_orig, "group" AS group_label '
    'FROM metadata', conn)

# plosONE has no pain group -- every subject is a healthy control.
groups.loc[groups['study'] == 'plosONE', 'group_label'] = 'Control'
groups['group_label'] = (groups['group_label']
                         .replace('', np.nan)
                         .str.title())   # 'control'/'Control' -> 'Control'

# One row per (study, subject); verified unique, so this merge cannot fan out.
assert not groups.duplicated(['study', 'subject_orig']).any(), \
    'metadata has conflicting group labels for a (study, subject)'

traces = traces.merge(groups, on=['study', 'subject_orig'], how='left')

missing = traces[traces['group_label'].isna()]
if len(missing):
    # Left as NaN rather than imputed. combining_datasets.py defaults blank
    # cLBP groups to 'High', which invents data for subjects that simply have
    # no lowbackpainint score on file.
    print(f'  {missing["subject_uid"].nunique()} subjects have no group label '
          f'(left as NaN): {sorted(missing["subject_uid"].unique())}')

print(traces.groupby(['dataset', 'group_label'], dropna=False)
      .agg(n_subjects=('subject', 'nunique'))
      .to_string())


#%%
# ------------------------------------------------------------------
# 4. Pain thresholds from the method of limits (optional)
# ------------------------------------------------------------------
# threshold_data currently holds plosONE only -- kneeOA and cLBP subjects get
# NaN. Drop rows into that table for the other studies and they will flow
# through here with no code change.
print('\n=== ATTACHING METHOD-OF-LIMITS THRESHOLDS ===')
thresholds = pd.read_sql_query(
    'SELECT study, subject AS subject_orig, limits1, limits2, limits3 '
    'FROM threshold_data', conn)
conn.close()

thresholds['threshold_limits'] = thresholds[['limits1', 'limits2', 'limits3']].mean(axis=1)
traces = traces.merge(thresholds[['study', 'subject_orig', 'threshold_limits']],
                      on=['study', 'subject_orig'], how='left')

have = traces.dropna(subset=['threshold_limits'])
print(f'  limits available for {have["subject_uid"].nunique()} subjects '
      f'in {sorted(have["dataset"].unique())}')
for dataset in DATASETS:
    if dataset not in set(have['dataset']):
        print(f'  {dataset}: no limits data on file -- use '
              f'extract_threshold_from_data() instead')


#%%
# ------------------------------------------------------------------
# 5. Tidy and save
# ------------------------------------------------------------------
COLUMNS = ['dataset', 'study', 'site', 'subject', 'subject_uid', 'subject_orig',
           'group_label', 'threshold_limits',
           'trial_num', 'visit', 'trial_in_visit', 'trial_type',
           'aligned_time', 'temperature', 'pain']
traces = traces[COLUMNS].sort_values(['subject', 'trial_num', 'aligned_time'])
traces = traces.reset_index(drop=True)

print('\n=== SUMMARY ===')
print(f'rows      {len(traces):,}')
print(f'subjects  {traces["subject"].nunique()}')
print(f'trials    {traces.groupby(["subject", "trial_num"]).ngroups:,}')
print(f'\ntrial types:\n{traces.groupby(["dataset", "trial_type"]).size().to_string()}')


out_path = OUT_STEM + '.pkl'
traces.to_pickle(out_path)
print(f'\nSaved -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)')

# %%
