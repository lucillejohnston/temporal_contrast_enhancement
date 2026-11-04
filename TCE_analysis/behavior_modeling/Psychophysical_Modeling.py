# -*- coding: utf-8 -*-
"""
Fit the Cecchi 2012 ODE model per subject across all three datasets
(plosONE, kneeOA, cLBP).

Loads the combined 1 Hz trace table built by
alter_collab_analysis/data_pipeline/build_combined_traces.py, which gives every
subject a globally unique id and carries trial_type and group_label alongside
the traces. Subjects are addressed by 'subject_uid' (e.g. 'cLBP_0201'), not by
the raw within-study number -- those collide across datasets, and inside cLBP
they collide across substudies.


Notes: 
This script only fits the simplified Cecchi 2012 model which runs faster
Use fit_full_model.py to fit the full Cecchi 2012 model which takes longer

This script fits one set of parameters per subject
Use per_trial_fits.py to fit one set of parameters per trial 

Run analyze_model_fits.py after this to look more detail into the model performance

Author: Lucille Johnston
Updated: 10/05/26
"""
#%%
import pandas as pd
import sys, time, pickle, random, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from psychophysics_modeling_functions import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from datetime import datetime
import glob

DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_traces_1Hz.pkl'
FIG_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/behavior_modeling/figures/' # path for saving figures
RESULTS_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/' # path for saving fit results

data_df = pd.read_pickle(DATA_PATH)

# kneeOA subjects were tested at two sites, forearm and knee
# Fit the two sites separately
data_df['fit_unit'] = np.where(
    data_df['site'].fillna('forearm') == 'forearm',
    data_df['subject_uid'],
    data_df['subject_uid'] + '@' + data_df['site'])

print(f"Loaded {len(data_df):,} rows")
print(f"  {data_df['subject_uid'].nunique()} subjects, "
      f"{data_df.groupby(['subject_uid', 'trial_num']).ngroups:,} trials")
print(data_df.groupby('dataset').agg(
    subjects=('subject_uid', 'nunique'),
    rows=('pain', 'size')).to_string())

# # Parameters from Petre 2017
# petre_params = {
#     'alpha': 2.4932,
#     'beta': 36.7552,
#     'gamma': 0.0204,
#     'lambda_param': 0.0169,
#     'theta': 37.1913,
# }
#%%
# First, determine pain threshold (theta) for each subject
# Keyed on subject_uid, since 'subject' alone is not unique across datasets
subject_thresholds_from_data = {}
for subject_uid in data_df['fit_unit'].unique():
    subject_data = data_df[data_df['fit_unit'] == subject_uid]
    try:
        threshold = extract_threshold_from_data(subject_data, vas_threshold=5)
        subject_thresholds_from_data[subject_uid] = threshold
    except Exception as e:
        print(f"Error processing subject {subject_uid}: {e}")
        subject_thresholds_from_data[subject_uid] = np.nan

thresholds = pd.Series(subject_thresholds_from_data, name='threshold_from_data')
print(f"\nData-derived thresholds for {thresholds.notna().sum()}/{len(thresholds)} subjects")
print(thresholds.describe().round(2).to_string())

missing_theta = sorted(thresholds[thresholds.isna()].index)
if missing_theta:
    print(f"\n⚠️  No threshold (VAS never reached 5) for {len(missing_theta)} subjects:")
    print(f"    {missing_theta}")

# #%%
# # ==================================================================
# # Limits thresholds -- plosONE only, disabled for now
# # ==================================================================
# # combined_data.sqlite only holds threshold_data for plosONE, so
# # 'threshold_limits' is populated for those 137 subjects and NaN everywhere
# # else. The previous version of this block compared against limits_data.csv
# # keyed on the raw subject number, which silently matched kneeOA and cLBP
# # subjects to unrelated plosONE values.
# #
# # When limits data turns up for the other datasets, add rows to the
# # threshold_data table in combined_data.sqlite, re-run build_combined_traces.py,
# # and this block works as-is -- nothing in it is plosONE-specific.
# COMPARE_THRESHOLD_METHODS = False

# if COMPARE_THRESHOLD_METHODS:
#     limits = (data_df.groupby('subject_uid')
#               .agg(threshold_from_limits=('threshold_limits', 'first'),
#                    dataset=('dataset', 'first')))
#     comparison_df = limits.join(thresholds).reset_index()
#     comparison_df['difference'] = (comparison_df['threshold_from_limits']
#                                    - comparison_df['threshold_from_data'])
#     comparison_df = comparison_df.dropna(subset=['threshold_from_limits',
#                                                  'threshold_from_data'])
#     print(f"\nComparing threshold methods for {len(comparison_df)} subjects "
#           f"in {sorted(comparison_df['dataset'].unique())}")

#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

#     # Histogram of thresholds from trial data
#     ax1.hist(comparison_df['threshold_from_data'], bins=15, alpha=0.7,
#              color='blue', edgecolor='black')
#     ax1.set_xlabel('Threshold Temperature (°C)')
#     ax1.set_ylabel('Number of Subjects')
#     ax1.set_title('Thresholds from Trial Data')
#     ax1.grid(True, alpha=0.3)

#     # Histogram of thresholds from limits data
#     ax2.hist(comparison_df['threshold_from_limits'], bins=15, alpha=0.7,
#              color='red', edgecolor='black')
#     ax2.set_xlabel('Threshold Temperature (°C)')
#     ax2.set_ylabel('Number of Subjects')
#     ax2.set_title('Thresholds from Limits Data')
#     ax2.grid(True, alpha=0.3)

#     # Scatter plot comparison
#     correlation = comparison_df['threshold_from_data'].corr(
#         comparison_df['threshold_from_limits'])
#     ax3.scatter(comparison_df['threshold_from_data'],
#                 comparison_df['threshold_from_limits'], alpha=0.7)
#     min_val = min(comparison_df['threshold_from_data'].min(),
#                   comparison_df['threshold_from_limits'].min())
#     max_val = max(comparison_df['threshold_from_data'].max(),
#                   comparison_df['threshold_from_limits'].max())
#     ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Unity')
#     ax3.set_xlabel('Threshold from Trial Data (°C)')
#     ax3.set_ylabel('Threshold from Limits Data (°C)')
#     ax3.set_title(f'Comparing Thresholds (r={correlation:.2f})')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)

#     # Difference histogram
#     ax4.hist(comparison_df['difference'], bins=15, alpha=0.7,
#              color='purple', edgecolor='black')
#     ax4.axvline(x=0, color='r', linestyle='--', label='No Difference')
#     ax4.axvline(x=comparison_df['difference'].mean(), color='b', linestyle='-',
#                 label=f'Mean: {comparison_df["difference"].mean():+.2f}°C')
#     ax4.set_xlabel('Difference (Limits - Data) (°C)')
#     ax4.set_ylabel('Number of Subjects')
#     ax4.set_title('Distribution of Differences')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(f'{FIG_PATH}{datetime.now():%Y%m%d}_thresholdComparison.png', dpi=150)
#     plt.show()

#%%
# Configuration for optimization approach
#
N_SUBJECTS = None           # None = every subject; set an int to subsample
OPTIMIZER = 'de'            # 'de' = differential evolution (global). The objective
                            # goes flat wherever theta exceeds the hottest stimulus,
                            # which strands a gradient-following search.
USE_MULTIPLE_STARTS = True  # only used when OPTIMIZER = 'multistart'
N_STARTS = 10               # only used when OPTIMIZER = 'multistart'
OPTIMIZE_THETA = True       # Fit theta, as Cecchi 2012 do 
TRIAL_TYPES = None          # None = all trial types, or e.g. ['offset', 'onset']

# Output paths
timestamp = datetime.now().strftime('%Y%m%d')
RESULTS_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/behavior_modeling/model_fit_results/'
RESULTS_FILE = f"{RESULTS_PATH}optimization_results_{timestamp}.pkl"

# Select subjects. Those with no usable threshold cannot be fit unless theta
# is being optimized, so drop them up front rather than failing mid-loop.
subjects_all = sorted(data_df['fit_unit'].unique())
if not OPTIMIZE_THETA:
    subjects_all = [s for s in subjects_all
                    if not np.isnan(subject_thresholds_from_data.get(s, np.nan))]

if N_SUBJECTS is not None and N_SUBJECTS < len(subjects_all):
    random.seed(42)
    subjects_to_process = sorted(random.sample(subjects_all, k=N_SUBJECTS))
else:
    subjects_to_process = subjects_all

print(f"\n{'='*60}")
print(f"🚀 OPTIMIZATION CONFIGURATION")
print(f"{'='*60}")
print(f"Model: Simplified Cecchi 2012 (per-trial integration from p=0)")
print(f"Subjects to process: {len(subjects_to_process)}")
print(f"Optimizer: {OPTIMIZER}")
print(f"Optimize theta: {OPTIMIZE_THETA}")
print(f"Trial types: {TRIAL_TYPES or 'all'}")
print(f"Results will be saved to: {RESULTS_FILE}")
print(f"{'='*60}\n")

# Initialize results dictionary
optimization_results = {}

# # Try to load existing checkpoint
# PREV_RESULTS_FILE = sorted(glob.glob(RESULTS_PATH + 'optimization_results_*.pkl'))[-1]
# if os.path.exists(PREV_RESULTS_FILE):
#     try:
#         with open(PREV_RESULTS_FILE, 'rb') as f:
#             checkpoint_data = pickle.load(f)
#             optimization_results = checkpoint_data.get('optimization_results', {})
#         print(f"📂 Loaded checkpoint with {len(optimization_results)} subjects already processed\n")
#     except Exception as e:
#         print(f"⚠️  Could not load checkpoint: {e}\n")

#%%
# Time a single subject before committing to the full run.
# 310 subjects x N_STARTS optimizations is a long job -- fit one first so the
# projected runtime is known before launching it.
TIME_ONE_SUBJECT = True

if TIME_ONE_SUBJECT:
    probe_uid = subjects_to_process[0]
    probe_data = data_df[data_df['fit_unit'] == probe_uid]
    t0 = time.time()
    probe_params, _ = optimize_cecchi_simplified(
        probe_data,
        threshold=subject_thresholds_from_data[probe_uid],
        initial_params=None,
        use_multiple_starts=USE_MULTIPLE_STARTS,
        n_starts=N_STARTS,
        optimizer=OPTIMIZER,
        verbose=True)
    elapsed = time.time() - t0
    projected = elapsed * len(subjects_to_process) / 60
    print(f"\n⏱  {probe_uid} took {elapsed:.1f}s "
          f"({probe_params['n_trials'] if probe_params else 0} trials)")
    print(f"   Projected for {len(subjects_to_process)} subjects: "
          f"{projected:.0f} min ({projected/60:.1f} h)")

#%%
# Process each subject
total_start_time = time.time()
for subject_idx, subject_uid in enumerate(subjects_to_process):
    # Skip if already processed
    if subject_uid in optimization_results:
        print(f"⏭️  Subject {subject_uid} already processed, skipping...\n")
        continue

    print(f"\n{'='*60}")
    print(f"📊 Processing subject {subject_uid} ({subject_idx+1}/{len(subjects_to_process)})")
    print(f"{'='*60}")

    subject_start_time = time.time()

    # Get subject data
    subject_data = data_df[data_df['fit_unit'] == subject_uid]
    if TRIAL_TYPES is not None:
        subject_data = subject_data[subject_data['trial_type'].isin(TRIAL_TYPES)]
        if subject_data.empty:
            print(f"⚠️  No {TRIAL_TYPES} trials for {subject_uid}, skipping...")
            continue

    # Get threshold (use data-derived unless optimizing)
    if OPTIMIZE_THETA:
        threshold = None
    else:
        threshold = subject_thresholds_from_data.get(subject_uid, np.nan)
        if np.isnan(threshold):
            print(f"⚠️  No threshold found for subject {subject_uid}, skipping...")
            continue
        print(f"Using data-derived threshold: {threshold:.2f}°C")

    try:
        # Run optimization
        best_params, best_result = optimize_cecchi_simplified(
            subject_data,
            threshold=threshold,
            initial_params=None,  # Use defaults
            use_multiple_starts=USE_MULTIPLE_STARTS,
            n_starts=N_STARTS,
            optimizer=OPTIMIZER,
            verbose=True
        )

        # Store results
        if best_params is not None:
            # Carry labels through so downstream analysis needs no re-merge
            meta = subject_data.iloc[0]
            best_params['dataset'] = meta['dataset']
            best_params['site'] = meta['site']
            best_params['subject'] = meta['subject_uid']
            best_params['study'] = meta['study']
            best_params['group_label'] = meta['group_label']

            optimization_results[subject_uid] = {
                'params': best_params,
                'result': best_result,
                'timestamp': datetime.now().isoformat()
            }

            subject_time = time.time() - subject_start_time
            print(f"\n✅ Subject {subject_uid} completed in {subject_time:.1f}s")
            print(f"   Final MSE: {best_params['mse']:.2f}")
            print(f"   Parameters: ᾱ={best_params['alpha_bar']:.4f}, γ̄={best_params['gamma_bar']:.4f}, θ={best_params['theta']:.2f}")


            # Save checkpoint after each subject
            checkpoint_data = {
                'optimization_results': optimization_results,
                'config': {
                    'n_subjects': len(subjects_to_process),
                    'use_multiple_starts': USE_MULTIPLE_STARTS,
                    'n_starts': N_STARTS,
                    'optimize_theta': OPTIMIZE_THETA,
                    'trial_types': TRIAL_TYPES,
                    'traces_file': DATA_PATH,
                    'timestamp': timestamp
                }
            }

            with open(RESULTS_FILE, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"   💾 Checkpoint saved")

        else:
            print(f"\n❌ Subject {subject_uid} optimization failed")

    except Exception as e:
        print(f"\n❌ Error processing subject {subject_uid}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Final summary
total_time = time.time() - total_start_time
print(f"\n{'='*60}")
print(f"📊 OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Subjects processed: {len(optimization_results)}/{len(subjects_to_process)}")
print(f"Average time per subject: {total_time/max(len(optimization_results),1):.1f}s")
print(f"Results saved to: {RESULTS_FILE}")
print(f"{'='*60}\n")


#%%
# ==================================================================
# Write out flat tables for analysis
# ==================================================================
# The pickle keeps everything, including the predicted traces needed for
# replotting, but it is a nested dict and awkward to compute on. These two
# CSVs are the analysis-ready views: one row per subject for parameter
# comparisons, one row per trial for the questions about trial type, group
# and trial order.
with open(RESULTS_FILE, 'rb') as f:
    checkpoint_data = pickle.load(f)
    optimization_results = checkpoint_data['optimization_results']

subject_rows, trial_rows = [], []
for uid, res in optimization_results.items():
    p = res['params']
    base = {'subject_uid': uid,
            'subject': p.get('subject'), 'site': p.get('site'),
            'dataset': p.get('dataset'),
            'study': p.get('study'),
            'group_label': p.get('group_label')}
    subject_rows.append({
        **base,
        'alpha_bar': p['alpha_bar'], 'gamma_bar': p['gamma_bar'], 'theta': p['theta'],
        'r': p.get('r'), 'r2': p.get('r2'), 'mse': p.get('mse'), 'sse': p.get('sse'),
        'n_trials': p.get('n_trials'), 'n_points': p.get('n_points'),
        # Non-empty means a bound, not the data, chose that parameter
        'at_bounds': ','.join(p.get('at_bounds') or []),
    })
    for tf in p.get('trial_fits', []):
        trial_rows.append({**base,
                           'alpha_bar': p['alpha_bar'], 'gamma_bar': p['gamma_bar'],
                           'theta': p['theta'], **tf})

fit_summary = pd.DataFrame(subject_rows).sort_values('subject_uid')
trial_summary = pd.DataFrame(trial_rows).sort_values(['subject_uid', 'trial_num'])

# cLBP encodes a repeat visit as trial_num + 100, so recover session and
# within-session order -- the across-trial analysis is within session.
trial_summary['visit'] = trial_summary['trial_num'] // 100
trial_summary['trial_in_visit'] = trial_summary['trial_num'] % 100
trial_summary['trial_order'] = (trial_summary
                                .groupby(['subject_uid', 'visit'])['trial_in_visit']
                                .rank(method='dense').astype(int))

SUBJECT_CSV = f"{RESULTS_PATH}model_fits_subject_{timestamp}.csv"
TRIAL_CSV = f"{RESULTS_PATH}model_fits_trial_{timestamp}.csv"
fit_summary.to_csv(SUBJECT_CSV, index=False)
trial_summary.to_csv(TRIAL_CSV, index=False)

print(f"\nFitted {len(fit_summary)} subjects, {len(trial_summary)} trials")
print(f"  {SUBJECT_CSV}")
print(f"  {TRIAL_CSV}")
print(f"  {RESULTS_FILE}  (full results incl. predicted traces)")

print(f"\nPer-subject parameters (median by dataset):")
print(fit_summary.groupby('dataset')[['alpha_bar', 'gamma_bar', 'theta']]
      .median().round(3).to_string())
print(f"\nFit quality -- r per trial is the measure comparable to Cecchi 2012 "
      f"(they report 0.92 simple / 0.88 complex):")
print(trial_summary.groupby('dataset')[['r', 'r2']].median().round(3).to_string())

n_flagged = (fit_summary['at_bounds'] != '').sum()
if n_flagged:
    print(f"\n⚠️  {n_flagged}/{len(fit_summary)} subjects have a parameter on a bound "
          f"(treat those fits as suspect):")
    print(fit_summary[fit_summary['at_bounds'] != '']['at_bounds']
          .value_counts().to_string())

#%%
# ==================================================================
# Example fits -- a sample rather than all 310 figures
# ==================================================================
PLOT_EVERY_N = 40   # 1 in 50 subjects
rng = np.random.default_rng(21)
uids = sorted(optimization_results.keys())
sampled = sorted(rng.choice(uids, size=max(1, len(uids) // PLOT_EVERY_N),
                            replace=False))
print(f"\nPlotting {len(sampled)} of {len(uids)} subjects: {sampled}")
for uid in sampled:
    plot_subject_trials(data_df[data_df['fit_unit'] == uid],
                        optimization_results[uid]['params'],
                        subject_uid=uid,
                        save_path=f"{FIG_PATH}fit_{uid}_{datetime.now():%Y%m%d}.png")

# Print summary table
print_optimization_summary(optimization_results)
# %%
