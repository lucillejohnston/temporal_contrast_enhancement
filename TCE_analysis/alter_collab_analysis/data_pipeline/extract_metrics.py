#%%
"""
Section 1: Imports and Data Loading

Loads trial data from JSON into a pandas DataFrame.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import Dict, Any
import sys
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/alter_collab_analysis/')
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/behavior_modeling/')
from utils.plotting_functions import *
from utils.subject_ids import canonical_clbp_id
from trial_windows import measure_temperature_edges, TrialWindowError

# =========================================================
# CONFIGURATION
# =========================================================

# Path to the cleaned, aligned trial data. This is the 10Hz file. The
# trimmed_downsampled file this used to point at carries only subject,
# trial_num, aligned_time, temperature and pain -- no trial_type, study or
# site -- so nothing below could ever have run on it.
dataset = os.environ.get('TCE_DATASET', 'cLBP')  # 'plosONE', 'kneeOA', 'cLBP'
DATA_PATH = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_data_cleaned_aligned.json'
OUTPUT_PATH = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_metrics.json'

# A/B/C windows are MEASURED from each trial's own temperature trace
# (behavior_modeling/trial_windows.py) instead of counted forward from an
# assumed 5 / 0.67 / 5 / 0.67 / 20s schedule anchored on a detected ramp time.
# The thermode does not keep to that schedule: the real step into C runs
# 0.7-1.7s later than assumed, so windows built that way straddle the step and
# mix two stimulus levels. 0.15C is the right tolerance for this 10Hz data;
# the 1Hz traces need 0.3 because their samples are averages.
AT_LEVEL = 0.15

# Length of period C used for the metrics, measured forward from the MEASURED
# start of C. The start is what was wrong before and is now measured; the
# duration is held fixed because auc_C is an integral and a longer window
# inflates it. Measured C runs 20.4-21.1s in plosONE and kneeOA but 22-23s in
# cLBP, whose traces are trimmed before the ramp down so C simply ends at the
# last sample -- taking that at face value would make cLBP's auc_C ~15% larger
# for a reason that is an artefact of trimming rather than a real difference.
# Set to None to use the fully measured C_end instead.
C_DURATION = 20.0

window_size = 5
period_c_duration = 20
trial_definitions = {
    'inv': ['T2', 'T1', 'T2'], # in plosONE dataset they are called 'inv'
    'onset': ['T2', 'T1', 'T2'], # in kneeOA and cLBP datasets they are called 'onset'
    't2_hold': ['T2', 'T2', 'T2'],
    'offset': ['T1', 'T2', 'T1'], 
    't1_hold': ['T1', 'T1', 'T1'],
    'stepdown': ['T2', 'T2', 'T1'], # in the plosONE dataset there are these stepdown trials
    'innocuous': ['T0','T0','T0'] # in kneeOA dataset there are innocuous trials
}
if dataset == 'plosONE':
    trial_type_info = {
        'inv':      {'kind': 'stepped', 'extrema_order': ['min', 'max'], 'reference': None},
        'offset':   {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
        'stepdown': {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
        't1_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': ['offset','stepdown']},
        't2_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': 'inv'},
    }
elif dataset == 'kneeOA':
    trial_type_info = {
        'onset':    {'kind': 'stepped', 'extrema_order': ['min', 'max'], 'reference': None},
        'offset':   {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
        't1_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': ['offset']},
        't2_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': ['onset']},
        'innocuous': {'kind': 'control', 'extrema_order': ['control'],   'reference': None}
    }
elif dataset == 'cLBP':
    trial_type_info = {
        'onset':   {'kind': 'stepped', 'extrema_order': ['min', 'max'], 'reference': None},
        'offset':  {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
        't1_hold': {'kind': 'control', 'extrema_order': ['control'],    'reference': ['offset']},
        't2_hold': {'kind': 'control', 'extrema_order': ['control'],    'reference': ['onset']}
    }

# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_json(DATA_PATH, orient='records')

# DataFrame columns:
# subject - subject ID
# trial_num - trial number (unique per subject/session)
# trial_type - e.g., 'offset','inv'
# temperature - temperature at each time point
# pain - pain rating at each time point 
# aligned_time - 10Hz sampled time, aligned to trial start 
# temperature_aligned_for_plot - temperature shifted for plotting

# for kneeOA data, filter for site == forearm
if dataset == 'kneeOA':
    df = df[df['site'] == 'forearm'].copy()

# Only the cLBP file carries 'study' (cLBP_DPOP / cLBP_MBPR), because that is
# the only dataset built from two substudies. Giving the other two a constant
# study column lets the grouping below be written once for all three.
if 'study' not in df.columns:
    df['study'] = dataset

# cLBP subject numbers are within-substudy, and 7 of them (1, 10, 22, 40, 43,
# 48, 53) are used by BOTH substudies -- 7 pairs of different people sharing a
# key. The output below is stored as structured_data[subject][trial_num], so
# without this the second person of each pair overwrites the first. Mapping
# onto the lab's DPOP_ID namespace (MBPR 1 -> 201) separates them and matches
# what build_combined_traces.py already does, so the metrics and the traces
# describe the same people.
if dataset == 'cLBP':
    df['subject_within_study'] = df['subject']
    df['subject'] = [canonical_clbp_id(s, st)
                     for s, st in zip(df['subject'], df['study'])]
    print(f'cLBP: {df.groupby(["study"])["subject"].nunique().to_dict()} subjects '
          f'-> {df["subject"].nunique()} distinct DPOP_IDs')


#%%
# =========================================================
# FUNCTIONS
# =========================================================

# Utility Functions
def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Filter calibration trials matching T1 hold temperature
import numpy as np
from scipy.spatial.distance import euclidean

# Time Window and AUC Functions
def measured_timewindows(trial_df, trial_type):
    """
    A/B/C windows for one trial, measured from its own temperature trace.

    Replaces define_timewindows (assumed 5 / 0.67 / 5 / 0.67 / 20s from a
    detected ramp time) and get_trial_ramp_time. Detection is level-based:
    find what temperature the thermode is sitting at, not how fast it is
    changing -- see trial_windows.py for why the rate-based version fails.

    Returns ({'A': (start, end), 'B': ..., 'C': ...}, raw_measurement) so the
    first element drops into everything downstream unchanged.
    Raises TrialWindowError if the trial's plateaus cannot be found.
    """
    edges = measure_temperature_edges(
        trial_df['aligned_time'].to_numpy(dtype=float),
        trial_df['temperature'].to_numpy(dtype=float),
        trial_type, at_level=AT_LEVEL, flat_tol=AT_LEVEL)
    if C_DURATION is not None:
        edges['C_end'] = min(edges['C_end'], edges['C_start'] + C_DURATION)
    windows = {p: (edges[f'{p}_start'], edges[f'{p}_end']) for p in ['A', 'B', 'C']}
    return windows, edges

def compute_aucs(pain_series, time_windows):
    """
    AUC, mean rating and duration for each period.

    Parameters:
        pain_series (pd.Series): pain ratings indexed by aligned_time.
        time_windows (dict): {'A': (start, end), ...} from measured_timewindows.

    Returns:
        dict: auc_/mean_/duration per period, plus auc_total.

    mean_* is new and is what cross-period or cross-dataset comparisons should
    use. AUC scales with window length, and measured periods are not all
    exactly 5/5/20s -- a step trial's period A runs ~5.3s because that is how
    long the thermode actually held T1, while a hold trial's is 5.0s by
    convention. Comparing their AUCs therefore compares window lengths as well
    as pain; comparing their means does not.
    """
    aucs = {}
    auc_total = 0
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        window = pain_series.loc[(pain_series.index >= start) & (pain_series.index <= end)]
        if window.empty:
            auc = np.nan
            aucs[f'mean_{period}'] = np.nan
        else:
            auc = np.trapezoid(window.values, x=window.index)
            aucs[f'mean_{period}'] = float(window.mean())
        aucs[f'auc_{period}'] = auc
        aucs[f'{period}_duration'] = end - start
        auc_total += auc if not np.isnan(auc) else 0
    aucs['auc_total'] = auc_total
    return aucs

def extract_extrema(
        pain_series: pd.Series,
        trial_type: str,
        time_windows: dict,
        mode: str = 'local',
        reference_times: dict = None,
    ) -> dict:
    """
    Extract extrema (min/max) and related metrics from a pain series.

    Parameters:
        pain_series (pd.Series): Pain ratings indexed by aligned_time.
        trial_type (str): The trial type key.
        time_windows (dict): {'A': (start, end), ...} from measured_timewindows.
        mode (str): 'local' for local extrema, 'control' for extracting at reference times.
        reference_times (dict): Dict with 'min_time' and 'max_time' (used if mode='control').
    Returns:
        dict: Extrema values, times, peak-to-peak, and latency.
    """
    if mode == 'local': # for offset, inv, stepdown trials
        return _extract_local_extrema(pain_series, trial_type, time_windows)
    elif mode == 'control': # for t1_hold, t2_hold trials
        return _extract_control_extrema(pain_series, trial_type, reference_times, time_windows)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _extract_local_extrema(
        pain_series: pd.Series,
        trial_type: str,
        time_windows: dict
        )-> Dict[str, float]:

        transition_time = time_windows['C'][0]  # Use the start of period C
        # Define search windows
        window_start = transition_time - window_size
        window_end = transition_time + window_size
        c_window_end = transition_time + period_c_duration 

        window_series = pain_series.loc[
            (pain_series.index >= window_start) & 
            (pain_series.index <= window_end)]
        
        if window_series.empty:
            return {x: np.nan for x in ['abs_min_val', 'abs_min_time', 'abs_max_val', 'abs_max_time', 'abs_peak_to_peak', 'abs_peak_to_peak_latency']}
        
        extrema_mode = trial_type_info[trial_type]['extrema_order']

        if extrema_mode == ['min', 'max']:      # for 'inv'/'onset' trials
            first_val = window_series.min()
            first_time = window_series.idxmin()
            after_first = pain_series.loc[(pain_series.index > first_time) & (pain_series.index <= c_window_end)]
            second_val = after_first.max()
            second_time = after_first.idxmax()
            min_val, min_time = first_val, first_time
            max_val, max_time = second_val, second_time
        elif extrema_mode == ['max', 'min']:    # for 'offset' and 'stepdown' trials
            first_val = window_series.max()
            first_time = window_series.idxmax()
            after_first = pain_series.loc[(pain_series.index > first_time) & (pain_series.index <= c_window_end)]
            second_val = after_first.min()
            second_time = after_first.idxmin()
            max_val, max_time = first_val, first_time
            min_val, min_time = second_val, second_time
        else:
            raise ValueError(f"Unknown extrema order for trial_type={trial_type}: {extrema_mode}")
        min_val = max(min_val, 0) # floor at 0
        peak_to_peak = (second_val - first_val) if not (np.isnan(second_val) or np.isnan(first_val)) else np.nan
        peak_to_peak_latency = second_time - first_time if not (np.isnan(second_time) or np.isnan(first_time)) else np.nan
        
        return {
            # Only return absolute values for experimental trials
            'abs_min_val': min_val, 
            'abs_min_time': min_time,
            'abs_max_val': max_val, 
            'abs_max_time': max_time,
            'abs_peak_to_peak': peak_to_peak,
            'abs_peak_to_peak_latency': peak_to_peak_latency
        }
    
def _extract_control_extrema(
        pain_series: pd.Series,
        trial_type: str,
        reference_times: Dict[str, float],
        time_windows: dict
        ) -> Dict[str, float]:
        """Extract time-yoked and absolute extrema for control trials"""

        # Absolute maximum and subsequent minimum
        abs_max_time = pain_series.idxmax()
        abs_max_val = pain_series.max()

        # Get period C boundaries
        c_start, c_end = time_windows['C']

        # Find minimum after max within period C only
        after_max_in_c = pain_series.loc[
            (pain_series.index > abs_max_time) & 
            (pain_series.index >= c_start) & 
            (pain_series.index <= c_end)
        ]
        if not after_max_in_c.empty:
            abs_min_val = after_max_in_c.min()
            abs_min_time = after_max_in_c.idxmin()
        else:
            # Fallback: just find min in period C
            period_c_data = pain_series.loc[
                (pain_series.index >= c_start) & 
                (pain_series.index <= c_end)
            ]
            if not period_c_data.empty:
                abs_min_val = period_c_data.min()
                abs_min_time = period_c_data.idxmin()
            else:
                abs_min_val = np.nan
                abs_min_time = np.nan

        # Extract reference time for time-yoked extrema 
        ref_min_time = reference_times.get('abs_min_time', np.nan)
        ref_max_time = reference_times.get('abs_max_time', np.nan)

        if not pain_series.empty:
            # Find the nearest index for min_time and max_time
            idx_array = np.array(pain_series.index)
            if not np.isnan(ref_min_time):
                min_idx = np.argmin(np.abs(idx_array - ref_min_time))
                time_yoked_min_val = pain_series.iloc[min_idx]
            else:
                time_yoked_min_val = np.nan
            if not np.isnan(ref_max_time):
                max_idx = np.argmin(np.abs(idx_array - ref_max_time))
                time_yoked_max_val = pain_series.iloc[max_idx]
            else:
                time_yoked_max_val = np.nan
        else:
            time_yoked_min_val = np.nan
            time_yoked_max_val = np.nan

        # Peak to peak for time-yoked extrema
        time_yoked_peak_to_peak = (time_yoked_max_val - time_yoked_min_val) if not (
            np.isnan(time_yoked_max_val) or np.isnan(time_yoked_min_val)) else np.nan
        time_yoked_peak_to_peak_latency = ref_max_time - ref_min_time if not (
            np.isnan(ref_max_time) or np.isnan(ref_min_time)) else np.nan

        # Peak to peak for absolute extrema (preserving sign based on time order)
        if not (np.isnan(abs_max_val) or np.isnan(abs_min_val)):
            # First in time minus second in time
            if abs_max_time < abs_min_time:
                abs_peak_to_peak = abs_max_val - abs_min_val
            else:
                abs_peak_to_peak = abs_min_val - abs_max_val
        else:
            abs_peak_to_peak = np.nan

        abs_peak_to_peak_latency = abs_min_time - abs_max_time if not (
            np.isnan(abs_min_time) or np.isnan(abs_max_time)) else np.nan

        return {
            # Time-yoked extrema
            'time_yoked_min_val': time_yoked_min_val,
            'time_yoked_max_val': time_yoked_max_val,
            'time_yoked_min_time': ref_min_time,
            'time_yoked_max_time': ref_max_time,
            'time_yoked_peak_to_peak': time_yoked_peak_to_peak,
            'time_yoked_peak_to_peak_latency': time_yoked_peak_to_peak_latency,

            # Absolute extrema
            'abs_min_val': abs_min_val,
            'abs_min_time': abs_min_time,
            'abs_max_val': abs_max_val,
            'abs_max_time': abs_max_time,
            'abs_peak_to_peak': abs_peak_to_peak,
            'abs_peak_to_peak_latency': abs_peak_to_peak_latency
        }

def calc_normalized_pain_change(row, use_time_yoked=True):
    """
    Calculate normalized pain change based on the trial type.
    Parameters:
        row: DataFrame row with trial metrics
        use_time_yoked: If True, use time_yoked values for control trials;
                        If False, use absolute values
    Returns:
        float: Normalized pain change percentage
    """
    tt = row['trial_type']
    if tt in ['offset','stepdown']:
        # For offset/stepdown trials: (min - max) / max * 100 
        min_val = row.get('abs_min_val')
        max_val = row.get('abs_max_val')
        if pd.notnull(min_val) and pd.notnull(max_val) and max_val !=0:
            return (min_val - max_val) / max_val * 100 
    
    elif tt in ['inv','onset']:
        # For inv/onset trials: (max - min) / max * 100
        min_val = row.get('abs_min_val')
        max_val = row.get('abs_max_val')
        if pd.notnull(min_val) and pd.notnull(max_val) and max_val !=0:
            return (max_val - min_val) / max_val * 100
    elif tt == 't1_hold':
        # t1_hold references 'offset' ---- practically, I don't really care about stepdown trials yet 
        if use_time_yoked:
            min_val = row.get('time_yoked_min_val_offset')
            max_val = row.get('time_yoked_max_val_offset')
        else:
            min_val = row.get('abs_min_val')
            max_val = row.get('abs_max_val')
        if pd.notnull(min_val) and pd.notnull(max_val) and max_val !=0:
            return (min_val - max_val) / max_val * 100
    elif tt == 't2_hold':
        # t2_hold references 'inv' ---- practically, I don't really care about stepdown trials yet 
        if use_time_yoked:
            min_val = row.get('time_yoked_min_val_onset' if dataset in ['kneeOA','cLBP'] else 'time_yoked_min_val_inv')
            max_val = row.get('time_yoked_max_val_onset' if dataset in ['kneeOA','cLBP'] else 'time_yoked_max_val_inv')
        else:
            min_val = row.get('abs_min_val')
            max_val = row.get('abs_max_val')
        if pd.notnull(min_val) and pd.notnull(max_val) and max_val !=0:
            return (max_val - min_val) / max_val * 100
    return np.nan 

#%%
# ========================================================
# METRICS EXTRACTION
# ========================================================

# Compute metrics for each individual trial
trial_metrics_list = []
subject_extrema_times = {}
unmeasurable_trials = []

# Extract stepped trials first to store their extrema times
for (subject, trial_num, study), trial_df in df.groupby(['subject', 'trial_num', 'study']):
    trial_type = trial_df['trial_type'].iloc[0]
    if trial_type not in trial_definitions:
        continue # skip other trial types
    trial_df = trial_df.sort_values('aligned_time')
    pain_df = (
        trial_df[['aligned_time', 'pain']]
        .dropna(subset=['pain'])
        .drop_duplicates(subset=['aligned_time'])
        .sort_values('aligned_time')
        )

    pain_series = pd.Series(
        pain_df['pain'].to_numpy(),
        index=pain_df['aligned_time'].to_numpy()
    )
    info = trial_type_info[trial_type]
    if info['kind'] == 'control':
        continue # skip control trials for now
    # Measure this trial's A/B/C windows from its own temperature trace.
    try:
        time_windows, edges = measured_timewindows(trial_df, trial_type)
    except TrialWindowError as e:
        unmeasurable_trials.append((subject, trial_num, trial_type, str(e)))
        continue
    time_window_record = {}
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        time_window_record[f'{period}_start'] = start
        time_window_record[f'{period}_end'] = end
    time_window_record['P1'] = edges['P1']   # first plateau temperature, °C
    time_window_record['P2'] = edges['P2']   # second plateau, or nan in holds

    # Compute AUCs
    aucs = compute_aucs(pain_series, time_windows)
    # Compute extrema
    extrema = extract_extrema(
        pain_series, trial_type, time_windows, mode='local')
    # Store extrema times for later use with control trials
    subject_extrema_times[(int(subject), str(trial_type), int(trial_num))] = {
        'abs_min_time': extrema['abs_min_time'],
        'abs_max_time': extrema['abs_max_time']
    }
    record = {
    'subject': subject,
    'trial_num': trial_num,
    'trial_type': trial_type,
    'study': study,
    **aucs,
    **extrema,
    **time_window_record
    }
    trial_metrics_list.append(record)

# Now extract control trials using reference times from stepped trials
for (subject, trial_num, study), trial_df in df.groupby(['subject', 'trial_num', 'study']):
    trial_type = trial_df['trial_type'].iloc[0]
    if trial_type not in trial_definitions:
        continue # skip other trial types
    trial_df = trial_df.sort_values('aligned_time')
    pain_df = (
        trial_df[['aligned_time', 'pain']]
        .dropna(subset=['pain'])
        .drop_duplicates(subset=['aligned_time'])
        .sort_values('aligned_time')
        )

    pain_series = pd.Series(
        pain_df['pain'].to_numpy(),
        index=pain_df['aligned_time'].to_numpy()
    )
    info = trial_type_info[trial_type]
    if info['kind'] != 'control':
        continue # skip stepped trials, already processed

    # Measure this trial's A/B/C windows from its own temperature trace. Hold
    # trials have no step, so only C is measured; A and B stay 5s each.
    try:
        time_windows, edges = measured_timewindows(trial_df, trial_type)
    except TrialWindowError as e:
        unmeasurable_trials.append((subject, trial_num, trial_type, str(e)))
        continue
    time_window_record = {}
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        time_window_record[f'{period}_start'] = start
        time_window_record[f'{period}_end'] = end
    time_window_record['P1'] = edges['P1']
    time_window_record['P2'] = edges['P2']

    # Compute AUCs
    aucs = compute_aucs(pain_series, time_windows)
    # Get reference times from the corresponding stepped trial
    references = info['reference']
    if isinstance(references, list):                # for t1_hold: multiple references
        extrema_dict = {}
        ref_trial_num_dict = {}
        for ref in references:
            # Find all reference trials for this subject/ref
            ref_trials = df[(df['subject'] == subject) & (df['trial_type'] == ref)]
            if not ref_trials.empty:
                # Find the reference trial_num with the closest trial_num
                ref_trial_nums = ref_trials['trial_num'].unique()
                trial_num_diffs = np.abs(ref_trial_nums - trial_num)
                min_diff_idx = np.argmin(trial_num_diffs)
                ref_trial_num = ref_trial_nums[min_diff_idx]
            else:
                ref_trial_num = None 

            if ref_trial_num is not None:
                reference_times = subject_extrema_times.get(
                    (int(subject), str(ref), int(ref_trial_num)), 
                    {'abs_min_time': np.nan, 'abs_max_time': np.nan})
            else:
                reference_times = {'abs_min_time': np.nan, 'abs_max_time': np.nan}
            # TEMP debugging prints to verify correct reference trial selection
            extrema = extract_extrema(
                pain_series, trial_type, time_windows, mode='control',
                reference_times=reference_times)
            
            # Add reference suffix only to reference-dependent extrema
            for k, v in extrema.items():
                if k.startswith('time_yoked_'): # only add suffix to time-yoked extrema
                    extrema_dict[f"{k}_{ref}"] = v
                else:                           # absolute extrema
                    if k not in extrema_dict: 
                        extrema_dict[k] = v # don't overwrite

            ref_trial_num_dict[f"reference_trial_num_{ref}"] = ref_trial_num

        record = {
            'subject': subject,
            'trial_num': trial_num,
            'trial_type': trial_type,
            'study': study,
            **aucs,
            **extrema_dict,
            **ref_trial_num_dict,
            **time_window_record
        }
        trial_metrics_list.append(record)

    else:                                       # For t2_hold: just do the single reference
        ref = references
        ref_trials = df[(df['subject'] == subject) & (df['trial_type'] == ref)]
        if not ref_trials.empty:
            ref_trial_nums = ref_trials['trial_num'].unique()
            trial_num_diffs = np.abs(ref_trial_nums - trial_num)
            min_diff_idx = np.argmin(trial_num_diffs)
            ref_trial_num = ref_trial_nums[min_diff_idx]
        else:
            ref_trial_num = None
        if ref_trial_num is not None:
            reference_times = subject_extrema_times.get(
                (int(subject), str(ref), int(ref_trial_num)), 
                {'abs_min_time': np.nan, 'abs_max_time': np.nan})
        else:
                reference_times = {'abs_min_time': np.nan, 'abs_max_time': np.nan}
        extrema = extract_extrema(
            pain_series, trial_type, time_windows, mode='control',
            reference_times=reference_times)

        extrema_dict = {}

        # Add reference suffix only to reference-dependent extrema
        for k, v in extrema.items():
            if k.startswith('time_yoked_'): # only add suffix to time-yoked extrema
                extrema_dict[f"{k}_{ref}"] = v
            else:                           # absolute extrema
                if k not in extrema_dict: 
                    extrema_dict[k] = v # don't overwrite

        record = {
            'subject': int(subject),
            'trial_num': int(trial_num),
            'trial_type': trial_type,
            'study': study,
            'reference_trial_num': ref_trial_num,
            **aucs,
            **extrema_dict,        
            **time_window_record
        }
        trial_metrics_list.append(record)


# Calculate normalized pain changes
for record in trial_metrics_list:
    record['time_yoked_normalized_pain_change'] = calc_normalized_pain_change(record, use_time_yoked=True)
    record['abs_normalized_pain_change'] = calc_normalized_pain_change(record, use_time_yoked=False)

if unmeasurable_trials:
    print(f'\n⚠️  {len(unmeasurable_trials)} trials had no findable plateaus and '
          f'were skipped:')
    for subj, tnum, ttype, why in unmeasurable_trials[:10]:
        print(f'     subject {subj} trial {tnum} ({ttype}): {why}')
    if len(unmeasurable_trials) > 10:
        print(f'     ... and {len(unmeasurable_trials) - 10} more')

# Convert to serializable format and organize by subject
structured_data = {}
overwritten = []
for record in trial_metrics_list:
    subject = int(record['subject'])
    trial_num = int(record['trial_num'])

    if subject not in structured_data:
        structured_data[subject] = {}
    # Two records on one key means two different people sharing a subject
    # number -- the cLBP DPOP/MBPR case this script now re-codes against. If it
    # ever fires again, one person's metrics are being discarded.
    if trial_num in structured_data[subject]:
        overwritten.append((subject, trial_num,
                            structured_data[subject][trial_num].get('study'),
                            record.get('study')))

    # Convert all values to JSON-serializable format
    clean_record = {}
    for key, value in record.items():
        clean_record[key] = convert_to_serializable(value)

    structured_data[subject][trial_num] = clean_record

if overwritten:
    raise RuntimeError(
        f'{len(overwritten)} trials collided on (subject, trial_num) and would '
        f'be silently overwritten, e.g. {overwritten[:3]}. Subject IDs are not '
        f'unique -- fix the namespace before trusting this output.')

# Save as JSON
with open(OUTPUT_PATH, 'w') as f:
    json.dump(structured_data, f, indent=2)

print(f"\nTrial metrics saved to {OUTPUT_PATH}")
print(f'  {len(structured_data)} subjects, '
      f'{sum(len(v) for v in structured_data.values()):,} trials')

# The cells below are interactive: the plot asks for trial types that only some
# datasets have, and the debugging block is pinned to one subject and trial. Set
# TCE_METRICS_ONLY=1 to stop here when running the file straight through for all
# three datasets.
if os.environ.get('TCE_METRICS_ONLY'):
    sys.exit(0)

# %%
# ========================================================
# TRIAL COMPARISON PLOTTING
# ========================================================
# the cLBP pain data is so sparse, it breaks the code
# for the sake of visualization, let's interpolate 


# Random selection (original behavior)
plot_trial_comparison(structured_data, df, 'onset', 't2_hold', 'onset')

# # # Specific subject and specific control trial
# subjects = [117, 121, 131, 144, 154, 158, 158, 161, 177]
# trials = [1, 8, 7, 9, 3, 3, 9, 7, 7]
# for subj, trial in zip(subjects, trials):
#     plot_trial_comparison(structured_data, df, 'inv', 't2_hold', 'inv', specific_subject=subj, specific_control_trial=trial)
# plot_trial_comparison(structured_data, df, 'onset','t2_hold','onset', specific_subject=7, specific_control_trial=7)

# %%
# Debugging weird trials
# Let's find the trial that actually has >1000%
print("=== FINDING TRIALS WITH EXTREME NORMALIZED PAIN CHANGE ===")

extreme_trials = []
for subject_id, trials in structured_data.items():
    for trial_num, trial_data in trials.items():
        abs_change = trial_data.get('abs_normalized_pain_change')
        time_yoked_change = trial_data.get('time_yoked_normalized_pain_change')
        
        if abs_change and abs(abs_change) > 500:
            extreme_trials.append((subject_id, trial_num, 'abs', abs_change, trial_data))
        
        if time_yoked_change and abs(time_yoked_change) > 500:
            extreme_trials.append((subject_id, trial_num, 'time_yoked', time_yoked_change, trial_data))

print(f"Found {len(extreme_trials)} trials with >500% change:")
for subject_id, trial_num, change_type, change_val, trial_data in extreme_trials[:5]:  # Show first 5
    print(f"Subject {subject_id}, Trial {trial_num} ({trial_data['trial_type']}): {change_type} = {change_val:.1f}%")
    print(f"  abs_min: {trial_data.get('abs_min_val')}, abs_max: {trial_data.get('abs_max_val')}")
    if 'time_yoked' in change_type:
        ref_type = 'onset' if 'onset' in str(trial_data.keys()) else 'offset'
        print(f"  time_yoked_min: {trial_data.get(f'time_yoked_min_val_{ref_type}')}")
        print(f"  time_yoked_max: {trial_data.get(f'time_yoked_max_val_{ref_type}')}")
    print()

# Print all of the things for weird trials
subject = 8 
trial_num = 12  # or whichever trial is showing the weird result

print("=== DEBUGGING NORMALIZED PAIN CHANGE ===")

# Get the trial data
trial_data = structured_data[subject][trial_num]
print(f"Trial type: {trial_data['trial_type']}")

# Check all the extrema values
print(f"abs_min_val: {trial_data.get('abs_min_val')}")
print(f"abs_max_val: {trial_data.get('abs_max_val')}")
print(f"abs_peak_to_peak: {trial_data.get('abs_peak_to_peak')}")

# Check time-yoked values too
time_yoked_min = trial_data.get('time_yoked_min_val_onset')
time_yoked_max = trial_data.get('time_yoked_max_val_onset')
print(f"time_yoked_min_val_onset: {time_yoked_min}")
print(f"time_yoked_max_val_onset: {time_yoked_max}")

# Calculate what the function would return
print(f"\nNormalized pain change calculations:")
print(f"abs_normalized_pain_change: {trial_data.get('abs_normalized_pain_change')}")
print(f"time_yoked_normalized_pain_change: {trial_data.get('time_yoked_normalized_pain_change')}")

# Manual calculation for abs values
if trial_data.get('abs_max_val') and trial_data.get('abs_max_val') != 0:
    manual_calc = (trial_data['abs_max_val'] - trial_data['abs_min_val']) / trial_data['abs_max_val'] * 100
    print(f"Manual abs calculation: ({trial_data['abs_max_val']} - {trial_data['abs_min_val']}) / {trial_data['abs_max_val']} * 100 = {manual_calc}")

# Manual calculation for time-yoked values
if time_yoked_max and time_yoked_max != 0:
    manual_calc_ty = (time_yoked_max - time_yoked_min) / time_yoked_max * 100
    print(f"Manual time-yoked calculation: ({time_yoked_max} - {time_yoked_min}) / {time_yoked_max} * 100 = {manual_calc_ty}")

# %%
