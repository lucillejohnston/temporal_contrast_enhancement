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
from typing import Dict, List, Tuple, Any, Optional
from plotting_functions import *

# =========================================================
# CONFIGURATION
# =========================================================

# Path to the cleaned, aligned trial data
DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json'
OUTPUT_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_metrics.json'

temp_change_offset = 0.67 # seconds, changing 1C in 1.5C/s
window_size = 5
period_c_duration = 20
trial_definitions = {
    'inv': ['T2', 'T1', 'T2'],
    't2_hold': ['T2', 'T2', 'T2'],
    'offset': ['T1', 'T2', 'T1'],
    't1_hold': ['T1', 'T1', 'T1'],
    'stepdown': ['T2', 'T2', 'T1']
}
trial_type_info = {
    'inv':      {'kind': 'stepped', 'extrema_order': ['min', 'max'], 'reference': None},
    'offset':   {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
    'stepdown': {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
    't1_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': ['offset','stepdown']},
    't2_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': 'inv'}
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

def get_trial_ramp_time(trial_df, default_ramp_time=15.0):
    """Calculate ramp time for a single trial"""
    post_ramp = trial_df[trial_df['aligned_time'] > 5].copy()
    if len(post_ramp) < 10:
        return default_ramp_time
    
    temp_diff = post_ramp['temperature'].diff().rolling(3).mean().abs()
    stable_mask = temp_diff < 0.1
    
    if stable_mask.any():
        stable_time = post_ramp.loc[stable_mask, 'aligned_time'].iloc[0]
        return stable_time
    else:
        return default_ramp_time
    

# Filter calibration trials matching T1 hold temperature
import numpy as np
from scipy.spatial.distance import euclidean

def find_matching_calibration_trials(calibration_trials, t1_hold_trials, temp_tolerance=1.0):
    """
    Find calibration trials that match T1 hold temperature profiles.
    
    Parameters:
    -----------
    calibration_trials : list of DataFrames
        Calibration trial data
    t1_hold_trials : list of DataFrames
        T1 hold trial data
    temp_tolerance : float
        Maximum temperature difference (°C) to consider a match
    
    Returns:
    --------
    matched_calibration : list of DataFrames
        Calibration trials matching T1 hold temperature profiles
    """
    matched_calibration = []
    
    # Get T1 hold temperature range for each subject
    t1_hold_temps = {}
    for df in t1_hold_trials:
        subject = df['subject'].iloc[0]
        max_temp = df['temperature'].max()
        if subject not in t1_hold_temps:
            t1_hold_temps[subject] = []
        t1_hold_temps[subject].append(max_temp)
    
    # Average T1 hold temperature per subject
    t1_hold_avg_temps = {subj: np.mean(temps) for subj, temps in t1_hold_temps.items()}
    
    print("\n=== T1 HOLD TEMPERATURE TARGETS ===")
    for subject, temp in t1_hold_avg_temps.items():
        print(f"Subject {subject}: {temp:.1f}°C")
    
    # Filter calibration trials
    print("\n=== MATCHING CALIBRATION TRIALS ===")
    for df in calibration_trials:
        subject = df['subject'].iloc[0]
        trial_num = df['trial_num'].iloc[0] if 'trial_num' in df.columns else 'unknown'
        cal_max_temp = df['temperature'].max()
        
        if subject in t1_hold_avg_temps:
            target_temp = t1_hold_avg_temps[subject]
            temp_diff = abs(cal_max_temp - target_temp)
            
            if temp_diff <= temp_tolerance:
                matched_calibration.append(df)
                print(f"✓ Subject {subject}, Trial {trial_num}: {cal_max_temp:.1f}°C (diff: {temp_diff:.1f}°C)")
            else:
                print(f"✗ Subject {subject}, Trial {trial_num}: {cal_max_temp:.1f}°C (diff: {temp_diff:.1f}°C) - excluded")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total calibration trials: {len(calibration_trials)}")
    print(f"Matched calibration trials: {len(matched_calibration)}")
    
    return matched_calibration

# First, get the calibration and t1_hold trials
calibration_trials = time_temp_aligned_trial_type.get('calibration', [])
t1_hold_trials = time_temp_aligned_trial_type.get('t1_hold', [])

# Apply the filter
matched_calibration = find_matching_calibration_trials(
    calibration_trials, 
    t1_hold_trials, 
    temp_tolerance=0.1  # Within 1°C
)

# Time Window and AUC Functions
def define_timewindows(trial_type, trial_definitions, base_offset):
    """
    Define the start and end times for the three periods based on the trial's temperature settings.
    Args:
        trial_type (str): The trial type key.
        trial_definitions (dict): Mapping of trial_type to temperature sequence.
        base_offset (float): Time offset for the start of the trial.

    Returns:
        dict: Keys 'A', 'B', 'C' mapping to (start, end) tuples.
    """
    temperature_sequence = trial_definitions[trial_type]
    # Period A: 5 seconds after base_offset
    start_A = base_offset
    end_A = start_A + 5
    # Gap A→B (if the temperature changes)
    gap_A_B = temp_change_offset if temperature_sequence[0] != temperature_sequence[1] else 0
    # Period B: 5 seconds following gap_A_B
    start_B = end_A + gap_A_B
    end_B = start_B + 5
    # Gap B→C (if the temperature changes)
    gap_B_C = temp_change_offset if temperature_sequence[1] != temperature_sequence[2] else 0
    # Period C: 20 seconds following gap_B_C
    start_C = end_B + gap_B_C
    end_C = start_C + 20

    return {'A': (start_A, end_A),
            'B': (start_B, end_B),
            'C': (start_C, end_C)}

def compute_aucs(pain_series, trial_type, trial_definitions, base_offset):
    """
    Compute AUCs over the time windows defined by the trial's temperature settings.
    
    Parameters:
        series (pd.Series): The pain (or similar) time series with a datetime/numeric index.
        trial_type (str): The trial type, used to look up the period temperature settings.
        base_offset (float): The starting offset for the analysis.
    
    Returns:
        dict: A dictionary with AUCs for each period and the total AUC.
    """
    # Get boundaries for each period
    time_windows = define_timewindows(trial_type, trial_definitions, base_offset)
    aucs = {}
    auc_total = 0
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        window = pain_series.loc[(pain_series.index >= start) & (pain_series.index <= end)]
        if window.empty:
            auc = np.nan
        else:
            auc = np.trapezoid(window.values, x=window.index)
        aucs[f'auc_{period}'] = auc
        auc_total += auc if not np.isnan(auc) else 0
    aucs['auc_total'] = auc_total
    return aucs

def extract_extrema(
        pain_series: pd.Series,
        trial_type: str,
        mode: str = 'local',
        reference_times: dict = None,
        base_offset: float = 5.0,               # In preprocessing, added 5s buffer before trial start
    ) -> dict:
    """
    Extract extrema (min/max) and related metrics from a pain series.

    Parameters:
        pain_series (pd.Series): Pain ratings indexed by aligned_time.
        trial_type (str): The trial type key.
        mode (str): 'local' for local extrema, 'control' for extracting at reference times.
        reference_times (dict): Dict with 'min_time' and 'max_time' (used if mode='control').
        base_offset (float): Time offset for the start of the trial.
    Returns:
        dict: Extrema values, times, peak-to-peak, and latency.
    """
    if mode == 'local': # for offset, inv, stepdown trials 
        return _extract_local_extrema(pain_series, trial_type, base_offset)
    elif mode == 'control': # for t1_hold, t2_hold trials
        return _extract_control_extrema(pain_series, trial_type, reference_times, base_offset)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _extract_local_extrema(
        pain_series: pd.Series, 
        trial_type: str, 
        base_offset: float
        )-> Dict[str, float]:

        # Define time windows
        time_windows = define_timewindows(trial_type, trial_definitions, base_offset)
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

        if extrema_mode == ['min', 'max']:      # for 'inv' trials
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
        
        peak_to_peak = abs(second_val - first_val) if not (np.isnan(second_val) or np.isnan(first_val)) else np.nan
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
        base_offset: float
        ) -> Dict[str, float]:
        """Extract time-yoked and absolute extrema for control trials"""

        # Absolute maximum and subsequent minimum
        abs_max_time = pain_series.idxmax()
        abs_max_val = pain_series.max()

        # Get period C boundaries
        time_windows = define_timewindows(trial_type, trial_definitions, base_offset) 
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
        time_yoked_peak_to_peak = abs(time_yoked_max_val - time_yoked_min_val) if not (
            np.isnan(time_yoked_max_val) or np.isnan(time_yoked_min_val)) else np.nan
        time_yoked_peak_to_peak_latency = ref_max_time - ref_min_time if not (
            np.isnan(ref_max_time) or np.isnan(ref_min_time)) else np.nan

        # Peak to peak for absolute extrema
        abs_peak_to_peak = abs(abs_max_val - abs_min_val) if not (
            np.isnan(abs_max_val) or np.isnan(abs_min_val)) else np.nan
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
    elif tt == 'inv':
        # For inv trials: (max - min) / max * 100
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
        # t2_hold references 'inv'
        if use_time_yoked:
            min_val = row.get('time_yoked_min_val_inv')
            max_val = row.get('time_yoked_max_val_inv')
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

# Extract stepped trials first to store their extrema times
for (subject, trial_num), trial_df in df.groupby(['subject', 'trial_num']):
    trial_type = trial_df['trial_type'].iloc[0]
    if trial_type not in trial_definitions:
        continue # skip other trial types
    trial_df = trial_df.sort_values('aligned_time')
    pain_series = pd.Series(trial_df['pain'].values, index=trial_df['aligned_time'].values)
    info = trial_type_info[trial_type]
    if info['kind'] == 'control':
        continue # skip control trials for now
    trial_ramp_time = get_trial_ramp_time(trial_df)
    # Compute AUCs 
    aucs = compute_aucs(pain_series, trial_type, trial_definitions, base_offset=trial_ramp_time)
    time_windows = define_timewindows(trial_type, trial_definitions, base_offset=trial_ramp_time)
    time_window_record = {}
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        time_window_record[f'{period}_start'] = start
        time_window_record[f'{period}_end'] = end

    # Compute extrema
    extrema = extract_extrema(
        pain_series, trial_type, mode='local', base_offset=trial_ramp_time)
    # Store extrema times for later use with control trials
    subject_extrema_times[(int(subject), str(trial_type), int(trial_num))] = {
        'abs_min_time': extrema['abs_min_time'],
        'abs_max_time': extrema['abs_max_time']
    }
    record = {
    'subject': subject,
    'trial_num': trial_num,
    'trial_type': trial_type,
    **aucs,
    **extrema,        
    **time_window_record
    }
    trial_metrics_list.append(record)

# Now extract control trials using reference times from stepped trials
for (subject, trial_num), trial_df in df.groupby(['subject', 'trial_num']):
    trial_type = trial_df['trial_type'].iloc[0]
    if trial_type not in trial_definitions:
        continue # skip other trial types
    trial_df = trial_df.sort_values('aligned_time')
    pain_series = pd.Series(trial_df['pain'].values, index=trial_df['aligned_time'].values)
    info = trial_type_info[trial_type]
    if info['kind'] != 'control':
        continue # skip stepped trials, already processed

    # Compute AUCs 
    trial_ramp_time = get_trial_ramp_time(trial_df)
    aucs = compute_aucs(pain_series, trial_type, trial_definitions, base_offset=trial_ramp_time)
    time_windows = define_timewindows(trial_type, trial_definitions, base_offset=trial_ramp_time)
    time_window_record = {}
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        time_window_record[f'{period}_start'] = start
        time_window_record[f'{period}_end'] = end
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
                ref_trial_num = np.nan
                
            reference_times = subject_extrema_times.get(
                (int(subject), str(ref), int(ref_trial_num)), 
                {'abs_min_time': np.nan, 'abs_max_time': np.nan})
            extrema = extract_extrema(
                pain_series, trial_type, mode='control', 
                reference_times=reference_times, base_offset=trial_ramp_time)
            
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
            ref_trial_num = np.nan

        reference_times = subject_extrema_times.get(
            (int(subject), str(ref), int(ref_trial_num)), 
            {'abs_min_time': np.nan, 'abs_max_time': np.nan})
        
        extrema = extract_extrema(
            pain_series, trial_type, mode='control', 
            reference_times=reference_times, base_offset=trial_ramp_time)
        
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

# Convert to serializable format and organize by subject
structured_data = {}
for record in trial_metrics_list:
    subject = int(record['subject'])
    trial_num = int(record['trial_num'])
    
    if subject not in structured_data:
        structured_data[subject] = {}
    
    # Convert all values to JSON-serializable format
    clean_record = {}
    for key, value in record.items():
        clean_record[key] = convert_to_serializable(value)
    
    structured_data[subject][trial_num] = clean_record

# Save as JSON
with open(OUTPUT_PATH, 'w') as f:
    json.dump(structured_data, f, indent=2)

print(f"Trial metrics saved to {OUTPUT_PATH}")

# %%
# ========================================================
# TRIAL COMPARISON PLOTTING
# ========================================================

# Random selection (original behavior)
plot_trial_comparison(structured_data, df, 'inv', 't2_hold', 'inv')

# # Specific subject and specific control trial
# plot_trial_comparison(structured_data, df, 'inv', 't2_hold', 'inv', specific_subject=117, specific_control_trial=1)

# %%
