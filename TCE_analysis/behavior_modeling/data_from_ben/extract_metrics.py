#%%
"""
Section 1: Imports and Data Loading

Loads trial data from JSON into a pandas DataFrame.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Path to the cleaned, aligned trial data
DATA_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data_cleaned_aligned.json'

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
"""
Section 2: Metric Calculation Functions

Defines helper functions for extracting metrics from trial data.
"""
temp_change_offset = 0.67 # seconds, changing 1C in 1.5C/s
trial_definitions = {
    'inv': ['T2', 'T1', 'T2'],
    't2_hold': ['T2', 'T2', 'T2'],
    'offset': ['T1', 'T2', 'T1'],
    't1_hold': ['T1', 'T1', 'T1'],
    'stepdown': ['T2', 'T2', 'T1']
}

def define_timewindows(trial_type, trial_definitions, base_offset=0.0):
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

def compute_aucs(pain_series, trial_type, trial_definitions, base_offset=0.0):
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
        extrema_order_map: dict,
        trial_definitions: dict,
        mode: str = 'local',
        reference_times: dict = None,
        base_offset: float = 0.0,
        window_size: float = 5.0
) -> dict:
    """
    Extract extrema (min/max) and related metrics from a pain series.

    Args:
        pain_series (pd.Series): Pain ratings indexed by aligned_time.
        trial_type (str): The trial type key.
        extrema_order_map (dict): Mapping of trial_type to extrema order.
        trial_definitions (dict): Mapping of trial_type to temperature sequence.
        mode (str): 'local' for local extrema, 'control' for extracting at reference times.
        reference_times (dict): Dict with 'min_time' and 'max_time' (used if mode='control').
        base_offset (float): Time offset for the start of the trial.
        window_size (float): Window size (seconds) around transition (used if mode='local').

    Returns:
        dict: Extrema values, times, peak-to-peak, and latency.
    """
    if mode == 'local': # for offset, inv, stepdown trials 
        time_windows = define_timewindows(trial_type, trial_definitions, base_offset)
        transition_time = time_windows['C'][0]  # Use the start of period C
        window_start = transition_time - window_size
        window_end = transition_time + window_size
        window_series = pain_series.loc[(pain_series.index >= window_start) & (pain_series.index <= window_end)]
        if window_series.empty:
            return {x: np.nan for x in ['min_val', 'min_time', 'max_val', 'max_time', 'peak_to_peak', 'peak_to_peak_latency']}
        extrema_mode = extrema_order_map.get(trial_type)
        c_window_end = transition_time + 20  # 20 seconds after the transition time
        if extrema_mode == ['min', 'max']:
            first_val = window_series.min()
            first_time = window_series.idxmin()
            after_first = pain_series.loc[(pain_series.index > first_time) & (pain_series.index <= c_window_end)]
            second_val = after_first.max()
            second_time = after_first.idxmax()
            min_val, min_time = first_val, first_time
            max_val, max_time = second_val, second_time
        elif extrema_mode == ['max', 'min']:
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
            'min_val': min_val, 'min_time': min_time,
            'max_val': max_val, 'max_time': max_time,
            'peak_to_peak': peak_to_peak,
            'peak_to_peak_latency': peak_to_peak_latency
        }
    elif mode == 'control':
        if reference_times is None:
            raise ValueError("reference_times must be provided when mode='control'")
        min_time = reference_times.get('min_time', np.nan)
        max_time = reference_times.get('max_time', np.nan)
        if not pain_series.empty:
            # Find the nearest index for min_time and max_time
            idx_array = np.array(pain_series.index)
            if not np.isnan(min_time):
                min_idx = np.argmin(np.abs(idx_array - min_time))
                min_val = pain_series.iloc[min_idx]
            else:
                min_val = np.nan
            if not np.isnan(max_time):
                max_idx = np.argmin(np.abs(idx_array - max_time))
                max_val = pain_series.iloc[max_idx]
            else:
                max_val = np.nan
        else:
            min_val = np.nan
            max_val = np.nan
        peak_to_peak = abs(max_val - min_val) if not (np.isnan(max_val) or np.isnan(min_val)) else np.nan
        peak_to_peak_latency = max_time - min_time if not (np.isnan(max_time) or np.isnan(min_time)) else np.nan
        return {
            'min_val': min_val,
            'max_val': max_val,
            'peak_to_peak': peak_to_peak,
            'peak_to_peak_latency': peak_to_peak_latency
        }

#%%
"""
Section 3: Compute metrics for each trial and assemble into a DataFrame
"""
trial_type_info = {
    'inv':      {'kind': 'stepped', 'extrema_order': ['min', 'max'], 'reference': None},
    'offset':   {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
    'stepdown': {'kind': 'stepped', 'extrema_order': ['max', 'min'], 'reference': None},
    't1_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': ['offset','stepdown']},
    't2_hold':  {'kind': 'control', 'extrema_order': ['control'],    'reference': 'inv'}
}

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
    extrema_order_map = {k: v['extrema_order'] for k, v in trial_type_info.items()}
    # Compute AUCs 
    aucs = compute_aucs(pain_series, trial_type, trial_definitions)
    time_windows = define_timewindows(trial_type, trial_definitions)
    time_window_record = {}
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        time_window_record[f'{period}_start'] = start
        time_window_record[f'{period}_end'] = end

    # Compute extrema
    extrema = extract_extrema(
        pain_series, trial_type, extrema_order_map, trial_definitions, mode='local')
    # Store extrema times for later use with control trials
    subject_extrema_times[(int(subject), str(trial_type), int(trial_num))] = {
        'min_time': extrema['min_time'],
        'max_time': extrema['max_time']
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

    extrema_order_map = {k: v['extrema_order'] for k, v in trial_type_info.items()}
    # Compute AUCs 
    aucs = compute_aucs(pain_series, trial_type, trial_definitions)
    time_windows = define_timewindows(trial_type, trial_definitions)
    time_window_record = {}
    for period in ['A', 'B', 'C']:
        start, end = time_windows[period]
        time_window_record[f'{period}_start'] = start
        time_window_record[f'{period}_end'] = end
    # Get reference times from the corresponding stepped trial
    references = info['reference']
    if isinstance(references, list):
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
            reference_times = subject_extrema_times.get((int(subject), str(ref), int(ref_trial_num)), {'min_time': np.nan, 'max_time': np.nan})
            extrema = extract_extrema(
                pain_series, trial_type, extrema_order_map, trial_definitions, mode='control', reference_times=reference_times)
            for k, v in extrema.items():
                extrema_dict[f"{k}_{ref}"] = v
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
    else:
        # For t2_hold: just do the single reference
        ref = references
        ref_trials = df[(df['subject'] == subject) & (df['trial_type'] == ref)]
        if not ref_trials.empty:
            ref_trial_nums = ref_trials['trial_num'].unique()
            trial_num_diffs = np.abs(ref_trial_nums - trial_num)
            min_diff_idx = np.argmin(trial_num_diffs)
            ref_trial_num = ref_trial_nums[min_diff_idx]
        else:
            ref_trial_num = np.nan
        reference_times = subject_extrema_times.get((int(subject), str(ref), int(ref_trial_num)), {'min_time': np.nan, 'max_time': np.nan})
        extrema = extract_extrema(
            pain_series, trial_type, extrema_order_map, trial_definitions, mode='control', reference_times=reference_times)
        record = {
            'subject': int(subject),
            'trial_num': int(trial_num),
            'trial_type': trial_type,
            'reference_trial_num': ref_trial_num,
            **aucs,
            **extrema,        
            **time_window_record
        }
        trial_metrics_list.append(record)

def calc_normalized_pain_change(row):
    tt = row['trial_type']
    if tt in ['offset','stepdown']:
        min_val = row['min_val']
        max_val = row['max_val']
        if pd.notnull(min_val) and pd.notnull(max_val) and max_val != 0:
            return (min_val - max_val) / max_val * 100 
    elif tt == 't1_hold':
        min_val = row.get('min_val_offset', np.nan)
        max_val = row.get('max_val_offset', np.nan)
        if pd.notnull(min_val) and pd.notnull(max_val) and max_val != 0:
            return (min_val - max_val) / max_val * 100
    elif tt in ['inv', 't2_hold']:
        min_val = row['min_val']
        max_val = row['max_val']
        if pd.notnull(min_val) and pd.notnull(max_val) and (100 - min_val) != 0:
            return (max_val - min_val) / (100 - min_val) * 100
    return np.nan # if no recognized trial_type or missing data 

trial_metrics_df = pd.DataFrame(trial_metrics_list)
trial_metrics_df['normalized_pain_change'] = trial_metrics_df.apply(calc_normalized_pain_change, axis=1)
# Save the trial metrics DataFrame to a CSV file
OUTPUT_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_metrics.csv'
trial_metrics_df.to_csv(OUTPUT_PATH, index=False)
print(f"Trial metrics saved to {OUTPUT_PATH}")

#%%