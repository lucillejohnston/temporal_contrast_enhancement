#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
with open('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/TCE DATA/data_from_ben/trial_data.json') as f:
    data = json.load(f)
trial_data = pd.DataFrame(data)

"""
'subject' - subject ID
'trial_num' - trial number per subject/session
'trial_date' - date of the trial 
'trial_type' - type of trial (e.g., CLEAR-AC3, CLEAR-ABC1, etc.)
'temperature' - temperature of the thermode at each timepoint
'pain' - pain rating at each timepoint
'notes' - additional notes for the trial
'relative_time' - time relative to the start of the trial (in seconds)
'relative_time_str' - string representation of relative time (e.g., "0.00s", "1.00s")
'actual_time' - timestamp of each timepoint in clock time (trial_date + relative_time)
"""
trial_key = {
    'CLEAR calibration - pain threshold': 'calibration',
    'Offset 45-46-45': 'offset_AV_conditioning',
    'Offset TEST': 'offset_AV_test',
    'CLEAR calibration LIMITS': 'limits',
    'CLEAR-ABC1': 'offset',
    'CLEAR-AC2': 't2_hold',
    'CLEAR-AC3': 'inv',
    'CLEAR-B2': 't1_hold',
    'CLEAR-B3': 'stepdown'
}
trial_data['trial_type'] = trial_data['trial_type'].map(trial_key)
#%%
# Functions to process the data
def resample_trials(trial_df, freq='100ms'):
    """
    Downsample a single trial DataFrame to a specified frequency (default 10Hz).
    Both 'relative_time' and 'actual_time' are downsampled/interpolated.
    """
    # Ensure the DataFrame is sorted by the chosen time column and set it as the index
    trial_df = trial_df.sort_values('relative_time').copy()
    # Set index to relative_time as TimedeltaIndex for resampling
    trial_df['relative_time'] = pd.to_timedelta(trial_df['relative_time'])
    trial_df = trial_df.set_index('relative_time')

    # Resample the DataFrame at the given frequency using mean for numeric fields.
    df_resampled = trial_df.resample(freq).mean(numeric_only=True)
    
    # Interpolate actual_time (convert to datetime if needed)
    if 'actual_time' in trial_df.columns:
        trial_df['actual_time'] = pd.to_datetime(trial_df['actual_time'])
        trial_df = trial_df.sort_index()
        df_resampled['actual_time'] = trial_df['actual_time'].reindex(df_resampled.index, method='nearest').values
   
    # Add back non-numeric columns (like trial_type)
    for col in ['trial_type']:
        if col in trial_df.columns:
            df_resampled[col] = trial_df[col].dropna().iloc[0]
    
    # Reset index so relative_time is a column again (in seconds)
    df_resampled = df_resampled.reset_index()
    if np.issubdtype(df_resampled['relative_time'].dtype, np.timedelta64):
    # Convert Timedelta to float seconds using .dt.total_seconds()
        df_resampled['relative_time'] = df_resampled['relative_time'].dt.total_seconds()
    else:
    # Already float or int, just ensure float
        df_resampled['relative_time'] = df_resampled['relative_time'].astype(float)
    return df_resampled

def align_trials_time(timeseries_dict, smoothing=True, smoothing_window=10,
                      rise_threshold=1.0, baseline_window=1.0):
    """
    Aligns each trial in the given timeseries_dict in time.
    
    For each trial:
      - If smoothing is True, a running mean (with the specified smoothing_window) is computed 
        on temperature values; otherwise, raw temperature values are used.
      - The event time is determined based on the maximum temperature (or maximum smoothed 
        temperature, if smoothing is enabled).
      - The DataFrame's index is shifted so that the event time becomes zero.
      - For trials of type 'CLEAR-ABC1', an additional offset is added.
    
    Parameters:
        timeseries_dict (dict): Dictionary with keys (e.g., (subject, trial_num)) and values as DataFrames.
        smoothing (bool): Whether to apply smoothing on temperature values (default True).
        smoothing_window (int): Window size for smoothing (default 10).
        baseline_temp (float): Known baseline temperature (default 32)
        rise_threshold (float): Degrees above baseline to define rise onset (default 1.5)
        
    Returns:
        aligned_dict (dict): Dictionary of time-aligned DataFrames.
    """
    aligned_dict = {}
    for key, df in timeseries_dict.items():
        df = df.copy()
        if smoothing:
            # Smooth the temperature values using a rolling mean and fill any NaN values.
            df['temp'] = df['temperature'].rolling(window=smoothing_window, center=True).mean()
            df['temp'] = df['temp'].bfill().ffill()
        else:
            # If smoothing is not enabled, use the raw temperature values.
            df['temp'] = df['temperature']
        df = df.set_index('relative_time')
        trial_type_name = df['trial_type'].dropna().iloc[0]

        # Calculate baseline temperature from first baseline_window seconds
        first_time = df.index.min()
        baseline_mask = df.index <= (first_time + baseline_window)
        
        if baseline_mask.sum() < 2:
            # If not enough points in baseline window, use first 5 points
            baseline_temp = df['temp'].iloc[:5].mean()
            print(f"⚠️  Trial {key}: Only {baseline_mask.sum()} points in first {baseline_window}s, using first 5 points for baseline")
        else:
            baseline_temp = df.loc[baseline_mask, 'temp'].mean()

        # Find rise onset: first time temp exceeds baseline + rise_threshold
        target_temp = baseline_temp + rise_threshold
        event_time = None
        for t, temp in df['temp'].items():
            if temp >= target_temp:
                event_time = t - 5 # 5s before crossing threshold
                break

        # FALLBACK: If no rise found
        if event_time is None:
            # Strategy 1: Find steepest temperature increase
            temp_diff = df['temp'].diff()
            if temp_diff.max() > 0.1:  # If there's any noticeable increase
                event_time = temp_diff.idxmax()
                print(f"⚠️  Trial {key}: No rise above {target_temp:.1f}°C. Using steepest rise at t={event_time:.2f}s "
                      f"(baseline={baseline_temp:.1f}°C, max={df['temp'].max():.1f}°C)")
            else:
                # Strategy 2: Use 15s before max temperature
                max_temp_time = df['temp'].idxmax()
                event_time = max(df.index.min(), max_temp_time - 15.0)
                print(f"⚠️  Trial {key}: No clear rise. Using 15s before max at t={event_time:.2f}s "
                      f"(baseline={baseline_temp:.1f}°C, max={df['temp'].max():.1f}°C)")

        df_aligned = df.copy()
        df_aligned.index = df_aligned.index - event_time
        aligned_dict[key] = df_aligned

    return aligned_dict

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

#%% LOAD, DOWNSAMPLE, ALIGN TRIALS
# Align data in time and make another column aligned in temperature for plotting 
timeseries_dict = {}
for (subject, trial_num), df in trial_data.groupby(['subject', 'trial_num']):
    timeseries_dict[(subject, trial_num)] = df

# Downsample to 10Hz
timeseries_10hz_dict = {}
for key, df in timeseries_dict.items():
    df_10hz = resample_trials(df, freq='100ms')
    timeseries_10hz_dict[key] = df_10hz

# Align trials in time
aligned_dict = align_trials_time(timeseries_10hz_dict, smoothing=False, smoothing_window=10)


#%% CLEAN THE DATA
"""
Manually go through the notes to clean data
Also exclude any trial that never changes from 0 pain
"""
import re

# Base note patterns:
base_pattern_simple = re.compile(r'^[ABC][123]\s+(proximal|medial|distal)\s+m\d+$', re.IGNORECASE)
base_pattern_offset = re.compile(r'^offset\d+\s+[ABC][123]\s+(proximal|medial|distal)\s+m\d+$', re.IGNORECASE)

def has_additional_info(note):
    # Return True if the note doesn't match the basic patterns exactly.
    # You can adjust the logic if a note might start with these patterns yet have extra text.
    return not (base_pattern_simple.fullmatch(note) or base_pattern_offset.fullmatch(note))

# Go through each trial type and extract unique extra notes:
selected_trial_types = trial_data['trial_type'].unique()

for trial_type in selected_trial_types:
    # Filter rows for the trial type and non-null notes
    subset = trial_data[(trial_data['trial_type'] == trial_type) & (trial_data['notes'].notna())]
    # Filter to only those rows with extra notes
    extra_info = subset[subset['notes'].apply(has_additional_info)]
    # Drop duplicate rows based on subject, trial_num, and notes
    extra_info = extra_info[['subject', 'trial_num', 'notes']].drop_duplicates()
    
    print(f"Extra notes for trial type {trial_type}:")
    if not extra_info.empty:
        # Print subject_trialNum and the note for each row
        for row in extra_info.itertuples(index=False):
            print(f"{row.subject}_{row.trial_num}: {row.notes}")
    else:
        print("None found")
    print("\n" + "-"*40 + "\n")

"""
Manually looked through notes and selected relevant ones
Format subject_trialNum: note

Extra notes for trial type CLEAR-AC2:
154_2: A2 proximal m10 - subj requested the thermode be taken off because it was too hot

Extra notes for trial type CLEAR-ABC1:
96_3: offset2 A1 medial m8 m9 (use m9 - m8 was hit accidentally before trigger and m9 was hit a little after trigger)
115_3: A1 medial m8 - first preconditioning trial after lowering T1 to 46
135_2: A1 proximal m4 - first trial of preconditioning after lowering T1 to 46
165_9: C1 distal m12 (subj said forgot to move covas to 0 once pain stopped)
"""
# manually plot trials with weird notes
selected_trials = ['154_2', '96_3', '115_3', '135_2', '165_9']
for key, df in aligned_dict.items():
    key_str = f"{key[0]}_{key[1]}"
    if key_str in selected_trials:
        plt.figure(figsize=(10, 6))
        
        # Plot Temperature trace
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['temperature'], color='blue', label='Temperature')
        plt.xlabel("Time (sec)")
        plt.ylabel("Temperature")
        plt.title(f"Trial {key_str} - Temperature")
        plt.legend()
        plt.grid(True)
        
        # Plot Pain trace
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['pain'], color='red', label='Pain')
        plt.xlabel("Time (sec)")
        plt.ylabel("Pain")
        plt.title(f"Trial {key_str} - Pain")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Find trials where pain is always zero
zero_pain_trials = []
for key, df in aligned_dict.items():
    if df['pain'].nunique() == 1 and df['pain'].iloc[0] == 0:
        zero_pain_trials.append(key)

for key in zero_pain_trials:
    key_str = f"{key[0]}_{key[1]}"
    df = aligned_dict[key]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Temperature on left y-axis
    ax1.plot(df.index, df['temperature'], color='blue', label='Temperature')
    ax1.set_xlabel("Aligned Time (s)")
    ax1.set_ylabel("Temperature", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Pain on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['pain'], color='red', label='Pain')
    ax2.set_ylabel("Pain", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f"Pain always zero: {key_str}")
    fig.tight_layout()
    plt.show()

zero_pain_trial_strs = [f"{key[0]}_{key[1]}" for key in zero_pain_trials]
print(f"Trials with always zero pain: {zero_pain_trial_strs}")
"""
Until I make a decision about what to do for these trials, 
I'll just exclude them and continue as normal
"""

# Identify flat temperature trials (< 0.5°C range)
flat_temp_trials = []
for key, df in aligned_dict.items():
    temp_range = df['temperature'].max() - df['temperature'].min()
    if temp_range < 0.5:  # Less than 0.5°C change
        flat_temp_trials.append(key)

flat_temp_trial_strs = [f"{key[0]}_{key[1]}" for key in flat_temp_trials]
print(f"\n⚠️  Excluding {len(flat_temp_trial_strs)} flat temperature trials: {flat_temp_trial_strs}")

#%% Clean out bad trials, finalize dataframe, save to JSON
""" 
I manually plotted all of them and found an additional few wonky temperature curves that I'm cleaning out manually:
offset_AV_test trials:
154_25 - temp weirdly dropped
126_19, 126_20, 126_21 - ramp rate was too high?
111_14, 111_15, 111_16 - temp never did offset step
75_13, 75_14, 75_15 - temp never did offset step
69_13, 69_14, 69_15 - temp never did offset step
68_13, 68_14, 68_15 - temp never did offset step
67_15, 67_16, 67_17 - temp never did offset step
61_14, 61_15, 61_16 - temp never did offset step
48_15, 48_16, 48_17 - temp never did offset step
44_13, 44_14, 44_15 - temp never did offset step
25_14, 25_15, 25_16 - temp never did offset step
22_14, 22_15, 22_16 - temp never did offset step
20_14, 20_15 - temp never did offset step
18_14, 18_15, 18_16 - temp never did offset step
17_15, 17_16, 17_17 - temp never did offset step
16_13, 16_14, 16_15 - temp never did offset step
15_15, 15_16, 15_17 - temp never did offset step
14_13, 14_14, 14_15 - temp never did offset step
13_13, 13_14, 13_15 - temp never did offset step
12_16, 12_17, 12_18 - temp never did offset step
11_14, 11_15, 11_16 - temp never did offset step
10_14, 10_15, 10_16 - temp never did offset step
9_15, 9_16, 9_17 - temp never did offset step
8_13, 8_14, 8_15 - temp never did offset step
7_13 - temp weirdly dropped
6_13, 6_14, 6_15 - temp never did offset step

offset_AV_conditioning trials:
11_9 - doesn't really have a baseline period starts basically right when temp ramps up
31_3 - temp was flatlined at baseline
36_15 - seems like a false start temp stayed at baseline and whole trial was 11s
68_9 - doesn't realy have a baseline period starts basically right when temp ramps up
154_26 - temp droped off and trial ended
"""
selected_trials = ['154_25','7_13','154_2', '96_3', '115_3', '135_2', 
                   '165_9', '31_3', '36_15', '154_26'] + zero_pain_trial_strs + flat_temp_trials
# Exclude selected trials from aligned_dict
aligned_dict = {
    key: df for key, df in aligned_dict.items()
    if f"{key[0]}_{key[1]}" not in selected_trials
}
# Plot separate figures for each trial type:
# Group trials by trial_type for plotting
trial_type_groups = {}
for key, df in aligned_dict.items():
    # Get trial type from the dataframe
    trial_type = df['trial_type'].dropna().iloc[0]
    if trial_type not in trial_type_groups:
        trial_type_groups[trial_type] = []
    trial_type_groups[trial_type].append(df)

# Plot separate figures for each trial type
for trial_type, dfs in trial_type_groups.items():
    plt.figure(figsize=(10, 6))
    for df in dfs:
        plt.plot(df.index, df['temperature'], alpha=0.5)
    plt.xlabel("Time from Temperature Rise Onset (seconds)")
    plt.ylabel("Temperature (°C)")
    plt.xlim(-15, 60)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Rise Onset (t=0)')
    plt.title(f"Time-Aligned Temperature Curves: {trial_type}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Combine all cleaned aligned data into a single DataFrame
dfs_with_time = []
for key, df in aligned_dict.items():
    df = df.copy()
    df['aligned_time'] = df.index  # Save the aligned time as a column
    df['subject'] = key[0]
    df['trial_num'] = key[1]
    dfs_with_time.append(df)

cleaned_aligned_df = pd.concat(dfs_with_time, ignore_index=True)

#%% MATCH CALIBRATION TRIALS TO T1_HOLD TRIALS
# Group aligned trials by trial_type
aligned_by_type = {}
for key, df in aligned_dict.items():
    trial_type = df['trial_type'].dropna().iloc[0]
    if trial_type not in aligned_by_type:
        aligned_by_type[trial_type] = []
    aligned_by_type[trial_type].append(df)

# Get calibration and t1_hold trials
calibration_trials = aligned_by_type.get('calibration', [])
t1_hold_trials = aligned_by_type.get('t1_hold', [])

print(f"\n=== MATCHING CALIBRATION TO T1_HOLD ===")
print(f"Calibration trials: {len(calibration_trials)}")
print(f"T1_hold trials: {len(t1_hold_trials)}")

# Find matching calibration trials
matched_calibration = find_matching_calibration_trials(
    calibration_trials, 
    t1_hold_trials, 
    temp_tolerance=0.5
)

print(f"\n=== RESULTS ===")
print(f"Matched calibration trials: {len(matched_calibration)}")

# Add matched calibration trials to cleaned_aligned_df as t1_hold
if matched_calibration:
    new_rows = []
    for cal_df in matched_calibration:
        cal_copy = cal_df.copy()
        if cal_copy.index.name == 'aligned_time' or pd.api.types.is_numeric_dtype(cal_copy.index):
            cal_copy = cal_copy.reset_index()
            if 'index' in cal_copy.columns and 'aligned_time' not in cal_copy.columns:
                cal_copy.rename(columns={'index': 'aligned_time'}, inplace=True)
        
        # Make sure we have the necessary columns
        if 'aligned_time' not in cal_copy.columns:
            print(f"⚠️  Warning: aligned_time not found in calibration trial. Columns: {cal_copy.columns.tolist()}")
            continue
        
        # Change trial_type to t1_hold
        cal_copy['trial_type'] = 't1_hold'
        cal_copy['original_trial_type'] = 'calibration'
        
        new_rows.extend(cal_copy.to_dict('records'))
    
    # Create dataframe from new rows
    new_df = pd.DataFrame(new_rows)
    
    # Add original_trial_type to existing df
    if 'original_trial_type' not in cleaned_aligned_df.columns:
        cleaned_aligned_df['original_trial_type'] = cleaned_aligned_df['trial_type']
    
    # Concatenate
    cleaned_aligned_df = pd.concat([cleaned_aligned_df, new_df], ignore_index=True)
    
    print(f"\n✅ Added {len(new_rows)} rows from matched calibration trials as t1_hold")
    print(f"New total rows: {len(cleaned_aligned_df)}")
    print(f"\nT1_hold breakdown:")
    print(cleaned_aligned_df[cleaned_aligned_df['trial_type'] == 't1_hold']['original_trial_type'].value_counts())


#%%
columns_to_save = [
    'subject', 'trial_num', 'trial_type', 'temperature', 'pain',
    'aligned_time', 'original_trial_type'
]
cleaned_aligned_df = cleaned_aligned_df[[col for col in columns_to_save if col in cleaned_aligned_df.columns]]
print(cleaned_aligned_df.head())
print(cleaned_aligned_df.columns)
# Save the cleaned and aligned DataFrame to a JSON file
cleaned_aligned_df.to_json(
    '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json',
    orient='records',
    date_format='iso'
)
# Check a few trials for aligned_time coverage and uniqueness
for (subject, trial_num), trial_df in cleaned_aligned_df.groupby(['subject', 'trial_num']):
    print(f"Trial: subject={subject}, trial_num={trial_num}")
    print("aligned_time min:", trial_df['aligned_time'].min())
    print("aligned_time max:", trial_df['aligned_time'].max())
    print("Number of unique aligned_time values:", trial_df['aligned_time'].nunique())
    print("First few aligned_time values:", trial_df['aligned_time'].head())
    print()
    # Only check a few trials
    if trial_num > 3:
        break

# %%
# Preprocess data: cut trials at -5s to 60s and resample at 5Hz
# This makes each trial shorter for faster paramater optimization and model fitting
from scipy.interpolate import interp1d
CUTOFF_TIME = 60.0  # seconds
BASELINE_START = -5.0 # seconds
TARGET_SAMPLE_RATE = 5.0 # Hz - downsampling for optimization speed
preprocessed_data = []

for subject in cleaned_aligned_df['subject'].unique():
    subject_data = cleaned_aligned_df[cleaned_aligned_df['subject'] == subject]
    
    for trial_num in subject_data['trial_num'].unique():
        trial_data = subject_data[subject_data['trial_num'] == trial_num].copy()
        
        # Clean and sort
        clean_data = trial_data.dropna(subset=['aligned_time', 'temperature', 'pain']).copy()
        clean_data = clean_data.sort_values('aligned_time').reset_index(drop=True)
        
        if len(clean_data) < 2:
            continue
        
        # Cut at CUTOFF_TIME
        clean_data = clean_data[(clean_data['aligned_time'] >= BASELINE_START) & 
                                (clean_data['aligned_time'] <= CUTOFF_TIME)].copy()
        
        if len(clean_data) < 2:
            continue
        
        # Resample at 5Hz
        time_min = clean_data['aligned_time'].min()
        time_max = clean_data['aligned_time'].max()
        
        # Create new time grid at 5Hz
        new_time = np.arange(time_min, time_max, 1.0 / TARGET_SAMPLE_RATE)
        
        if len(new_time) < 2:
            continue
        
        # Interpolate temperature and pain onto new time grid
        temp_interp = interp1d(clean_data['aligned_time'].values, 
                              clean_data['temperature'].values,
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        
        pain_interp = interp1d(clean_data['aligned_time'].values,
                              clean_data['pain'].values, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Create resampled dataframe
        resampled_trial = pd.DataFrame({
            'subject': subject,
            'trial_num': trial_num,
            'aligned_time': new_time,
            'temperature': temp_interp(new_time),
            'pain': pain_interp(new_time)
        })
        
        preprocessed_data.append(resampled_trial)

# Combine all preprocessed trials
data_df_preprocessed = pd.concat(preprocessed_data, ignore_index=True)

print(f"✅ Preprocessing complete:")
print(f"   Original data: {len(cleaned_aligned_df)} rows")
print(f"   Preprocessed data: {len(data_df_preprocessed)} rows")
print(f"   Subjects: {len(data_df_preprocessed['subject'].unique())}")
print(f"   Average points per trial: {len(data_df_preprocessed) / len(data_df_preprocessed.groupby(['subject', 'trial_num'])):.1f}")

# Save preprocessed data to JSON
data_df_preprocessed.to_json(
    '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_trimmed_downsampled.json',
    orient='records',
    date_format='iso'
)

# %%
