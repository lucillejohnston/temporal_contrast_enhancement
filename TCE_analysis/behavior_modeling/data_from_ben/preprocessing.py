#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
with open('/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data.json') as f:
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

# Example usage:
# df_single = trial_data[trial_data['trial_num'] == 1]  # or however you filter a single trial
# df_resampled = resample_trial(df_single, time_col='actual_time')  # or use 'relative_time'
# print(df_resampled.head())


def align_trials_time(timeseries_dict, smoothing=True, smoothing_window=10):
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
        
    Returns:
        aligned_dict (dict): Dictionary of time-aligned DataFrames.
    """
    aligned_dict = {}
    for key, df in timeseries_dict.items():
        df = df.copy()
        if smoothing:
            # Smooth the temperature values using a rolling mean and fill any NaN values.
            df['temp'] = df['temperature'].rolling(window=smoothing_window, center=True).mean()
            df['temp'] = df['temp_smoothed'].bfill().ffill()
        else:
            # If smoothing is not enabled, use the raw temperature values.
            df['temp'] = df['temperature']
        df = df.set_index('relative_time')
        trial_key = df['trial_type'].dropna().iloc[0]
        alignment_value = df['temp'].max()
        event_time = None
        # Determine event_time based on trial type.
        if trial_key == 'CLEAR-AC3':
            tol = 0.15 # Increase tolerance for AC3 trials
            candidate_times = []
            for t, temp in df['temp'].items():
                if np.isclose(temp, alignment_value, atol=tol):
                    candidate_times.append(t)
            event_time = min(candidate_times) if candidate_times else df.index[0]
        else:
            tol = 0.05
            for t, temp in df['temp'].items():
                if np.isclose(temp, alignment_value, atol=tol):
                    event_time = t
                    break
            if event_time is None:
                event_time = df.index[0]
        # Create a copy of the DataFrame and shift the index to align the event time to zero.
        df_aligned = df.copy()
        df_aligned.index = df_aligned.index - event_time
        if trial_key == 'CLEAR-ABC1': # Add an additional offset for ABC1 trials
            df_aligned.index = df_aligned.index + 5.67  # 5s hold + 0.67s ramp
        aligned_dict[key] = df_aligned
    return aligned_dict

# Example usage:
# aligned_data = align_trials_time(timeseries_dict, smoothing=True, smoothing_window=10)

def align_trials_temperature(aligned_dict):
    """
    Align the trials in temperature (subtract max temp for each trial).
    Groups the aligned data by trial type.

    Args:
        aligned_dict (dict): Dict of {key: DataFrame} with time-aligned trials.

    Returns:
        aligned_temp_dict (dict): Dict of {key: DataFrame} with 'temperature_aligned_for_plot' column.
        time_temp_aligned_trial_type (dict): Dict of {trial_type: [DataFrame, ...]}.
    """
    aligned_temp_dict = {}
    for key, df in aligned_dict.items():
        max_temp = df['temperature'].max()
        df_temp_aligned = df.copy()
        df_temp_aligned['temperature_aligned_for_plot'] = df_temp_aligned['temperature'] - max_temp
        df_temp_aligned['trial_type'] = df_temp_aligned['trial_type'].map(trial_key).fillna(df_temp_aligned['trial_type'])
        aligned_temp_dict[key] = df_temp_aligned

    time_temp_aligned_trial_type = {}
    for key, df in aligned_temp_dict.items():
        trial_key_mapped = df['trial_type'].dropna().iloc[0]
        time_temp_aligned_trial_type.setdefault(trial_key_mapped, []).append(df)

    return aligned_temp_dict, time_temp_aligned_trial_type


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

# Align trials in temperature
aligned_temp_dict, time_temp_aligned_trial_type = align_trials_temperature(aligned_dict)

# # Plot separate figures for each trial type:
# for trial_key in time_temp_aligned_trial_type:
#     plt.figure(figsize=(10, 6))
#     for df in time_temp_aligned_trial_type[trial_key]:
#         plt.plot(df.index, df['temperature_aligned_for_plot'], alpha=0.5)
#     plt.xlabel("Time (seconds, aligned in time)")
#     plt.xlim(-15, 40)
#     plt.gca().get_yaxis().set_visible(False)  # Hide y-axis because irrelevant
#     plt.title(f"Aligned (by time and temp) Curves for Trial Type: {trial_key}")
#     plt.grid(True)
#     plt.show()



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
for key, df in aligned_temp_dict.items():
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
for key, df in aligned_temp_dict.items():
    if df['pain'].nunique() == 1 and df['pain'].iloc[0] == 0:
        zero_pain_trials.append(key)

for key in zero_pain_trials:
    key_str = f"{key[0]}_{key[1]}"
    df = aligned_temp_dict[key]
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
#%% Clean out bad trials, finalize dataframe, save to JSON
selected_trials = ['154_2', '96_3', '115_3', '135_2', '165_9'] + zero_pain_trial_strs
# Exclude selected trials from aligned_temp_dict
aligned_temp_dict = {
    key: df for key, df in aligned_temp_dict.items()
    if f"{key[0]}_{key[1]}" not in selected_trials
}

dfs_with_time = []
for key, df in aligned_temp_dict.items():
    df = df.copy()
    df['aligned_time'] = df.index  # Save the aligned time as a column
    df['time_from_zero'] = df['aligned_time'] - df['aligned_time'].min()  # Recalculate time_from_zero
    df['subject'] = key[0]
    df['trial_num'] = key[1]
    dfs_with_time.append(df)

cleaned_aligned_df = pd.concat(dfs_with_time, ignore_index=True)
cleaned_aligned_df['trial_type'] = cleaned_aligned_df['trial_type'].map(trial_key).fillna(cleaned_aligned_df['trial_type']) # rename trial types

columns_to_save = [
    'subject', 'trial_num', 'trial_type', 'temperature', 'pain',
    'aligned_time','temperature_aligned_for_plot','time_from_zero'
]
cleaned_aligned_df = cleaned_aligned_df[[col for col in columns_to_save if col in cleaned_aligned_df.columns]]
print(cleaned_aligned_df.head())
print(cleaned_aligned_df.columns)
# Save the cleaned and aligned DataFrame to a JSON file
cleaned_aligned_df.to_json(
    '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data_cleaned_aligned.json',
    orient='records',
    date_format='iso'
)
# # Check a few trials for aligned_time coverage and uniqueness
# for (subject, trial_num), trial_df in cleaned_aligned_df.groupby(['subject', 'trial_num']):
#     print(f"Trial: subject={subject}, trial_num={trial_num}")
#     print("aligned_time min:", trial_df['aligned_time'].min())
#     print("aligned_time max:", trial_df['aligned_time'].max())
#     print("Number of unique aligned_time values:", trial_df['aligned_time'].nunique())
#     print("First few aligned_time values:", trial_df['aligned_time'].head())
#     print()
#     # Only check a few trials
#     if trial_num > 3:
#         break

# %%

