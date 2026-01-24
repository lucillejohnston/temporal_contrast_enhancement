#%%
# =============================================================
# Load data and packages
# =============================================================
import json, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.ticker as ticker
import plotting_functions as pf
with open('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

#%%
# =============================================================
# Functions to align trials by temperature periods
# =============================================================

temp_change_offset = 0.67 # seconds 
def define_timewindows(trial_type, trial_definitions, base_offset=0.0):
    """
    Define the start and end times for the three periods based on the trial's temperature settings.
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

time_temp_aligned_trial_type = {}

for trial_type in df['trial_type'].unique():
    trial_type_df = df[df['trial_type'] == trial_type]
    aligned_trials = []
    
    # Group by subject and trial_num
    for (subject, trial_num), trial_df in trial_type_df.groupby(['subject', 'trial_num']):
        # Skip if too short
        if len(trial_df) < 10:
            continue
        
        # Set aligned_time as index if it exists, otherwise use relative_time
        if 'aligned_time' in trial_df.columns:
            trial_df = trial_df.set_index('aligned_time').sort_index()
        elif 'relative_time' in trial_df.columns:
            trial_df = trial_df.set_index('relative_time').sort_index()
        else:
            continue
        
        aligned_trials.append(trial_df)
    
    time_temp_aligned_trial_type[trial_type] = aligned_trials
    print(f"Loaded {len(aligned_trials)} trials for {trial_type}")

def align_temperature_for_plot(df):
    """
    Adjust temperature for plotting
    """


#%% 
# =============================================================
# Plot comparisons between trial types (time series)
# =============================================================

# Define the trial pairs to compare
trial_pairs = [
    ['inv', 't2_hold'],
    ['offset', 't1_hold'],
    ['stepdown', 't1_hold'],
    ['offset', 'inv']
]

for pair in trial_pairs:
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 3]}
    )
    color_map = cm.get_cmap('tab10')
    color_dict = {k: color_map(i) for i, k in enumerate(pair)}

    # Top: Plot temperature traces
    for trial_type in pair:
        if trial_type not in time_temp_aligned_trial_type:
            continue
        dfs = time_temp_aligned_trial_type[trial_type]
        for df in dfs:
            subj = df['subject'].iloc[0]
            # Optionally offset t1_hold
            temp = df['temperature_aligned_for_plot'] - 1 if trial_type == 't1_hold' else df['temperature_aligned_for_plot']
            # axes[0].plot(df.index, temp, alpha=0.3, color=color_dict[trial_type])
        # Plot mean curve
        all_curves = []
        all_times = []
        for df in dfs:
            temp = df['temperature_aligned_for_plot'] - 1 if trial_type == 't1_hold' else df['temperature_aligned_for_plot']
            all_curves.append(temp.values)
            all_times.append(df.index.values)
        # Concatenate and compute mean at each unique time
        all_times_flat = np.concatenate(all_times)
        all_curves_flat = np.concatenate(all_curves)
        mean_df = pd.DataFrame({'time': all_times_flat, 'temp': all_curves_flat})
        mean_curve = mean_df.groupby('time').mean().sort_index()
        axes[0].plot(mean_curve.index, mean_curve['temp'], label=trial_type, color=color_dict[trial_type], linewidth=2)
    axes[0].set_ylabel("Temperature")
    axes[0].set_xlim(-5, 30)
    axes[0].set_ylim(-1.5, 1)
    axes[0].set_title(f"Avg Aligned Temperature: {pair[0]} vs {pair[1]}")
    axes[0].legend()
    axes[0].grid(True)

    # Bottom: Plot pain traces
    for trial_type in pair:
        if trial_type not in time_temp_aligned_trial_type:
            continue
        dfs = time_temp_aligned_trial_type[trial_type]
        # for df in dfs:
            # axes[1].plot(df.index, df['pain'], alpha=0.3, color=color_dict[trial_type])
        # Plot mean curve
        time_grid = np.arange(-15, 40, 0.1)
        all_interp_curves = []
        for df in dfs:
            interp_curve = np.interp(time_grid, df.index.values, df['pain'].values)
            all_interp_curves.append(interp_curve)
        all_interp_curves = np.array(all_interp_curves)
        mean_curve = np.mean(all_interp_curves, axis=0)
        sem_curve = np.std(all_interp_curves, axis=0, ddof=1) / np.sqrt(all_interp_curves.shape[0])
        axes[1].plot(time_grid, mean_curve, label=trial_type, color=color_dict[trial_type], linewidth=2)
        axes[1].fill_between(time_grid, mean_curve - 1.96*sem_curve, mean_curve + 1.96*sem_curve, color=color_dict[trial_type], alpha=0.2)
    axes[1].set_xlabel("Aligned Time (s)")
    axes[1].set_ylabel("Pain")
    axes[1].set_xlim(-5, 30)
    axes[1].set_ylim(10, 70)
    axes[1].set_title(f"Average Pain Curves: {pair[0]} vs {pair[1]}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %% 
# ============================================================
# Try adding in calibration trials that match t1_hold trials
# ============================================================

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
# Create a modified version of time_temp_aligned_trial_type with combined t1_hold
time_temp_aligned_with_calibration = time_temp_aligned_trial_type.copy()
time_temp_aligned_with_calibration['t1_hold_combined'] = t1_hold_trials + matched_calibration

print(f"\n=== COMBINED T1_HOLD TRIALS ===")
print(f"Original t1_hold trials: {len(t1_hold_trials)}")
print(f"Matched calibration trials: {len(matched_calibration)}")
print(f"Combined total: {len(time_temp_aligned_with_calibration['t1_hold_combined'])}")

# Define the trial pairs to compare (using combined t1_hold)
trial_pairs = [
    ['inv', 't2_hold'],
    ['offset', 't1_hold_combined'],
    ['stepdown', 't1_hold_combined'],
    ['offset', 'inv']
]

for pair in trial_pairs:
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': [1, 3]}
    )
    color_map = plt.get_cmap('tab10')
    color_dict = {k: color_map(i) for i, k in enumerate(pair)}

    # Top: Plot temperature traces
    for trial_type in pair:
        if trial_type not in time_temp_aligned_with_calibration:
            continue
        dfs = time_temp_aligned_with_calibration[trial_type]
        
        # Plot mean curve
        all_curves = []
        all_times = []
        for df in dfs:
            # Optionally offset t1_hold
            if trial_type == 't1_hold_combined':
                temp = df['temperature_aligned_for_plot'] - 1
                display_label = 't1_hold'
            else:
                temp = df['temperature_aligned_for_plot']
                display_label = trial_type
            all_curves.append(temp.values)
            all_times.append(df.index.values)
        
        # Concatenate and compute mean at each unique time
        all_times_flat = np.concatenate(all_times)
        all_curves_flat = np.concatenate(all_curves)
        mean_df = pd.DataFrame({'time': all_times_flat, 'temp': all_curves_flat})
        mean_curve = mean_df.groupby('time').mean().sort_index()
        
        label = display_label if trial_type == 't1_hold_combined' else trial_type
        axes[0].plot(mean_curve.index, mean_curve['temp'], label=label, 
                    color=color_dict[trial_type], linewidth=2)
    
    axes[0].set_ylabel("Temperature")
    axes[0].set_xlim(-5, 30)
    axes[0].set_ylim(-1.5, 0.5)
    pair_display = [p if p != 't1_hold_combined' else 't1_hold' for p in pair]
    axes[0].set_title(f"Avg Aligned Temperature: {pair_display[0]} vs {pair_display[1]}")
    axes[0].legend()
    axes[0].grid(True)

    # Bottom: Plot pain traces
    for trial_type in pair:
        if trial_type not in time_temp_aligned_with_calibration:
            continue
        dfs = time_temp_aligned_with_calibration[trial_type]
        
        # Plot mean curve with better handling of missing data
        time_grid = np.arange(-15, 40, 0.1)
        all_interp_curves = []
        
        for df in dfs:
            # Only interpolate within the valid time range of each trial
            valid_mask = (time_grid >= df.index.min()) & (time_grid <= df.index.max())
            interp_curve = np.full(len(time_grid), np.nan)
            
            if valid_mask.any() and len(df['pain']) > 0:
                # Remove NaN values before interpolation
                valid_data = df['pain'].dropna()
                if len(valid_data) > 1:
                    interp_curve[valid_mask] = np.interp(
                        time_grid[valid_mask], 
                        valid_data.index.values, 
                        valid_data.values,
                        left=np.nan,
                        right=np.nan
                    )
            all_interp_curves.append(interp_curve)
        
        if len(all_interp_curves) > 0:
            all_interp_curves = np.array(all_interp_curves)
            
            # Compute mean and SEM only where we have data
            with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                mean_curve = np.nanmean(all_interp_curves, axis=0)
                n_valid = np.sum(~np.isnan(all_interp_curves), axis=0)
                sem_curve = np.nanstd(all_interp_curves, axis=0, ddof=1) / np.sqrt(n_valid)
                
                # Only plot where we have at least 3 trials contributing
                valid_points = n_valid >= 3
                
                label = 't1_hold' if trial_type == 't1_hold_combined' else trial_type
                axes[1].plot(time_grid[valid_points], mean_curve[valid_points], 
                           label=f"{label} (n={len(dfs)})", 
                           color=color_dict[trial_type], linewidth=2)
                axes[1].fill_between(
                    time_grid[valid_points], 
                    (mean_curve - 1.96*sem_curve)[valid_points], 
                    (mean_curve + 1.96*sem_curve)[valid_points], 
                    color=color_dict[trial_type], alpha=0.2
                )
    
    axes[1].set_xlabel("Aligned Time (s)")
    axes[1].set_ylabel("Pain")
    axes[1].set_xlim(-5, 30)
    axes[1].set_ylim(0, 100)
    axes[1].set_title(f"Average Pain Curves: {pair_display[0]} vs {pair_display[1]}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


#%% 
# ============================================================
# Create a 2x2 grid of comparison plots
# ============================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.25, wspace=0.25)

color_map = plt.get_cmap('tab10')

for comp in trial_pairs:
    pair = comp['pair']
    col = comp['col']
    
    # Create axes for this column
    ax_temp = fig.add_subplot(gs[0, col])
    ax_pain = fig.add_subplot(gs[1, col])
    
    color_dict = {k: color_map(i) for i, k in enumerate(pair)}
    
    # Top: Plot temperature traces
    for trial_type in pair:
        if trial_type not in time_temp_aligned_with_calibration:
            continue
        dfs = time_temp_aligned_with_calibration[trial_type]
        
        # Plot mean curve
        all_curves = []
        all_times = []
        for df in dfs:
            # Optionally offset t1_hold
            if trial_type == 't1_hold_combined':
                temp = df['temperature_aligned_for_plot'] - 1
                display_label = 't1_hold'
            else:
                temp = df['temperature_aligned_for_plot']
                display_label = trial_type
            all_curves.append(temp.values)
            all_times.append(df.index.values)
        
        # Concatenate and compute mean at each unique time
        all_times_flat = np.concatenate(all_times)
        all_curves_flat = np.concatenate(all_curves)
        mean_df = pd.DataFrame({'time': all_times_flat, 'temp': all_curves_flat})
        mean_curve = mean_df.groupby('time').mean().sort_index()
        
        label = display_label if trial_type == 't1_hold_combined' else trial_type
        ax_temp.plot(mean_curve.index, mean_curve['temp'], label=label, 
                    color=color_dict[trial_type], linewidth=2)
    
    ax_temp.set_ylabel("Temperature (°C)", fontsize=11)
    ax_temp.set_xlim(-5, 30)
    ax_temp.set_ylim(-1.5, 0.5)
    pair_display = [p if p != 't1_hold_combined' else 't1_hold' for p in pair]
    ax_temp.set_title(comp['title'], fontsize=13, fontweight='bold')
    ax_temp.legend(fontsize=10)
    ax_temp.grid(True, alpha=0.3)
    
    # Bottom: Plot pain traces
    for trial_type in pair:
        if trial_type not in time_temp_aligned_with_calibration:
            continue
        dfs = time_temp_aligned_with_calibration[trial_type]
        
        # Plot mean curve with better handling of missing data
        time_grid = np.arange(-15, 40, 0.1)
        all_interp_curves = []
        
        for df in dfs:
            # Only interpolate within the valid time range of each trial
            valid_mask = (time_grid >= df.index.min()) & (time_grid <= df.index.max())
            interp_curve = np.full(len(time_grid), np.nan)
            
            if valid_mask.any() and len(df['pain']) > 0:
                # Remove NaN values before interpolation
                valid_data = df['pain'].dropna()
                if len(valid_data) > 1:
                    interp_curve[valid_mask] = np.interp(
                        time_grid[valid_mask], 
                        valid_data.index.values, 
                        valid_data.values,
                        left=np.nan,
                        right=np.nan
                    )
            all_interp_curves.append(interp_curve)
        
        if len(all_interp_curves) > 0:
            all_interp_curves = np.array(all_interp_curves)
            
            # Compute mean and SEM only where we have data
            with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                mean_curve = np.nanmean(all_interp_curves, axis=0)
                n_valid = np.sum(~np.isnan(all_interp_curves), axis=0)
                sem_curve = np.nanstd(all_interp_curves, axis=0, ddof=1) / np.sqrt(n_valid)
                
                # Only plot where we have at least 3 trials contributing
                valid_points = n_valid >= 3
                
                label = 't1_hold' if trial_type == 't1_hold_combined' else trial_type
                ax_pain.plot(time_grid[valid_points], mean_curve[valid_points], 
                           label=f"{label} (n={len(dfs)})", 
                           color=color_dict[trial_type], linewidth=2)
                ax_pain.fill_between(
                    time_grid[valid_points], 
                    (mean_curve - 1.96*sem_curve)[valid_points], 
                    (mean_curve + 1.96*sem_curve)[valid_points], 
                    color=color_dict[trial_type], alpha=0.2
                )
    
    ax_pain.set_xlabel("Aligned Time (s)", fontsize=11)
    ax_pain.set_ylabel("Pain Rating", fontsize=11)
    ax_pain.set_xlim(-5, 30)
    ax_pain.set_ylim(0, 80)
    ax_pain.legend(fontsize=10)
    ax_pain.grid(True, alpha=0.3)

# Add overall title
fig.suptitle('Temperature Contrast Effects on Pain', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# %%