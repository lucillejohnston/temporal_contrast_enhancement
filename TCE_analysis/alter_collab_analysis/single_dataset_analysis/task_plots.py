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
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/alter_collab_analysis/')
import utils.plotting_functions as pf
dataset = 'kneeOA' # options: 'plosONE', 'kneeOA'
with open(f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_data_cleaned_aligned.json') as f:
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

def align_temperature_for_plot(df, subject_t2_temps):
    """
    Adjust temperature for plotting
    Normalized to the subject's T2 temperature (from offset trials)
    """
    df = df.copy()
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    subject = df['subject'].iloc[0]
    # Try multiple key formats to handle int/float/string mismatches
    subject_key = None
    for key_format in [int(float(subject)), float(subject), str(subject), subject]:
        if key_format in subject_t2_temps:
            subject_key = key_format
            break
    
    if subject_key is not None:
        t2_temp = subject_t2_temps[subject_key]
        df['temperature_aligned_for_plot'] = df['temperature'] - t2_temp
    else:
        print(f"Warning: T2 temperature not found for subject {subject} (type: {type(subject)}). Keys sample: {list(subject_t2_temps.keys())[:3]}")
        df['temperature_aligned_for_plot'] = df['temperature']
    return df

#%% 
subject_t2_temps = {}
# Get all trial types that contain 'offset' (case-insensitive) or are specifically 'offset'
offset_trial_types = [trial_type for trial_type in df['trial_type'].unique() 
                      if 'offset' in str(trial_type).lower()]
print(f"Offset trial types found: {offset_trial_types}")

for subject in df['subject'].unique():
    subject_temps = []
    
    # Look through all offset trial types for this subject
    for trial_type in offset_trial_types:
        subject_trial_data = df[(df['subject'] == subject) & (df['trial_type'] == trial_type)]
        if not subject_trial_data.empty:
            temps = pd.to_numeric(subject_trial_data['temperature'], errors='coerce').dropna()
            if not temps.empty:
                max_temp = temps.max()
                subject_temps.append(max_temp)
    
    # Take the overall max across all offset trials for this subject
    if subject_temps:
        subject_t2_temps[int(subject)] = float(np.max(subject_temps))

print(f"Got T2 temperatures for {len(subject_t2_temps)} subjects")
print("Sample T2 baselines:", dict(list(subject_t2_temps.items())[:5]))

# Apply the alignment using T2 temperatures
for trial_type in time_temp_aligned_trial_type:
    for i, trial_df in enumerate(time_temp_aligned_trial_type[trial_type]):
        time_temp_aligned_trial_type[trial_type][i] = align_temperature_for_plot(trial_df, subject_t2_temps)

print(f"\nTemperature alignment complete using T2 from offset trials!")
#%%
# =============================================================
# Plot comparisons between trial types (time series)
# =============================================================

# Define the trial pairs to compare
if dataset == 'plosONE':
    trial_pairs = [
        ['inv', 't2_hold'],
        ['offset', 't1_hold'],
        ['stepdown', 't1_hold'],
        ['offset', 'inv']
    ]
elif dataset == 'kneeOA':
    trial_pairs = [
        ['onset', 't2_hold'],
        ['offset', 't1_hold'],
        ['offset','onset'],
        ['innocuous','t1_hold'],
        ['innocuous','t2_hold']
    ]
elif dataset == 'cLBP':
    trial_pairs = [
        ['onset', 't2_hold'],
        ['offset', 't1_hold'],
        ['offset','onset'],
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

        for trial_df in dfs:
            subj = trial_df['subject'].iloc[0]
            # Optionally offset t1_hold
            temp = trial_df['temperature_aligned_for_plot']
        # Plot mean curve with interpolation (for plot)
        time_grid = np.arange(-15, 40, 0.1)
        all_interp_curves = []
        for trial_df in dfs:
            interp_curve = np.interp(time_grid, trial_df.index.values, trial_df['temperature_aligned_for_plot'].values)
            all_interp_curves.append(interp_curve)
        all_interp_curves = np.array(all_interp_curves)
        mean_curve = np.mean(all_interp_curves, axis=0)
        axes[0].plot(time_grid, mean_curve, label=trial_type, color=color_dict[trial_type], linewidth=2)
    axes[0].set_ylabel("Temperature")
    axes[0].set_xlim(10, 40)
    axes[0].set_ylim((-18, 1) if 'innocuous' in pair else (-1.5, 1))
    axes[0].set_title(f"Avg Aligned Temperature: {pair[0]} vs {pair[1]} ({dataset})")
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
        for trial_df in dfs:
            interp_curve = np.interp(time_grid, trial_df.index.values, trial_df['pain'].values)
            all_interp_curves.append(interp_curve)
        all_interp_curves = np.array(all_interp_curves)
        mean_curve = np.mean(all_interp_curves, axis=0)
        sem_curve = np.std(all_interp_curves, axis=0, ddof=1) / np.sqrt(all_interp_curves.shape[0])
        axes[1].plot(time_grid, mean_curve, label=trial_type, color=color_dict[trial_type], linewidth=2)
        axes[1].fill_between(time_grid, mean_curve - 1.96*sem_curve, mean_curve + 1.96*sem_curve, color=color_dict[trial_type], alpha=0.2)
    axes[1].set_xlabel("Aligned Time (s)")
    axes[1].set_ylabel("Pain")
    axes[1].set_xlim(10, 40)
    axes[1].set_ylim((0, 70) if 'innocuous' in pair else (10, 100))
    axes[1].set_title(f"Average Pain Curves: {pair[0]} vs {pair[1]} ({dataset})")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# #%% 
# # ============================================================
# # Create a 2x2 grid of comparison plots
# # ============================================================

# fig = plt.figure(figsize=(16, 10))
# gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.25, wspace=0.25)

# color_map = plt.get_cmap('tab10')

# for comp in trial_pairs:
#     pair = comp['pair']
#     col = comp['col']
    
#     # Create axes for this column
#     ax_temp = fig.add_subplot(gs[0, col])
#     ax_pain = fig.add_subplot(gs[1, col])
    
#     color_dict = {k: color_map(i) for i, k in enumerate(pair)}
    
#     # Top: Plot temperature traces
#     for trial_type in pair:
#         if trial_type not in time_temp_aligned_trial_type:
#             continue
#         dfs = time_temp_aligned_trial_type[trial_type]
        
#         # Plot mean curve
#         all_curves = []
#         all_times = []
#         for df in dfs:
#             # Optionally offset t1_hold
#             if trial_type == 't1_hold':
#                 temp = df['temperature_aligned_for_plot'] - 1
#                 display_label = 't1_hold'
#             else:
#                 temp = df['temperature_aligned_for_plot']
#                 display_label = trial_type
#             all_curves.append(temp.values)
#             all_times.append(df.index.values)
        
#         # Concatenate and compute mean at each unique time
#         all_times_flat = np.concatenate(all_times)
#         all_curves_flat = np.concatenate(all_curves)
#         mean_df = pd.DataFrame({'time': all_times_flat, 'temp': all_curves_flat})
#         mean_curve = mean_df.groupby('time').mean().sort_index()
        
#         label = display_label if trial_type == 't1_hold_combined' else trial_type
#         ax_temp.plot(mean_curve.index, mean_curve['temp'], label=label, 
#                     color=color_dict[trial_type], linewidth=2)
    
#     ax_temp.set_ylabel("Temperature (°C)", fontsize=11)
#     ax_temp.set_xlim(-5, 30)
#     ax_temp.set_ylim(-1.5, 0.5)
#     pair_display = [p if p != 't1_hold_combined' else 't1_hold' for p in pair]
#     ax_temp.set_title(comp['title'], fontsize=13, fontweight='bold')
#     ax_temp.legend(fontsize=10)
#     ax_temp.grid(True, alpha=0.3)
    
#     # Bottom: Plot pain traces
#     for trial_type in pair:
#         if trial_type not in time_temp_aligned_trial_type:
#             continue
#         dfs = time_temp_aligned_trial_type[trial_type]
        
#         # Plot mean curve with better handling of missing data
#         time_grid = np.arange(-15, 40, 0.1)
#         all_interp_curves = []
        
#         for df in dfs:
#             # Only interpolate within the valid time range of each trial
#             valid_mask = (time_grid >= df.index.min()) & (time_grid <= df.index.max())
#             interp_curve = np.full(len(time_grid), np.nan)
            
#             if valid_mask.any() and len(df['pain']) > 0:
#                 # Remove NaN values before interpolation
#                 valid_data = df['pain'].dropna()
#                 if len(valid_data) > 1:
#                     interp_curve[valid_mask] = np.interp(
#                         time_grid[valid_mask], 
#                         valid_data.index.values, 
#                         valid_data.values,
#                         left=np.nan,
#                         right=np.nan
#                     )
#             all_interp_curves.append(interp_curve)
        
#         if len(all_interp_curves) > 0:
#             all_interp_curves = np.array(all_interp_curves)
            
#             # Compute mean and SEM only where we have data
#             with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
#                 mean_curve = np.nanmean(all_interp_curves, axis=0)
#                 n_valid = np.sum(~np.isnan(all_interp_curves), axis=0)
#                 sem_curve = np.nanstd(all_interp_curves, axis=0, ddof=1) / np.sqrt(n_valid)
                
#                 # Only plot where we have at least 3 trials contributing
#                 valid_points = n_valid >= 3
                
#                 label = 't1_hold' if trial_type == 't1_hold_combined' else trial_type
#                 ax_pain.plot(time_grid[valid_points], mean_curve[valid_points], 
#                            label=f"{label} (n={len(dfs)})", 
#                            color=color_dict[trial_type], linewidth=2)
#                 ax_pain.fill_between(
#                     time_grid[valid_points], 
#                     (mean_curve - 1.96*sem_curve)[valid_points], 
#                     (mean_curve + 1.96*sem_curve)[valid_points], 
#                     color=color_dict[trial_type], alpha=0.2
#                 )
    
#     ax_pain.set_xlabel("Aligned Time (s)", fontsize=11)
#     ax_pain.set_ylabel("Pain Rating", fontsize=11)
#     ax_pain.set_xlim(-5, 30)
#     ax_pain.set_ylim(0, 80)
#     ax_pain.legend(fontsize=10)
#     ax_pain.grid(True, alpha=0.3)

# # Add overall title
# fig.suptitle('Temperature Contrast Effects on Pain', fontsize=16, fontweight='bold', y=0.98)
# plt.show()

#%%
# =============================================================
# Plot example trials with period highlights and extracted metrics
# =============================================================
#%%
# =============================================================
# Plot example trials with period highlights and extracted metrics
# =============================================================
# Load metrics to get the period boundaries and extrema
with open(f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_metrics.json') as f:
    metrics_data = json.load(f)
np.random.seed(2)  

# Choose which trial types to visualize
trial_types_to_plot = ['offset', 'inv', 't1_hold', 't2_hold'] if dataset == 'plosONE' else ['offset', 'onset', 't1_hold', 't2_hold']

for trial_type in trial_types_to_plot:
    # Pick a random example trial (or you can specify subject/trial_num)
    trial_type_df = df[df['trial_type'] == trial_type]
    if trial_type_df.empty:
        continue
    
    # Filter to only include trials that exist in metrics_data
    def trial_has_metrics(row):
        subj_key = str(int(row['subject']))
        trial_key = str(int(row['trial_num']))
        return subj_key in metrics_data and trial_key in metrics_data[subj_key]
    
    trial_type_df_with_metrics = trial_type_df[trial_type_df.apply(trial_has_metrics, axis=1)]
    
    if trial_type_df_with_metrics.empty:
        print(f"No trials with metrics found for {trial_type}")
        continue
    
    # Get random subject and trial from those with metrics
    random_row = trial_type_df_with_metrics.sample(1).iloc[0]
    example_subject = random_row['subject']
    example_trial_num = random_row['trial_num']

    # Get trial data and metrics
    trial_data = df[(df['subject'] == example_subject) & 
                    (df['trial_num'] == example_trial_num)].copy()
    trial_metrics = metrics_data[str(int(example_subject))][str(int(example_trial_num))]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax_temp = ax.twinx()
    
    # Determine if this is a stepped trial (experimental) or control trial
    is_stepped = trial_type in ['offset', 'inv', 'onset', 'stepdown']
    
    # Plot pain on left axis (blue)
    pain_line = ax.plot(trial_data['aligned_time'], trial_data['pain'], 
                       color='blue', linewidth=2, label='Pain Rating')[0]
    
    # Plot temperature on right axis (red)
    temp_line = ax_temp.plot(trial_data['aligned_time'], trial_data['temperature'], 
                            color='red', alpha=0.7, linewidth=2, label='Temperature')[0]
    
    # Set axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_ylabel('Pain Rating', color='blue', fontsize=12)
    ax.tick_params(axis='y', labelcolor='blue')
    
    ax_temp.set_ylim(29, 50)
    ax_temp.set_ylabel('Temperature (°C)', color='red', fontsize=12)
    ax_temp.tick_params(axis='y', labelcolor='red')
    
    # Mark periods A, B, C
    for period in ['A', 'B', 'C']:
        start = trial_metrics[f'{period}_start']
        end = trial_metrics[f'{period}_end']
        color = {'A': 'yellow', 'B': 'lightgreen', 'C': 'lightblue'}[period]
        ax.axvspan(start, end, color=color, alpha=0.2, label=f'Period {period}')
    
    # Mark extrema
    if is_stepped:
        # Mark local extrema for stepped trials
        if trial_metrics.get('abs_min_time') is not None:
            ax.axvline(trial_metrics['abs_min_time'], color='green', linestyle='--', 
                      linewidth=2, label=f"Local Min ({trial_metrics['abs_min_val']:.1f})")
            ax.plot(trial_metrics['abs_min_time'], trial_metrics['abs_min_val'], 
                   'go', markersize=10)
        
        if trial_metrics.get('abs_max_time') is not None:
            ax.axvline(trial_metrics['abs_max_time'], color='orange', linestyle='--', 
                      linewidth=2, label=f"Local Max ({trial_metrics['abs_max_val']:.1f})")
            ax.plot(trial_metrics['abs_max_time'], trial_metrics['abs_max_val'], 
                   'o', color='orange', markersize=10)
    else:
        # For control trials, determine the reference suffix
        if trial_type == 't1_hold':
            reference_suffix = 'offset'  # t1_hold references offset
        elif trial_type == 't2_hold':
            reference_suffix = 'onset' if dataset in ['kneeOA', 'cLBP'] else 'inv'
        else:
            reference_suffix = None
        
        if reference_suffix:
            # Mark time-yoked extrema (from reference trial)
            min_time_col = f'time_yoked_min_time_{reference_suffix}'
            min_val_col = f'time_yoked_min_val_{reference_suffix}'
            max_time_col = f'time_yoked_max_time_{reference_suffix}'
            max_val_col = f'time_yoked_max_val_{reference_suffix}'
            
            if trial_metrics.get(min_time_col) is not None and not pd.isna(trial_metrics.get(min_time_col)):
                ax.axvline(trial_metrics[min_time_col], color='green', 
                          linestyle=':', linewidth=2, alpha=0.7,
                          label=f"Time-yoked Min ({trial_metrics[min_val_col]:.1f})")
                ax.plot(trial_metrics[min_time_col], trial_metrics[min_val_col], 
                       'o', color='green', markersize=8, alpha=0.7, markeredgecolor='darkgreen', markeredgewidth=2)
            
            if trial_metrics.get(max_time_col) is not None and not pd.isna(trial_metrics.get(max_time_col)):
                ax.axvline(trial_metrics[max_time_col], color='orange', 
                          linestyle=':', linewidth=2, alpha=0.7,
                          label=f"Time-yoked Max ({trial_metrics[max_val_col]:.1f})")
                ax.plot(trial_metrics[max_time_col], trial_metrics[max_val_col], 
                       'o', color='orange', markersize=8, alpha=0.7, markeredgecolor='darkorange', markeredgewidth=2)
        
        # Mark absolute extrema (the actual extrema in this control trial)
        if trial_metrics.get('abs_max_time') is not None and not pd.isna(trial_metrics.get('abs_max_time')):
            ax.axvline(trial_metrics['abs_max_time'], color='purple', 
                      linestyle='--', linewidth=2,
                      label=f"Absolute Max ({trial_metrics['abs_max_val']:.1f})")
            ax.plot(trial_metrics['abs_max_time'], trial_metrics['abs_max_val'], 
                   's', color='purple', markersize=8)
        
        if trial_metrics.get('abs_min_time') is not None and not pd.isna(trial_metrics.get('abs_min_time')):
            ax.axvline(trial_metrics['abs_min_time'], color='darkviolet', 
                      linestyle='--', linewidth=2,
                      label=f"Absolute Min ({trial_metrics['abs_min_val']:.1f})")
            ax.plot(trial_metrics['abs_min_time'], trial_metrics['abs_min_val'], 
                   's', color='darkviolet', markersize=8)
    
    ax.set_xlabel('Aligned Time (s)', fontsize=12)
    ax.set_title(f"Example {trial_type.upper()} Trial with Extracted Metrics\nSubject {example_subject}, Trial {example_trial_num}", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0,60)
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_temp.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %%
