#%%
#!/usr/bin/env python3
import json, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
# Updated 2026-03-10 now it only works for the kneeOA data not sure what is going on with the plosONE data since I've re-preprocessed it
# But it takes literally FOREVER to load in the .json 
dataset = 'cLBP' # options: 'plosONE', 'kneeOA', 'cLBP'
with open(f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/{dataset}_trial_data.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
df['actual_time'] = pd.to_datetime(df['actual_time'])
sys.path.append('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/TCE_analysis/alter_collab_analysis/')
import plotting_functions as pf
# %%
# Now plot temperature and pain over actual_time for a random subject

# Filter for a random subject.
subjects = df['subject'].unique()
random_subject = np.random.choice(subjects)
filtered_df = df[df['subject'] == random_subject]
print("Plotting data for subject:", random_subject)

fig, ax1 = plt.subplots(figsize=(12,6))
color_temp = 'tab:blue'
ax1.set_xlabel('Actual Clock Time')
ax1.set_ylabel('Temperature', color=color_temp)
ax1.plot(filtered_df['actual_time'], filtered_df['temperature'], color=color_temp, label='Temperature')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))

if dataset == 'kneeOA' or dataset == 'plosONE':
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_pain = 'tab:red'
    ax2.set_ylabel('Pain', color=color_pain)
    ax2.plot(filtered_df['actual_time'], filtered_df['pain'], color=color_pain, label='Pain')
    ax2.tick_params(axis='y', labelcolor=color_pain)
    ax2.set_ylim(0, 100)
elif dataset == 'cLBP':
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_pain = 'tab:red'
    ax2.set_ylabel('Pain', color=color_pain)

    # Plot pain data as both line and markers
    pain_data = filtered_df.dropna(subset=['pain'])
    if not pain_data.empty:
        ax2.plot(pain_data['actual_time'], pain_data['pain'], 
                color=color_pain, marker='o', markersize=4, 
                linewidth=1, alpha=0.8, label='Pain')
        print(f"Plotting {len(pain_data)} pain points")

    ax2.tick_params(axis='y', labelcolor=color_pain)
    ax2.set_ylim(0, 100)

if 'trial_num' in filtered_df.columns and 'trial_type' in filtered_df.columns:
    trial_groups = filtered_df.groupby('trial_num')
    for trial_num, group in trial_groups:
        trial_type = group['trial_type'].iloc[0]
        start_time = group['actual_time'].iloc[0]
        end_time = group['actual_time'].iloc[-1]
        mid_time = group['actual_time'].iloc[len(group)//2]
        # Draw vertical lines at trial boundaries
        ax1.axvline(x=start_time, color='gray', linestyle='--', alpha=0.5)
        # Annotate trial_type above the plot
        ax1.annotate(
            f'{trial_type}',
            xy=(mid_time, 1.05),
            xycoords=('data', 'axes fraction'),
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
    # Draw last boundary
    ax1.axvline(x=end_time, color='gray', linestyle='--', alpha=0.5)

plt.title(f'Temperature and Pain Over Time for Subject {random_subject}')
fig.tight_layout()  # for proper layout
plt.show()

# %% Plot a single trial for the same subject
trials = filtered_df['trial_num'].unique()
random_trial = np.random.choice(trials) # Select a random trial for the filtered subject
trial_df = filtered_df[filtered_df['trial_num'] == random_trial]

# Extract trial_label for the selected trial
if 'trial_type' in trial_df.columns:
    trial_type = trial_df['trial_type'].iloc[0]
else:
    trial_type = None

print("Plotting trial", random_trial, "for subject", random_subject)

fig, ax1 = plt.subplots(figsize=(12,6))

color_temp = 'tab:blue'
ax1.set_xlabel('Actual Time')
ax1.set_ylabel('Temperature', color=color_temp)
ax1.plot(trial_df['actual_time'], trial_df['temperature'], color=color_temp, label='Temperature')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))

ax2 = ax1.twinx()  # share the x-axis
color_pain = 'tab:red'
ax2.set_ylabel('Pain', color=color_pain)
ax2.plot(trial_df['actual_time'], trial_df['pain'], color=color_pain, label='Pain')
ax2.tick_params(axis='y', labelcolor=color_pain)
ax2.set_ylim(0, 100)

plt.title(f'Trial {random_trial} - Temperature and Pain Over Time for Subject {random_subject}')
if trial_type is not None:
    plt.suptitle(f'Trial Type: {trial_type}', y=1.03, fontsize=14)
fig.tight_layout()
plt.show()


# %%
# Create a plot of all 4 trial types in a series

# %%
def find_all_trial_series(df, required_types=['OA', 'OH', 'T1', 'T2']):
    """Find all possible series and return them"""
    all_matches = []
    subjects = df['subject'].unique()
    
    for subject in subjects:
        subject_df = df[df['subject'] == subject].copy()
        trials = sorted(subject_df['trial_num'].unique())
        
        for i in range(len(trials) - 3):
            trial_group = trials[i:i+4]
            trial_types = []
            for trial in trial_group:
                trial_data = subject_df[subject_df['trial_num'] == trial]
                if not trial_data.empty:
                    trial_types.append(trial_data['trial_type'].iloc[0])
            
            if set(trial_types) == set(required_types):
                all_matches.append((subject, trial_group, trial_types))
    
    return all_matches
# Find all matches
all_options = find_all_trial_series(df)
print(f"Found {len(all_options)} total matches:")
for i, (subj, trials, types) in enumerate(all_options[:5]):  # Show first 5
    print(f"Option {i}: Subject {subj}, Trials {trials}, Types {types}")

# Pick one (change the index to try different ones)
if all_options:
    subject, trial_series, trial_types = all_options[8]  # Change 0 to 1, 2, etc.

# Plot the series
subject_df = df[df['subject'] == subject]
filtered_df = subject_df[subject_df['trial_num'].isin(trial_series)].copy()
filtered_df = filtered_df.sort_values('actual_time')

fig, ax1 = plt.subplots(figsize=(16, 8))

# Temperature plot
color_temp = 'tab:blue'
ax1.set_xlabel('Actual Clock Time', fontsize=12)
ax1.set_ylabel('Temperature (°C)', color=color_temp, fontsize=12)
ax1.plot(filtered_df['actual_time'], filtered_df['temperature'], 
         color=color_temp, linewidth=1.5, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))

# Pain plot
ax2 = ax1.twinx()
color_pain = 'tab:red'
ax2.set_ylabel('Pain Rating', color=color_pain, fontsize=12)
ax2.plot(filtered_df['actual_time'], filtered_df['pain'], 
         color=color_pain, linewidth=1.5, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color_pain)
ax2.set_ylim(0, 100)

# Add trial boundaries and labels
trial_groups = filtered_df.groupby('trial_num')
for trial_num, group in trial_groups:
    trial_type = group['trial_type'].iloc[0]
    start_time = group['actual_time'].iloc[0]
    mid_time = group['actual_time'].iloc[len(group)//2]
    
    ax1.axvline(x=start_time, color='gray', linestyle='--', alpha=0.7)
    ax1.annotate(f'Trial {trial_num}\n{trial_type}',
                xy=(mid_time, 1.05), xycoords=('data', 'axes fraction'),
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))

# Final boundary
final_group = list(trial_groups)[-1][1]
ax1.axvline(x=final_group['actual_time'].iloc[-1], color='gray', linestyle='--', alpha=0.7)
plt.title(f'4-Trial Series: Temperature and Pain Over Time - Subject {subject}', fontsize=14)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
fig.tight_layout()
plt.savefig('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/Example4Trials.svg')


plt.show()

