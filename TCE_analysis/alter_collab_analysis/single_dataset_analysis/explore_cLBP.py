#%%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open(f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/cLBP_trial_data.json') as f:
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
    'offset': 'offset',
    't1_hold': 't1_hold',
    't2_hold': 't2_hold',
    'onset': 'onset',
}
trial_data['trial_type'] = trial_data['trial_type'].map(trial_key)

# Convert relative_time to numeric seconds
trial_data['relative_time'] = (
    pd.to_timedelta(trial_data['relative_time'])
    .dt.total_seconds()
)

print(trial_data[['relative_time', 'temperature', 'pain']].head())
print(trial_data['relative_time'].describe())
#%%
# ============================================================
# RAW TEMPERATURE TRACES BY TRIAL TYPE
# ============================================================

trial_types = ['onset', 'offset', 't1_hold', 't2_hold']

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 10),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, trial_type in zip(axes, trial_types):

    subset = trial_data[trial_data['trial_type'] == trial_type]

    for (subject, trial_num, study), trial_df in subset.groupby(
        ['subject', 'trial_num', 'study']
    ):

        trial_df = trial_df.sort_values('relative_time')

        ax.plot(
            trial_df['relative_time'],
            trial_df['temperature'],
            alpha=0.15,
            linewidth=1
        )

    ax.set_title(trial_type)
    ax.set_xlabel('Relative Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.grid(True, alpha=0.3)

plt.suptitle('Raw cLBP Temperature Traces')
plt.tight_layout()
plt.show()

#%%
# ============================================================
# RANDOM SINGLE-TRIAL TEMP + PAIN PLOT
# ============================================================

# Pick a random trial
random_key = (
    trial_data[['subject', 'trial_num', 'study']]
    .drop_duplicates()
    .sample(1)
    .iloc[0]
)

subject = random_key['subject']
trial_num = random_key['trial_num']
study = random_key['study']

print(f"Subject: {subject}")
print(f"Trial: {trial_num}")
print(f"Study: {study}")

# Get trial data
trial_df = trial_data[
    (trial_data['subject'] == subject) &
    (trial_data['trial_num'] == trial_num) &
    (trial_data['study'] == study)
].sort_values('relative_time')

# Pain samples only
pain_df = trial_df.dropna(subset=['pain']).copy()

# ============================================================
# Plot
# ============================================================

fig, ax1 = plt.subplots(figsize=(12, 5))

# Temperature trace
ax1.plot(
    trial_df['relative_time'],
    trial_df['temperature'],
    color='red',
    linewidth=2,
    alpha=0.8,
    label='Temperature'
)

ax1.set_xlabel('Relative Time (s)')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_ylim(30, 50)

# Pain axis
ax2 = ax1.twinx()

# Scatter actual pain samples
ax2.scatter(
    pain_df['relative_time'],
    pain_df['pain'],
    color='blue',
    s=40,
    edgecolors='black',
    linewidths=0.5,
    zorder=3,
    label='Pain samples'
)

# Optional: connect pain points lightly just for visualization
ax2.plot(
    pain_df['relative_time'],
    pain_df['pain'],
    color='blue',
    alpha=0.4,
    linewidth=1
)

ax2.set_ylabel('Pain', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(0, 100)

plt.title(
    f'cLBP Trial\nSubject={subject} | Trial={trial_num} | Study={study}'
)

ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()