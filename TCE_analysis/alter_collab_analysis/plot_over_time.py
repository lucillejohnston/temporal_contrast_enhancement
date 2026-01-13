#%%
#!/usr/bin/env python3
import json, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
with open('/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
sys.path.append('/userdata/ljohnston/TCE_analysis/data_from_ben/')
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

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color_pain = 'tab:red'
ax2.set_ylabel('Pain', color=color_pain)
ax2.plot(filtered_df['actual_time'], filtered_df['pain'], color=color_pain, label='Pain')
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
plt.savefig(f'/userdata/ljohnston/TCE_analysis/data_from_ben/whole_session_example.svg', dpi=300)
plt.show()

# %%
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


