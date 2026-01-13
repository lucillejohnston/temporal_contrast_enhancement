import os
import sys
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from ast import literal_eval  # for safely converting string to list
import csv
import numpy as np

#!/usr/bin/env python3
"""
Quick script to load today's data and plot temperature and pain rating over time,
and overlay experimenter notes on the plot.

Assumptions:
- Files are stored as CSV files in the calibration_data folder.
- Trial rows contain 4 fields:
    trial_count,timestamp,pain_rating,temperature 
  where 'temperature' is a stringified list.
- Note rows contain 2 fields:
    timestamp,note text
- Timestamps are in H:M:S.ms format.
"""

# Define the default data directory.
current_dir = os.path.dirname(__file__)
tce_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Open a file dialog to allow the user to manually select a file.
root = tk.Tk()
root.withdraw()
selected_file = filedialog.askopenfilename(
    title="Select calibration CSV file to plot",
    initialdir=tce_dir,
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)

def load_data(file_path):
    trial_rows = []
    note_rows = []
    with open(file_path, newline='', encoding='latin-1') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Skip empty rows.
            if not row or row[0] == 'trial_count': #skip header and empty rows
                continue
            try:
                int(row[0])
                trial_rows.append(row)
            except ValueError:
                note_rows.append(row)
    num_cols = len(trial_rows[0])
    if num_cols == 4:
        columns = ["trial_count", "timestamp", "pain_rating", "temperature"]
    elif num_cols == 5:
        columns = ["trial_count", "trial_type", "timestamp", "pain_rating", "temperature"]
    else:
        print(f"Unexpected number of columns ({num_cols}) in trial data.")
        sys.exit(1)
    trial_data = pd.DataFrame(trial_rows, columns=columns)
    try:
        # Parse timestamp column; format H:M:S.ms.
        trial_data['timestamp'] = pd.to_datetime(trial_data['timestamp'], format="%H:%M:%S.%f")
    except Exception as e:
        print(f"Error parsing timestamps in trial data: {e}")
        sys.exit(1)
    try:
        trial_data['pain_rating'] = pd.to_numeric(trial_data['pain_rating'], errors='raise')
    except Exception as e:
        print(f"Error converting pain_rating to numeric: {e}")
        sys.exit(1)
    try:
        # Process temperature column: convert stringified list to average.
        trial_data['temperature'] = trial_data['temperature'].apply(
            lambda x: max(literal_eval(x)) if pd.notnull(x) and x != "" else np.nan)
    except Exception as e:
        print(f"Error processing temperature values: {e}")
        sys.exit(1)

    note_data_list = []
    for row in note_rows:
        if len(row) >= 2:
            ts_str, note = row[0], row[1]
            if ts_str.lower() == "timestamp":
                continue
            try:
                ts = pd.to_datetime(ts_str, format="%H:%M:%S.%f")
            except Exception as e:
                print(f"Error parsing note timestamp: {e}")
                continue
            note_data_list.append({"timestamp": ts, "note": note})
    note_data = pd.DataFrame(note_data_list)

    # Resample trial_data at 10Hz to insert NaNs for gaps.
    trial_data = trial_data.sort_values('timestamp')
    trial_data['timestamp'] = trial_data['timestamp'].dt.round('100ms')
    trial_data = trial_data.drop_duplicates(subset=['timestamp'])
    trial_data = trial_data.set_index('timestamp')
    new_index = pd.date_range(start=trial_data.index.min(), 
                              end=trial_data.index.max(), 
                              freq='100ms')
    trial_data = trial_data.reindex(new_index)
    trial_data.index.name = 'timestamp'
    trial_data = trial_data.reset_index()
    trial_data['pain_rating'] = trial_data['pain_rating'].ffill()


    return trial_data, note_data

def plot_data(trial_data, note_data):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot temperature on left y-axis.
    valid_temp = trial_data[trial_data['temperature'].notna()]
    ax1.plot(valid_temp['timestamp'], valid_temp['temperature'], color='red', label='Temperature')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Plot pain_rating on right y-axis.
    ax2 = ax1.twinx()
    ax2.plot(trial_data['timestamp'], trial_data['pain_rating'], color='blue', label='Pain Rating')
    ax2.set_ylabel('Pain Rating', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 100)

    plt.title('Temperature and Pain Rating Over Time')

    # Add background shading in blocks for each trial.
    # Group data by trial_count and shade alternating blocks.
    unique_trials = sorted(trial_data['trial_count'].dropna().unique(), key=lambda t: int(t))
    for idx, trial in enumerate(unique_trials):
        trial_rows = trial_data[trial_data['trial_count'] == trial]
        start_time = trial_rows['timestamp'].min()
        end_time = trial_rows['timestamp'].max()
        # Only set trial_type if the column exists.
        trial_type = trial_rows['trial_type'].iloc[0] if 'trial_type' in trial_data.columns else ""
        if idx % 2 == 0:
            ax1.axvspan(start_time, end_time, facecolor='lightgray', alpha=0.3)
        
        if trial_type:
            mid_time = start_time + (end_time - start_time) / 2
            y_top = ax1.get_ylim()[1] * 0.98
            ax1.text(mid_time, y_top, str(trial_type),
                     ha='center', va='top', fontsize=8, color='black')
    
    # Overlay notes on the plot.
    if not note_data.empty:
        # Update the plot so ax1 has a finalized y-limit.
        plt.draw()  
        ylim = ax1.get_ylim()
        y_pos = ylim[1] * 0.95  # 95% of the max y-value for text placement
        for idx, row in note_data.iterrows():
            timestamp = row['timestamp']
            note = row['note']
            # Draw a dotted vertical line.
            ax1.axvline(x=timestamp, linestyle=":", color='green')
            # Place the note text at the top (rotated vertically for clarity).
            ax1.annotate(note,
                xy=(timestamp, y_pos),
                xytext=(0, 5),
                textcoords="offset points",
                ha='left',
                va='bottom',
                color='green',
                fontsize=9,
                rotation=90)
    
    plt.tight_layout()
    plt.show()

def main():
    if selected_file:
        file_path = selected_file
    else:
        print("No file selected.")
        sys.exit(1)
    print(f"Loading data from: {file_path}")
    trial_data, note_data = load_data(file_path)
    plot_data(trial_data, note_data)

if __name__ == '__main__':
    main()
