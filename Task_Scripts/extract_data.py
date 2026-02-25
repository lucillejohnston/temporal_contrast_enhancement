import sqlite3
import pandas as pd
import os, re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ---------------------------
# 1. Create a metadata table.
conn = sqlite3.connect('combined_data.sqlite')
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        subject INTEGER,
        trial_num INTEGER,
        trial_date TEXT,
        PRIMARY KEY (subject, trial_num)
    )
""")

# create threshold_data table 
cur.execute("""
    CREATE TABLE IF NOT EXISTS threshold_data (
        subject INTEGER PRIMARY KEY,
        limits1 REAL,
        limits2 REAL,
        limits3 REAL
    )
""")

# create calibration_data table #### for now we will leave blank 
cur.execute("""
    CREATE TABLE IF NOT EXISTS calibration_data (
        subject INTEGER PRIMARY KEY
    )
""")

# create trial_data table
cur.execute("""
    CREATE TABLE IF NOT EXISTS trial_data (
        subject INTEGER,
        trial_num INTEGER,
        trial_type TEXT,
        timestamp TEXT,
        temperature REAL,
        pain REAL,
        notes TEXT
    )    
""")

conn.commit()

# ---------------------------
# 2. Extract data 

# # Extract subject numbers
mainpath = '/Users/paulettebogan/UCSF DBS for Pain Dropbox/PainNeuromodulationLab/DATA ANALYSIS/Lucy/BenAlter_Collab_Data/CLEAR SUBJECT DATA/'
subject_info = {}
metadata_rows = []

for folder_name in os.listdir(mainpath):
    match = re.match(r'^CLEAR subject (\d+)$', folder_name)
    if match:
        subj = int(match.group(1))
        subject_info[subj] = "" # empty string for now
subject_numbers = sorted(subject_info.keys())
print('subject_info:', subject_info)
print('subject_numbers:', subject_numbers)

# Load limits data (pain threshold) 
excel_path = os.path.join(mainpath, '17-06-15 Limits compiled.xlsx')
thresholds_df = pd.read_excel(excel_path, usecols=['subj', 'limits1', 'limits2', 'limits3'])

def extract_trial_start(filename):
    match = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4} \d{1,2}h\d{2}m\d{2}s)', filename)
    if match:
        ts_str = match.group(1)
        try:
            return pd.to_datetime(ts_str, format='%d-%b-%Y %Hh%Mm%Ss')
        except Exception:
            return pd.NaT
    return pd.NaT

# # Load trial data
subject_data = {}
for subject in subject_numbers:
    subject_folder = os.path.join(mainpath, f'CLEAR subject {subject:03d}/')
    trial_files = [f for f in os.listdir(subject_folder) if f.startswith('clear #') and f.endswith('.xlsx')]
    
    trial_files.sort(key=extract_trial_start)  # Sort files by trial start time
    trials = {}
    for trial_num, trial_file in enumerate(trial_files):
        trial_path = os.path.join(subject_folder, trial_file)
        trial_start_dt = extract_trial_start(trial_file)
        trial_date_str = trial_start_dt.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(trial_start_dt) else ""
        # Save metadata for each trial.
        metadata_rows.append({
            'subject': subject,
            'trial_num': trial_num,
            'trial_date': trial_date_str
        })
        trial_df = pd.read_excel(trial_path, sheet_name='Data', usecols=['Timestamp [msec]', 'Tec [C]', 'COVAS'], engine='openpyxl')
        descr_df = pd.read_excel(trial_path, sheet_name='Description', header=None, engine='openpyxl')
        trial_type = descr_df.loc[descr_df[0] == 'Program', 1].values[0]
        trial_notes = descr_df.loc[descr_df[0] == 'Comments', 1].values[0]
        trial_df.columns = ['timestamp', 'temperature', 'pain']
        trial_df['timestamp'] = pd.to_datetime(trial_df['timestamp'], unit='ms')
        trial_df['trial_type'] = trial_type
        trial_df['notes'] = trial_notes
        trial_df['subject'] = subject
        trial_df['trial_num'] = trial_num
        trials[trial_num] = trial_df
    
        # Concatenate all trials for the subject
        subject_data[subject] = trials

# 3. Write metadata to SQL
meta_df = pd.DataFrame(metadata_rows)
print('meta_df columns:', meta_df.columns)
meta_df.to_sql('metadata', conn, if_exists='replace', index=False)

# ----- threshold data ------ 
threshold_df = thresholds_df.rename(columns={'subj':'subject'})[['subject','limits1','limits2','limits3']]
threshold_df.to_sql('threshold_data',conn, if_exists='replace',index=False)

# ----- calibration data (currently just subjects) -----
calibration_df = pd.DataFrame({'subject': list(subject_numbers)})
calibration_df.to_sql('calibration_data', conn, if_exists='replace',index=False)

# ----- trial data -----
trial_rows = []
for subject, trials in subject_data.items():
    for trial_num, trial_df in trials.items():
        trial_df = trial_df.copy()
        trial_df['subject'] = subject
        trial_df['trial_num'] = trial_num
        if 'trial_type' not in trial_df.columns:
            trial_df['trial_type'] = None
        trial_rows.append(trial_df)
trial_data_df = pd.concat(trial_rows, ignore_index=True)
trial_data_df['timestamp'] = trial_data_df['timestamp'].astype(str) # for storage purposes
trial_data_df.to_sql('trial_data', conn, if_exists='replace', index=False)

conn.close()
print("Combined database created: combined_data.sqlite")

