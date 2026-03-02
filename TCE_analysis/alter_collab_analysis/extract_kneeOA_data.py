"""
Extracts data from the PAIN 2025 paper 
Organizes it into a database with 4 tables: metadata, threshold_data, calibration_data, trial_data

metadata: primary key is (subject, trial_num, study)
- subject (int)
- trial_num (int)
- study (text): 'plosONE', 'kneeOA', etc. 
- trial_date (text) 

threshold_data: primary key is (subject, study)
- subject (int)
- limits1 (real)
- limits2 (real)
- limits3 (real)
- study (text)

calibration_data: primary key is (subject, study)
- subject (int)
- study (text)
currently a placeholder eventually will put calibration data here 

trial_data: primary key is (subject, trial_num, timestamp)
- subject (int)
- trial_num (int)
- trial_type (text)
- timestamp (text)
- temperature (real)
- pain (real)
- notes (text)

"""
#%%
import pandas as pd
import numpy as np
import scipy.io 
import sqlite3

# Load in the kneeOA data
trial_info_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/kneeOA_trial_info.mat'
trial_metadata = scipy.io.loadmat(trial_info_path)
kneeOA_data_path = '/Users/ljohnston1/UCSF DBS for Pain Dropbox/PainNeuromodulationLab/DATA ANALYSIS/Lucy/BenAlter_Collab_Data/KneeNIRS data/Ben Matlab dataset and pipeline/cleanRS.mat'
data_raw = scipy.io.loadmat(kneeOA_data_path)
subject_info_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/kneeOA_subject_groups.mat'
subject_groups = scipy.io.loadmat(subject_info_path)['subject_groups_struct'][0,0]


all_trial_info = trial_metadata['all_trial_info']
print(f"Loaded trial info structure: {type(all_trial_info)}")

data = data_raw['data']
ohoa_data = data['OHOA'][0, 0]
subject_fields = [field for field in ohoa_data.dtype.names if field.startswith('subj')]
study_ids = np.squeeze(subject_groups['study_id']).astype(int)
group_cells = np.squeeze(subject_groups['group'])
groups = [str(np.squeeze(g)) for g in group_cells]
subject_info_df = pd.DataFrame({
    'subject': study_ids,
    'group_label': groups,
    'study': 'kneeOA'
})

#%% Create the SQLite database and tables
# Clean data extraction excluding subject 92 (FIXED VERSION)
metadata_list = []
trial_data_list = []
trial_info_data = all_trial_info[0, 0]
print("Processing all subjects (excluding subject 92 due to data corruption)...")

for subj_field in subject_fields:
    subject_num = int(subj_field.replace('subj', ''))
    
    # Skip subject 92 due to data corruption
    if subject_num == 92:
        print(f"Skipping {subj_field} due to data corruption...")
        continue
        
    print(f"Processing {subj_field}...")
    
    # Get trial info
    subj_trial_info = trial_info_data[subj_field][0, 0]
    subj_data = ohoa_data[subj_field][0, 0]
    resampled_data = subj_data['resampled'][0, 0]
    
    
    # Trial metadata
    trial_types = subj_trial_info['trialtype']
    sites = subj_trial_info['site'] 
    spots = subj_trial_info['spot']
    znos = subj_trial_info['zno'].flatten()
    trial_dates = subj_trial_info['trial_date']
    
    # Find valid trials
    max_zno = int(np.max(znos))
    
    for trial_idx in range(len(znos)):
        zno = int(znos[trial_idx])
        
        if 1 <= zno <= max_zno:
            # Extract trial info
            trial_type = trial_types[trial_idx, 0][0] if trial_types[trial_idx, 0].size > 0 else 'unknown'
            site = sites[trial_idx, 0][0] if sites[trial_idx, 0].size > 0 else 'unknown'  
            spot = spots[trial_idx, 0][0] if spots[trial_idx, 0].size > 0 else 'unknown'
            try:
                trial_date = str(trial_dates[trial_idx, 0][0])
            except (IndexError, TypeError):
                trial_date = ''
                print(f" Warning: no trial_date for {subj_field} trial {zno}")
            # Add to metadata
            metadata_list.append({
                'subject': subject_num,
                'trial_num': zno,
                'trial_date': trial_date,
                'group': subject_info_df.loc[subject_info_df['subject'] == subject_num, 'group_label'].values[0],
                'study': 'kneeOA'
            })
            
            # Extract time series for this trial
            trial_timeseries = resampled_data[:, :, zno-1]
            
            # Correct column assignments:
            timestamp_data = trial_timeseries[:, 0]  # Timestamps in seconds
            temp_data = trial_timeseries[:, 1]       # Temperature (~32°C)
            pain_data = trial_timeseries[:, 2]       # Pain ratings (0-100 scale)
            
            notes = f"site={site}, spot={spot}"
            
            # Add time series data 
            for timepoint in range(len(pain_data)):
                ts_valid = float(timestamp_data[timepoint])
                if ts_valid <= 0 or np.isnan(ts_valid):
                    continue # Skip invalid timestamps
                trial_data_list.append({
                    'subject': subject_num,
                    'trial_num': zno,
                    'trial_type': trial_type,
                    'timestamp': str(timestamp_data[timepoint]),
                    'temperature': float(temp_data[timepoint]),
                    'pain': float(pain_data[timepoint]),
                    'notes': notes,
                    'study': 'kneeOA' 
                })

#%%
# Create DataFrames
metadata_df = pd.DataFrame(metadata_list)
trial_data_df = pd.DataFrame(trial_data_list)

print(f"\n=== EXTRACTION COMPLETE (Subject 92 excluded) ===")
print(f"Metadata records: {len(metadata_df)}")
print(f"Trial data records: {len(trial_data_df)}")
print(f"Subjects included: {len(metadata_df['subject'].unique())}")

# Verify columns
print(f"\nColumns in metadata_df: {metadata_df.columns.tolist()}")
print(f"Columns in trial_data_df: {trial_data_df.columns.tolist()}")

# Clean the data by removing any remaining problematic records
print("\n=== CLEANING DATA ===")
print(f"Original trial data records: {len(trial_data_df)}")

# Remove rows with missing values
cleaned_trial_data = trial_data_df.dropna()
print(f"After removing NaN: {len(cleaned_trial_data)}")

# Remove rows with extreme temperatures (reasonable range for thermal stimuli)
cleaned_trial_data = cleaned_trial_data[
    (cleaned_trial_data['temperature'] >= 0) & 
    (cleaned_trial_data['temperature'] <= 60)
]
print(f"After temperature filtering (0-60°C): {len(cleaned_trial_data)}")

# Remove rows with invalid pain values
cleaned_trial_data = cleaned_trial_data[
    (cleaned_trial_data['pain'] >= -2) & 
    (cleaned_trial_data['pain'] <= 102)
]
print(f"After pain filtering (-2 to 102): {len(cleaned_trial_data)}")

# Update metadata to only include trials that have valid data
valid_trials = cleaned_trial_data[['subject', 'trial_num', 'study']].drop_duplicates()
cleaned_metadata = metadata_df.merge(valid_trials, on=['subject', 'trial_num', 'study'])

print(f"\n=== FINAL CLEAN DATA SUMMARY ===")
print(f"Clean metadata records: {len(cleaned_metadata)}")
print(f"Clean trial data records: {len(cleaned_trial_data)}")
print(f"Valid subjects: {len(cleaned_metadata['subject'].unique())}")
print(f"Temperature range: {cleaned_trial_data['temperature'].min():.2f} to {cleaned_trial_data['temperature'].max():.2f}")
print(f"Pain range: {cleaned_trial_data['pain'].min():.2f} to {cleaned_trial_data['pain'].max():.2f}")

print(f"\nUnique trial types: {sorted(cleaned_trial_data['trial_type'].unique())}")
print(f"Trials per subject (sample):")
print(cleaned_metadata.groupby('subject')['trial_num'].count().head())

# Replace with cleaned data
trial_data_df = cleaned_trial_data
metadata_df = cleaned_metadata
# %% Connect to the SQLite database
sql_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_data.sqlite'
print(f"\n=== INSERTING DATA INTO SQL DATABASE ===")
print(f"Database path: {sql_path}")

try:
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()
    
    # Check current database contents
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study = 'plosONE'")
    plos_metadata_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM trial_data WHERE study = 'plosONE'")
    plos_trial_count = cur.fetchone()[0]
    
    print(f"Current database contents:")
    print(f"  plosONE metadata records: {plos_metadata_count}")
    print(f"  plosONE trial data records: {plos_trial_count}")
    
    # Check if kneeOA data already exists
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study = 'kneeOA'")
    existing_kneeoa = cur.fetchone()[0]
    
    if existing_kneeoa > 0:
        print(f"WARNING: Found {existing_kneeoa} existing kneeOA records. Do you want to:")
        print("1. Skip insertion (data already exists)")
        print("2. Delete existing kneeOA data and insert new data")
        print("3. Proceed anyway (may create duplicates)")
        
        choice = input("Enter choice (1, 2, or 3): ")
        
        if choice == '1':
            print("Skipping insertion.")
            conn.close()
            exit()
        elif choice == '2':
            print("Deleting existing kneeOA data...")
            cur.execute("DELETE FROM metadata WHERE study = 'kneeOA'")
            cur.execute("DELETE FROM trial_data WHERE study = 'kneeOA'")
            conn.commit()
            print("Existing kneeOA data deleted.")
    
    # Add group column to metadata
    try:
        cur.execute('ALTER TABLE metadata ADD COLUMN "group" TEXT')
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise
    cur.execute('UPDATE metadata SET "group" = ? WHERE study = ? AND ("group" IS NULL OR "group" = "")',
                ("control", "plosONE")) # make all plosONE subjects "control"
    conn.commit()
    # Insert metadata
    print(f"\nInserting {len(metadata_df)} kneeOA metadata records...")
    metadata_df.to_sql('metadata', conn, if_exists='append', index=False)
    
    # Insert trial data in chunks (it's a lot of data - 22M records!)
    chunk_size = 50000  # Larger chunks for efficiency
    total_chunks = (len(trial_data_df) // chunk_size) + 1
    
    print(f"Inserting {len(trial_data_df)} kneeOA trial data records in {total_chunks} chunks...")
    
    for i in range(0, len(trial_data_df), chunk_size):
        chunk_num = (i // chunk_size) + 1
        chunk = trial_data_df.iloc[i:i+chunk_size]
        
        print(f"  Inserting chunk {chunk_num}/{total_chunks} ({len(chunk)} records)...")
        chunk.to_sql('trial_data', conn, if_exists='append', index=False)
        
        # Commit every few chunks to avoid memory issues
        if chunk_num % 5 == 0:
            conn.commit()
            print(f"    Committed chunks 1-{chunk_num}")
    
    # Final commit
    conn.commit()
    
    # Verify insertion
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study = 'kneeOA'")
    inserted_metadata = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM trial_data WHERE study = 'kneeOA'")
    inserted_trial_data = cur.fetchone()[0]
    
    print(f"\n=== INSERTION COMPLETE ===")
    print(f"Inserted kneeOA metadata records: {inserted_metadata}")
    print(f"Inserted kneeOA trial data records: {inserted_trial_data}")
    
    # Show final database summary
    cur.execute("SELECT study, COUNT(*) FROM metadata GROUP BY study")
    metadata_summary = cur.fetchall()
    cur.execute("SELECT study, COUNT(*) FROM trial_data GROUP BY study")
    trial_data_summary = cur.fetchall()
    
    print(f"\n=== FINAL DATABASE SUMMARY ===")
    print("Metadata by study:")
    for study, count in metadata_summary:
        print(f"  {study}: {count:,} trials")
    
    print("Trial data by study:")
    for study, count in trial_data_summary:
        print(f"  {study}: {count:,} data points")
    
    conn.close()
    print("\nDatabase connection closed. Data insertion successful!")
    
except Exception as e:
    print(f"Error during database insertion: {e}")
    if 'conn' in locals():
        conn.close()
# %%
