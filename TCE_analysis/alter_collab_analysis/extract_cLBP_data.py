"""
Extracts data from the Irina's cLBP study
Organizes it into a database with 4 tables: metadata, threshold_data, calibration_data, trial_data

metadata: primary key is (subject, trial_num, study)
- subject (int)
- trial_num (int)
- study (text): 'plosONE', 'kneeOA', 'cLBP_DPOP', 'cLBP_MBPR'
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
import matplotlib.pyplot as plt

# Load in the DPOP and MBPR data
data_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/cLBP_raw_extracted.mat'
raw_data = scipy.io.loadmat(data_path)
info_path = '/Users/ljohnston1/UCSF DBS for Pain Dropbox/PainNeuromodulationLab/DATA ANALYSIS/Lucy/BenAlter_Collab_Data/cLBP_Strigo/MYP_DPOP_MBPR_DATA_09062023.csv'
info_df = pd.read_csv(info_path)
# Extract the subjects data
extracted_data = raw_data['extracted_data']
subjects_data = extracted_data['subjects'][0, 0]
#%% Extract group labels from lowbackpainint 
lbp_scores = pd.to_numeric(info_df['lowbackpainint'], errors='coerce')
# Create a histogram of lbp_scores
plt.figure(figsize=(10, 6))
plt.hist(lbp_scores, bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Low Back Pain Intensity Scores', fontsize=16)
plt.xlabel('Low Back Pain Intensity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
# add lines for mild/moderate/severe cutoffs
median_score = lbp_scores.median()
plt.axvline(x=median_score, color='red', linestyle='--', label=f'Mild/Moderate cutoff (median={median_score:.1f})')
plt.axvline
plt.show()

#%% Process all subjects with visit-aware trial numbering
# Basically make second visit trials 101-112 instead of 1-12, so we can keep them in the same table without duplicates and easily identify them later
print(f"\nProcessing all {len(subjects_data)} sessions...")
metadata_list = []
trial_data_list = []

trial_type_mapping = {
    '3step': 'offset',
    'T1': 't1_hold',
    'T2': 't2_hold', 
    'Inv': 'onset'
}

for subj_idx, subj in enumerate(subjects_data):
    subj = subjects_data[subj_idx, 0]
    
    # Extract subject info
    subject_id = int(subj['subject_id'][0, 0])
    study = subj['study'][0]
    if subj['visit'].size == 0 or np.isnan(subj['visit'][0, 0]):
        visit = 0  # Assume visit 0 if missing/NaN
    else:
        visit = int(subj['visit'][0, 0])

    # Extract trial boundaries
    trial_boundaries = subj['trial_boundaries']  # Nx2 array of [start_time, end_time]
    smoothtemp = subj['smoothtemp'].flatten()
    smoothtime = subj['smoothtime'].flatten() / 1000.0  # Convert ms to seconds
    pain_times = subj['pain_times'].flatten() / 1000.0  # Convert ms to seconds
    pain_values = subj['pain_values'].flatten()
    stim_sequence = [str(s[0]) for s in subj['stim_sequence'][0]] if subj['stim_sequence'].size > 0 else []
    
    # Adjust pain_times that align with trial boundaries
    adjustment = 0.001  # Adjust by 1 millisecond (in seconds)

    # Iterate through trials
    for trial_num, (start_time, end_time) in enumerate(trial_boundaries, start=1):
        # Identify pain_times that align with trial boundaries
        boundary_mask = (pain_times == start_time / 1000.0) | (pain_times == end_time / 1000.0)
        if len(pain_times) != len(pain_values):
            print(f"Warning: pain_times and pain_values length mismatch for subject {subject_id}, trial {trial_num}. Skipping boundary adjustment.")
        else:
            # Adjust boundary-aligned pain_times
            pain_times[boundary_mask] += adjustment

        # Mask data for this trial
        pain_mask = (pain_times >= start_time / 1000.0) & (pain_times <= end_time / 1000.0)
        trial_pain_times = pain_times[pain_mask] - (start_time / 1000.0)  # Relative to trial start
        trial_pain_values = pain_values[pain_mask]
        
        # Extract temperature data for this trial
        temp_mask = (smoothtime >= start_time / 1000.0) & (smoothtime <= end_time / 1000.0)
        trial_temp_times = smoothtime[temp_mask] - (start_time / 1000.0)  # Relative to trial start
        trial_temp_values = smoothtemp[temp_mask]

        # Adjust trial number for visit
        adjusted_trial_num = trial_num + (visit * 100)

        # Get trial type
        trial_type = stim_sequence[trial_num - 1] if trial_num <= len(stim_sequence) else 'unknown'

        # Add metadata
        metadata_list.append({
            'subject': subject_id,
            'trial_num': adjusted_trial_num,
            'trial_date': subj['session_start_time'][0] if 'session_start_time' in subj.dtype.names else 'unknown',
            'group': 'cLBP',
            'study': study
        })

        # Align pain and temp
        for i, temp_time in enumerate(trial_temp_times):
            # Find closest pain time to this temp time
            pain_value = np.nan
            if len(trial_pain_times) > 0:
                time_diffs = np.abs(trial_pain_times - temp_time)
                closest_idx = np.argmin(time_diffs)
                if time_diffs[closest_idx] <= 0.1:  # If within 100ms, consider it a match
                    pain_value = trial_pain_values[closest_idx]
            trial_data_list.append({
                'subject': subject_id,
                'trial_num': adjusted_trial_num,
                'trial_type': trial_type,
                'timestamp': float(temp_time),
                'temperature': float(trial_temp_values[i]),
                'pain': float(pain_value) if not np.isnan(pain_value) else np.nan,
                'notes': f'visit_{visit}',
                'study': study
            })

print(f"\n=== EXTRACTION COMPLETE ===")
print(f"Metadata records: {len(metadata_list)}")
print(f"Trial data records: {len(trial_data_list)}")


# %% Check that worked as expected before saving to SQL
# Create DataFrames first
metadata_df = pd.DataFrame(metadata_list)
trial_data_df = pd.DataFrame(trial_data_list)

print("=== VERIFICATION ===")

# Check unique studies
print(f"Unique studies: {metadata_df['study'].value_counts()}")
# Check trial number ranges to verify visit encoding
print(f"\nTrial number ranges:")
print(f"Min trial_num: {metadata_df['trial_num'].min()}")
print(f"Max trial_num: {metadata_df['trial_num'].max()}")
print(f"Unique trial numbers: {sorted(metadata_df['trial_num'].unique())}")

# Verify all trial types are now valid
print(f"\nAll trial types: {sorted(trial_data_df['trial_type'].unique())}")
# Check subjects with multiple visits
subjects_with_multiple_visits = []
for subject in metadata_df['subject'].unique():
    subject_trials = metadata_df[metadata_df['subject'] == subject]['trial_num'].values
    has_visit_0 = any(t <= 12 for t in subject_trials)
    has_visit_1 = any(t > 100 for t in subject_trials)
    if has_visit_0 and has_visit_1:
        subjects_with_multiple_visits.append(subject)

# Print total number of datapoints
print(f"\nTotal trial data records: {len(trial_data_df)}")
print(f"Subjects with multiple visits: {len(subjects_with_multiple_visits)}")
print(f"Metadata summary:")
print(metadata_df['study'].value_counts())
print(f"Trial data summary:")
print(trial_data_df['study'].value_counts())

#%% Clean the cLBP data before SQL insertion
print("\n=== CLEANING cLBP DATA ===")
print(f"Original metadata records: {len(metadata_df)}")
print(f"Original trial data records: {len(trial_data_df)}")

# Remove rows with missing values in critical fields
cleaned_trial_data = trial_data_df.dropna(subset=['temperature', 'timestamp'])
print(f"After removing NaN in temperature/timestamp: {len(cleaned_trial_data)}")

# Remove rows with extreme temperatures (reasonable range for thermal stimuli)
cleaned_trial_data = cleaned_trial_data[
    (cleaned_trial_data['temperature'] >= 0) & 
    (cleaned_trial_data['temperature'] <= 60)
]
print(f"After temperature filtering (0-60°C): {len(cleaned_trial_data)}")

# Remove rows with invalid pain values (keep NaN pain values - they're expected with finger scale)
cleaned_trial_data = cleaned_trial_data[
    (cleaned_trial_data['pain'].isna()) |  # Keep NaN values
    ((cleaned_trial_data['pain'] >= -2) & (cleaned_trial_data['pain'] <= 102))
]
print(f"After pain filtering (-2 to 102, keeping NaN): {len(cleaned_trial_data)}")

# Update metadata to only include trials that have valid data
valid_trials = cleaned_trial_data[['subject', 'trial_num', 'study']].drop_duplicates()
cleaned_metadata = metadata_df.merge(valid_trials, on=['subject', 'trial_num', 'study'])

print(f"\n=== FINAL CLEAN DATA SUMMARY ===")
print(f"Clean metadata records: {len(cleaned_metadata)}")
print(f"Clean trial data records: {len(cleaned_trial_data)}")
print(f"Valid subjects: {len(cleaned_metadata['subject'].unique())}")
print(f"Temperature range: {cleaned_trial_data['temperature'].min():.2f} to {cleaned_trial_data['temperature'].max():.2f}")

# Pain coverage (excluding NaN)
pain_coverage = (~cleaned_trial_data['pain'].isna()).sum() / len(cleaned_trial_data) * 100
print(f"Pain coverage: {pain_coverage:.1f}% ({(~cleaned_trial_data['pain'].isna()).sum():,}/{len(cleaned_trial_data):,} timepoints)")

pain_range = cleaned_trial_data['pain'].dropna()
if len(pain_range) > 0:
    print(f"Pain range: {pain_range.min():.2f} to {pain_range.max():.2f}")

print(f"\nUnique trial types: {sorted(cleaned_trial_data['trial_type'].unique())}")
print(f"Study breakdown:")
print(cleaned_metadata['study'].value_counts())

# Replace with cleaned data
trial_data_df = cleaned_trial_data
metadata_df = cleaned_metadata

#%% Connect to the SQLite database
sql_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_data.sqlite'
print(f"\n=== INSERTING cLBP DATA INTO SQL DATABASE ===")
print(f"Database path: {sql_path}")

try:
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()
    
    # Check current database contents
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study = 'plosONE'")
    plos_metadata_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study = 'kneeOA'")
    knee_metadata_count = cur.fetchone()[0]
    
    print(f"Current database contents:")
    print(f"  plosONE metadata records: {plos_metadata_count}")
    print(f"  kneeOA metadata records: {knee_metadata_count}")
    
    # Check if cLBP data already exists
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study LIKE 'cLBP%'")
    existing_clbp = cur.fetchone()[0]
    
    if existing_clbp > 0:
        print(f"WARNING: Found {existing_clbp} existing cLBP records. Do you want to:")
        print("1. Skip insertion (data already exists)")
        print("2. Delete existing cLBP data and insert new data")
        print("3. Proceed anyway (may create duplicates)")
        
        choice = input("Enter choice (1, 2, or 3): ")
        
        if choice == '1':
            print("Skipping insertion.")
            conn.close()
            exit()
        elif choice == '2':
            print("Deleting existing cLBP data...")
            cur.execute("DELETE FROM metadata WHERE study LIKE 'cLBP%'")
            cur.execute("DELETE FROM trial_data WHERE study LIKE 'cLBP%'")
            conn.commit()
            print("Existing cLBP data deleted.")
    
    # Ensure group column exists in metadata table
    try:
        cur.execute('ALTER TABLE metadata ADD COLUMN "group" TEXT')
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise
    
    # Insert metadata
    print(f"\nInserting {len(metadata_df)} cLBP metadata records...")
    metadata_df.to_sql('metadata', conn, if_exists='append', index=False)
    
    # Insert trial data in chunks (3.7M records is a lot!)
    chunk_size = 50000  # Same as kneeOA
    total_chunks = (len(trial_data_df) // chunk_size) + 1
    
    print(f"Inserting {len(trial_data_df)} cLBP trial data records in {total_chunks} chunks...")
    
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
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study LIKE 'cLBP%'")
    inserted_metadata = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM trial_data WHERE study LIKE 'cLBP%'")
    inserted_trial_data = cur.fetchone()[0]
    
    print(f"\n=== INSERTION COMPLETE ===")
    print(f"Inserted cLBP metadata records: {inserted_metadata}")
    print(f"Inserted cLBP trial data records: {inserted_trial_data}")
    
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
    
    # Show breakdown by cLBP study
    cur.execute("SELECT study, COUNT(*) FROM metadata WHERE study LIKE 'cLBP%' GROUP BY study")
    clbp_breakdown = cur.fetchall()
    print(f"\ncLBP breakdown:")
    for study, count in clbp_breakdown:
        print(f"  {study}: {count:,} trials")
    
    # Show visit breakdown (trial numbers 1-12 vs 101-112)
    cur.execute("""
        SELECT 
            CASE 
                WHEN trial_num <= 12 THEN 'Visit 0' 
                WHEN trial_num > 100 THEN 'Visit 1' 
                ELSE 'Other'
            END as visit,
            COUNT(*) 
        FROM metadata 
        WHERE study LIKE 'cLBP%' 
        GROUP BY 
            CASE 
                WHEN trial_num <= 12 THEN 'Visit 0' 
                WHEN trial_num > 100 THEN 'Visit 1' 
                ELSE 'Other'
            END
    """)
    visit_breakdown = cur.fetchall()
    print(f"\nVisit breakdown:")
    for visit, count in visit_breakdown:
        print(f"  {visit}: {count:,} trials")
    
    conn.close()
    print("\nDatabase connection closed. cLBP data insertion successful!")
    
except Exception as e:
    print(f"Error during database insertion: {e}")
    if 'conn' in locals():
        conn.close()
    raise

print(f"\n🎉 cLBP data successfully added to combined database!")
print(f"   - Studies: cLBP_DPOP, cLBP_MBPR") 
print(f"   - Visit encoding: trials 1-12 (visit 0), trials 101-112 (visit 1)")
print(f"   - Pain coverage: ~{pain_coverage:.0f}% (finger scale sampled at ~1Hz)")
# %%
