"""
Extracts behavioral data from the sEEG studies (RCS and Presidio)
and inserts it into the combined SQLite database.

Differences from kneeOA/plosONE/cLBP extraction:
- Source files are _all_events.csv (already preprocessed long-format output
  from the behavioral preprocessing pipeline — already quality-controlled)
- Long format has separate rows for 'temperature' and 'pain_rating' events
  at each timestamp → must be pivoted to one row per timepoint
- Subject IDs are assigned sequentially (1, 2, 3...) rather than pulled
  from a MATLAB struct

metadata table (primary key: subject, trial_num, study):
- subject (int)
- trial_num (int)
- trial_date (text)
- group (text): NULL for now — to be decided
- study (text): 'sEEG_RCS' or 'sEEG_Presidio'

trial_data table (primary key: subject, trial_num, timestamp):
- subject (int)
- trial_num (int)
- trial_type (text): step_down, step_up, hold_T1, hold_T2
- timestamp (text)
- temperature (real)
- pain (real)
- notes (text)
- study (text)
"""

### FROM CLAUDE HAVE TO REVIEW AND RUN
### 06/29/26
#%%
import pandas as pd
import numpy as np
import sqlite3
import glob

# ---------------------------
# Patient manifest
# Each patient gets a sequential integer subject ID.
# study is determined by patient prefix (RCS → sEEG_RCS, PR → sEEG_Presidio).
# group is left NULL for now — to be decided later.
# ---------------------------
patients = [
    {'pt': 'RCS08', 'subject': 1, 'study': 'sEEG_RCS',      'group': None},
    {'pt': 'RCS09', 'subject': 2, 'study': 'sEEG_RCS',      'group': None},
    {'pt': 'PR08',  'subject': 3, 'study': 'sEEG_Presidio', 'group': None},
]

data_base_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data'
sql_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_data.sqlite'

#%%
# ---------------------------
# Load and process each patient
# ---------------------------
metadata_list  = []
trial_data_list = []

for pt_info in patients:
    pt      = pt_info['pt']
    subject = pt_info['subject']
    study   = pt_info['study']
    group   = pt_info['group']

    print(f"\nProcessing {pt} (subject={subject}, study={study})...")

    # Find the _all_events.csv for this patient.
    # This is the cleaned long-format output from the behavioral preprocessing pipeline.
    pt_data_dir = f"{data_base_path}/{pt}_data"
    csv_files = glob.glob(f"{pt_data_dir}/{pt}_*_all_events.csv")

    if len(csv_files) == 0:
        print(f"  WARNING: No _all_events.csv found for {pt}, skipping.")
        continue
    if len(csv_files) > 1:
        print(f"  WARNING: Multiple _all_events.csv files found for {pt}, using first: {csv_files[0]}")
    csv_file = csv_files[0]

    # Extract the session date from the filename
    # e.g. RCS08_20250606_all_events.csv → '20250606'
    filename = csv_file.split('/')[-1]
    date_str = filename.replace(f'{pt}_', '').replace('_all_events.csv', '')

    print(f"  Loading: {filename}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows")

    # ---------------------------
    # Pivot long → wide format
    #
    # The _all_events.csv has separate rows for 'temperature' and 'pain_rating'
    # at the same timestamp and trial_count. The SQL trial_data table expects one
    # row per timepoint with both as columns (matching kneeOA/plosONE format).
    #
    # Strategy: separate the two event types, merge on timestamp + trial_count
    # + trial_type (outer so no timepoints are dropped), then forward-fill pain
    # within each trial to fill any gaps (matches kneeOA/plosONE handling).
    # ---------------------------
    temp_df = df[df['event'] == 'temperature'][['timestamp', 'trial_count', 'trial_type', 'value']].copy()
    pain_df = df[df['event'] == 'pain_rating'][['timestamp', 'trial_count', 'trial_type', 'value']].copy()

    temp_df = temp_df.rename(columns={'value': 'temperature'})
    pain_df = pain_df.rename(columns={'value': 'pain'})

    wide_df = pd.merge(
        temp_df, pain_df,
        on=['timestamp', 'trial_count', 'trial_type'],
        how='outer'
    )

    # Forward-fill pain within each trial (VAS slider holds last value)
    wide_df = wide_df.sort_values(['trial_count', 'timestamp'])
    wide_df['pain'] = wide_df.groupby('trial_count')['pain'].ffill()

    print(f"  Pivoted to {len(wide_df)} timepoint rows")
    print(f"  Trial types: {sorted(wide_df['trial_type'].dropna().unique())}")
    print(f"  Trial count range: {int(wide_df['trial_count'].min())} - {int(wide_df['trial_count'].max())}")

    # ---------------------------
    # Build metadata rows — one row per unique trial
    # trial_date is the session date extracted from the filename
    # ---------------------------
    unique_trials = wide_df[['trial_count', 'trial_type']].drop_duplicates()
    for _, row in unique_trials.iterrows():
        metadata_list.append({
            'subject':   subject,
            'trial_num': int(row['trial_count']),
            'trial_date': date_str,
            'group':     group,
            'study':     study,
        })

    # ---------------------------
    # Build trial_data rows
    # ---------------------------
    for _, row in wide_df.iterrows():
        trial_data_list.append({
            'subject':     subject,
            'trial_num':   int(row['trial_count']),
            'trial_type':  row['trial_type'],
            'timestamp':   str(row['timestamp']),
            'temperature': float(row['temperature']) if pd.notna(row['temperature']) else np.nan,
            'pain':        float(row['pain'])        if pd.notna(row['pain'])        else np.nan,
            'notes':       '',
            'study':       study,
        })

print(f"\n=== EXTRACTION COMPLETE ===")
print(f"Metadata records: {len(metadata_list)}")
print(f"Trial data records: {len(trial_data_list)}")

#%%
# ---------------------------
# Create DataFrames and clean
# (parallel to kneeOA extraction)
# ---------------------------
metadata_df    = pd.DataFrame(metadata_list)
trial_data_df  = pd.DataFrame(trial_data_list)

print(f"\n=== CLEANING DATA ===")
print(f"Original trial data records: {len(trial_data_df)}")

# Remove rows missing both temperature and pain
cleaned_trial_data = trial_data_df.dropna(subset=['temperature', 'pain'])
print(f"After removing rows missing temp and pain: {len(cleaned_trial_data)}")

# Temperature range filter (same bounds as kneeOA extraction)
cleaned_trial_data = cleaned_trial_data[
    (cleaned_trial_data['temperature'] >= 0) &
    (cleaned_trial_data['temperature'] <= 60)
]
print(f"After temperature filtering (0-60°C): {len(cleaned_trial_data)}")

# Pain range filter (same bounds as kneeOA extraction)
cleaned_trial_data = cleaned_trial_data[
    (cleaned_trial_data['pain'] >= -2) &
    (cleaned_trial_data['pain'] <= 102)
]
print(f"After pain filtering (-2 to 102): {len(cleaned_trial_data)}")

# Update metadata to only include trials that survived cleaning
valid_trials     = cleaned_trial_data[['subject', 'trial_num', 'study']].drop_duplicates()
cleaned_metadata = metadata_df.merge(valid_trials, on=['subject', 'trial_num', 'study'])

print(f"\n=== FINAL CLEAN DATA SUMMARY ===")
print(f"Clean metadata records: {len(cleaned_metadata)}")
print(f"Clean trial data records: {len(cleaned_trial_data)}")
print(f"Subjects: {sorted(cleaned_metadata['subject'].unique())}")
print(f"Temperature range: {cleaned_trial_data['temperature'].min():.2f} to {cleaned_trial_data['temperature'].max():.2f}")
print(f"Pain range: {cleaned_trial_data['pain'].min():.2f} to {cleaned_trial_data['pain'].max():.2f}")
print(f"Unique trial types: {sorted(cleaned_trial_data['trial_type'].unique())}")
print(f"Trials per subject:")
print(cleaned_metadata.groupby('subject')['trial_num'].count())

trial_data_df = cleaned_trial_data
metadata_df   = cleaned_metadata

#%%
# ---------------------------
# Insert into SQLite database
# (parallel to kneeOA extraction)
# ---------------------------
print(f"\n=== INSERTING DATA INTO SQL DATABASE ===")
print(f"Database path: {sql_path}")

try:
    conn = sqlite3.connect(sql_path)
    cur  = conn.cursor()

    # Show what's already in the database for context
    for study_name in ['sEEG_RCS', 'sEEG_Presidio']:
        cur.execute("SELECT COUNT(*) FROM metadata WHERE study = ?", (study_name,))
        print(f"  Existing {study_name} metadata records: {cur.fetchone()[0]}")

    # Check if sEEG data already exists and offer options (same pattern as kneeOA)
    cur.execute("SELECT COUNT(*) FROM metadata WHERE study IN ('sEEG_RCS', 'sEEG_Presidio')")
    existing_count = cur.fetchone()[0]

    if existing_count > 0:
        print(f"\nWARNING: Found {existing_count} existing sEEG records. Do you want to:")
        print("1. Skip insertion (data already exists)")
        print("2. Delete existing sEEG data and insert new data")
        print("3. Proceed anyway (may create duplicates)")
        choice = input("Enter choice (1, 2, or 3): ")

        if choice == '1':
            print("Skipping insertion.")
            conn.close()
            exit()
        elif choice == '2':
            print("Deleting existing sEEG data...")
            cur.execute("DELETE FROM metadata WHERE study IN ('sEEG_RCS', 'sEEG_Presidio')")
            cur.execute("DELETE FROM trial_data WHERE study IN ('sEEG_RCS', 'sEEG_Presidio')")
            conn.commit()
            print("Existing sEEG data deleted.")

    # Ensure the 'group' column exists in metadata (added during kneeOA extraction)
    try:
        cur.execute('ALTER TABLE metadata ADD COLUMN "group" TEXT')
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise

    # Insert metadata
    print(f"\nInserting {len(metadata_df)} sEEG metadata records...")
    metadata_df.to_sql('metadata', conn, if_exists='append', index=False)

    # Insert trial data in chunks (same pattern as kneeOA)
    chunk_size   = 50000
    total_chunks = (len(trial_data_df) // chunk_size) + 1
    print(f"Inserting {len(trial_data_df)} sEEG trial data records in {total_chunks} chunk(s)...")

    for i in range(0, len(trial_data_df), chunk_size):
        chunk_num = (i // chunk_size) + 1
        chunk = trial_data_df.iloc[i:i + chunk_size]
        print(f"  Inserting chunk {chunk_num}/{total_chunks} ({len(chunk)} records)...")
        chunk.to_sql('trial_data', conn, if_exists='append', index=False)
        if chunk_num % 5 == 0:
            conn.commit()

    conn.commit()

    # Verify insertion
    print(f"\n=== INSERTION COMPLETE ===")
    for study_name in ['sEEG_RCS', 'sEEG_Presidio']:
        cur.execute("SELECT COUNT(*) FROM metadata WHERE study = ?", (study_name,))
        n_meta = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM trial_data WHERE study = ?", (study_name,))
        n_data = cur.fetchone()[0]
        print(f"  {study_name}: {n_meta} metadata records, {n_data} trial data records inserted")

    # Final database summary across all studies
    print(f"\n=== FINAL DATABASE SUMMARY ===")
    print("Metadata by study:")
    cur.execute("SELECT study, COUNT(*) FROM metadata GROUP BY study")
    for study, count in cur.fetchall():
        print(f"  {study}: {count:,} trials")
    print("Trial data by study:")
    cur.execute("SELECT study, COUNT(*) FROM trial_data GROUP BY study")
    for study, count in cur.fetchall():
        print(f"  {study}: {count:,} data points")

    conn.close()
    print("\nDatabase connection closed. Data insertion successful!")

except Exception as e:
    print(f"Error during database insertion: {e}")
    if 'conn' in locals():
        conn.close()

# %%
