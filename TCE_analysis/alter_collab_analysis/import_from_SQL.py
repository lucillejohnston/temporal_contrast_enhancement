"""
This script imports data from the SQL database back into a workable format for analysis
Updated 4/1/26 to extract the cLBP data as well

Do preprocessing.py next 
"""
#%% Imports and setup
#!/usr/bin/env python3
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Setup some things 
sql_path = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_data.sqlite'
save_dir = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/'
def print_tables_and_columns(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        print(f"Table: {table_name}")

        # Get columns for the table
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        for col in columns:
            cid, name, col_type, notnull, default_value, pk = col
            print(f"  Column: {name}, Type: {col_type}, NotNull: {bool(notnull)}, Default: {default_value}, PrimaryKey: {bool(pk)}")
        print("-" * 40)
    conn.close()

if __name__ == "__main__":
    print_tables_and_columns(sql_path)

#%% Extract data for specific study
def extract_study_data(sql_path, study_name):
    """
    Extract data for a specific study from the database and save as JSON
    """
    print(f"Extracting {study_name} data from database...")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(sql_path)
    
    # Modified query to filter by study and handle the kneeOA data structure
    query = '''
    SELECT 
        m.subject,
        m.trial_num,
        m.trial_date,
        m.study,
        COALESCE(NULLIF(m."group", ""), 'control') AS group_label,
        t.timestamp AS trial_timestamp,
        t.trial_type,
        t.temperature,
        t.pain,
        t.notes
    FROM metadata m
    JOIN trial_data t
        ON m.subject = t.subject 
        AND m.trial_num = t.trial_num 
        AND m.study = t.study
    WHERE m.study = ?
    ORDER BY m.subject, m.trial_num, t.timestamp
    '''
    
    # Load the data for the specific study
    df = pd.read_sql_query(query, conn, params=(study_name,))
    conn.close()
    
    print(f"Loaded {len(df)} records for {study_name}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Trials: {df['trial_num'].nunique()}")
    print(f"Trial types: {sorted(df['trial_type'].unique())}")
    
    # Handle different timestamp formats for different studies
    if study_name == 'plosONE':
        # plosONE has datetime timestamps
        df['trial_timestamp'] = pd.to_datetime(df['trial_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        if 'trial_date' in df.columns and not df['trial_date'].isna().all():
            df['trial_date'] = pd.to_datetime(df['trial_date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Add any plosONE-specific processing here
        
    elif study_name == 'kneeOA':
        # kneeOA has sequential timestamps (seconds from start)
        df['trial_timestamp'] = pd.to_numeric(df['trial_timestamp'], errors='coerce')
        
        # Parse site and spot information from notes
        df['site'] = df['notes'].str.extract(r'site=([^,]+)')
        df['spot'] = df['notes'].str.extract(r'spot=([^,\s]+)')
        
        # Create trial_order (1-based indexing per subject)
        df['trial_order'] = df.groupby('subject')['trial_num'].transform(lambda x: pd.factorize(x)[0] + 1)
    
    elif study_name in ['cLBP_DPOP', 'cLBP_MBPR']:
        # cLBP has numeric timestamps (seconds from start)
        df['trial_timestamp'] = pd.to_numeric(df['trial_timestamp'], errors='coerce')
        
        # Parse visit information from notes
        df['visit'] = df['notes'].str.extract(r'visit_(\d+)').astype(int)
        
        # Create trial_order (1-based indexing per subject)
        df['trial_order'] = df.groupby('subject')['trial_num'].transform(lambda x: pd.factorize(x)[0] + 1)
    return df


# # Extract plosONE data (rename your existing file)
# print("=== EXTRACTING PLOSONE DATA ===")
# plosone_df = extract_study_data(sql_path, 'plosONE')

# # Extract kneeOA data
# print("\n=== EXTRACTING KNEEOA DATA ===")
# kneeoa_df = extract_study_data(sql_path, 'kneeOA')

# Extract cLBP data
# Extract cLBP data - both studies
print("\n=== EXTRACTING CLBP DATA ===")
clbp_dpop_df = extract_study_data(sql_path, 'cLBP_DPOP')
clbp_mbpr_df = extract_study_data(sql_path, 'cLBP_MBPR')

# Combine them
clbp_df = pd.concat([clbp_dpop_df, clbp_mbpr_df], ignore_index=True)
print(f"Combined cLBP data: {len(clbp_df)} records")
print(f"Studies: {clbp_df['study'].value_counts()}")

#%% Check data quality
def build_timestamps(df, study_name):
    """Build relative_time and actual_time columns to match preprocessing.py expectations"""
    df = df.copy()
    df['trial_date'] = pd.to_datetime(df['trial_date'], errors='coerce')

    if study_name == 'plosONE':
        df['trial_timestamp'] = pd.to_datetime(df['trial_timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        df['relative_time'] = df['trial_timestamp'] - df['trial_date']
    elif study_name == 'kneeOA':
        df['trial_timestamp'] = pd.to_numeric(df['trial_timestamp'], errors='coerce')
        df['relative_time'] = pd.to_timedelta(df['trial_timestamp'], unit='s')
    elif study_name == 'cLBP':
        df['relative_time'] = pd.to_timedelta(df['trial_timestamp'], unit='s')
    df['relative_time_str'] = df['relative_time'].astype(str).str.replace(r'^0 days ', '', regex=True)
    df['actual_time'] = df['trial_date'] + df['relative_time']
    df['trial_order'] = df.groupby('subject')['trial_num'].transform(lambda x: pd.factorize(x)[0] + 1)
    df = df.drop(columns=['trial_timestamp']).sort_values(['subject', 'trial_num', 'actual_time']).reset_index(drop=True)
    return df

def check_data_quality(df, study_name):
    """Plot data quality diagnostics and print summary stats"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'{study_name} — Data Quality Check', fontsize=13)

    # 1) actual_time by row index
    axes[0].plot(df['actual_time'].values, '.', markersize=1, alpha=0.3)
    axes[0].set_xlabel('Row index')
    axes[0].set_ylabel('actual_time')
    axes[0].set_title('actual_time by row\n(should step up smoothly)')

    # 2) Sampling interval distribution (within trials only)
    diffs = df.groupby(['subject', 'trial_num'], group_keys=False).apply(
        lambda g: g['actual_time'].diff().dt.total_seconds().dropna()
    )
    axes[1].hist(diffs, bins=100, color='steelblue', edgecolor='none')
    axes[1].set_xlabel('Δt within trial (s)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Sampling interval\nmedian={diffs.median():.3f}s')

    # 3) Timepoints per trial
    counts = df.groupby(['subject', 'trial_num']).size()
    axes[2].hist(counts, bins=40, color='salmon', edgecolor='none')
    axes[2].set_xlabel('Timepoints per trial')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'Trial length\nmedian={counts.median():.0f} points')

    plt.tight_layout()
    plt.show()

    print(f"\n--- {study_name} Quality Summary ---")
    print(f"Total rows:                {len(df):,}")
    print(f"Subjects:                  {df['subject'].nunique()}")
    print(f"Trials:                    {df.groupby(['subject','trial_num']).ngroups}")
    print(f"NaT in actual_time:        {df['actual_time'].isna().sum()}")
    print(f"Zero Δt within trials:     {(diffs == 0).sum()}")
    print(f"Negative Δt within trials: {(diffs < 0).sum()}")
    print(f"Median sampling interval:  {diffs.median():.4f} s")

    return diffs

# Build timestamps and check quality
# plosone_df = build_timestamps(plosone_df, 'plosONE')
# kneeoa_df  = build_timestamps(kneeoa_df,  'kneeOA')
clbp_df    = build_timestamps(clbp_df,    'cLBP')

# plosone_diffs = check_data_quality(plosone_df, 'plosONE')
# kneeoa_diffs  = check_data_quality(kneeoa_df,  'kneeOA')
clbp_diffs    = check_data_quality(clbp_df,    'cLBP')


#%% Save cleaned extracted data as a JSON
# plosone_df.to_json(f'{save_dir}plosONE_trial_data.json', orient='records', date_format='iso')
# kneeoa_df.to_json(f'{save_dir}kneeOA_trial_data.json', orient='records', date_format='iso')
clbp_df.to_json(f'{save_dir}cLBP_trial_data.json', orient='records', date_format='iso')
print("Saved all datasets to JSON")


# %%
