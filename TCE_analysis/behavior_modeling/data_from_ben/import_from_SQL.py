import sqlite3

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
    db_file = "combined_data.sqlite"
    print_tables_and_columns(db_file)

#%%
#!/usr/bin/env python3
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = '/userdata/ljohnston/TCE_analysis/data_from_ben/combined_data.sqlite'
# Connect to the SQLite database.
conn = sqlite3.connect(db_path)
# Join metadata and trial_data on subject and trial_num.
query = '''
SELECT 
    m.subject,
    m.trial_num,
    m.trial_date,
    t.timestamp AS trial_timestamp,
    t.trial_type,
    t.temperature,
    t.pain,
    t.notes
FROM metadata m
JOIN trial_data t
    ON m.subject = t.subject AND m.trial_num = t.trial_num
'''

# Load the joined data into a DataFrame.
df = pd.read_sql_query(query, conn)
conn.close()
#%%
df['trial_timestamp'] = pd.to_datetime(df['trial_timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
df['relative_time'] = pd.to_timedelta(df['relative_time'])
df['relative_time_str'] = df['relative_time'].astype(str).str.replace(r'^0 days ', '', regex=True)
df['trial_date'] = pd.to_datetime(df['trial_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
df['actual_time'] = df['trial_date'] + df['relative_time']
df = df.sort_values('actual_time')
# Create a trial_order column with 1-based indexing per subject.
df['trial_order'] = df.groupby('subject')['trial_num']\
                      .transform(lambda x: pd.factorize(x)[0] + 1)
if 'trial_timestamp' in df.columns: # drop trial_timestamp because it isn't needed 
    trial_data = df.drop(columns=['trial_timestamp'])
    
# Save the DataFrame as a JSON file to use later
df.to_json('trial_data.json', orient='records', date_format='iso')