"""
Extracts data from the sEEG studies (RCS and Presidio)
Organizes it into a database with 4 tables: metadata, threshold_data, calibration_data, trial_data

metadata: primary key is (subject, trial_num, study)
- subject (int)
- trial_num (int)
- study (text): 'plosONE', 'kneeOA', 'cLBP_DPOP', 'cLBP_MBPR', 'sEEG_RCS', 'sEEG_Presidio'
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

# Patients (updated 5/7/26)
pts = ['RCS08',
       'RCS09',
       'PR08'] 
# Loop through each patient and load in the data:
for pt in pts:
    if pt.startswith('RCS'):
            study = 'sEEG_RCS'
    elif pt.startswith('PR'):
        study = 'sEEG_Presidio'
    # Load in the _all_events.csv file for this patient
    path = f'/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/{pt}_data/'
    import glob
    csv_file = glob.glob(f'{path}{pt}*_all_events.csv')[0]
    events_df = pd.read_csv(csv_file)