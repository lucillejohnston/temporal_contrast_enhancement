#%%
"""
Section 1: Load the data and import relevant packages
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os
import scipy.stats as stats
sys.path.append('/userdata/ljohnston/TCE_analysis/data_from_ben')
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_metrics.csv'
TRIAL_DATA_PATH = '/userdata/ljohnston/TCE_analysis/data_from_ben/trial_data_cleaned_aligned.json'

# Load the metrics data
trial_metrics_df = pd.read_csv(TRIAL_METRICS_PATH)

# Load the raw trial data (for time series plotting)
df = pd.read_json(TRIAL_DATA_PATH, orient='records')
#%%
