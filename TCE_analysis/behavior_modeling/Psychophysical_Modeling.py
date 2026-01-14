# -*- coding: utf-8 -*-
"""
Fit the Cecchi 2012 ODE model
and testing with leave one trial out cross-validation
on temporal contrast enhancement data.

Author: Lucille Johnston
Updated: 1/13/26
"""
#%%
import pandas as pd
import sys, time, pickle, random, os
sys.path.append('/Users/ljohnston1/Desktop/Python/temporal_contrast_enhancement/')
from psychophysics_modeling_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.model_selection import LeaveOneOut
import pytensor.tensor as pt
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from datetime import datetime

DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/' # path for data files
FIG_PATH = '/Users/ljohnston1/Desktop/Python/TCE_Figures/' # path for saving figures

TRIAL_DATA = DATA_PATH + 'trial_data_trimmed_downsampled.json'
LIMITS_DATA = DATA_PATH + 'limits_data.csv'
# Load the raw trial data (for time series plotting)
data_df = pd.read_json(TRIAL_DATA, orient='records')
limits_data = pd.read_csv(LIMITS_DATA)
# Parameters from Petre 2017
petre_params = {
    'alpha': 2.4932,
    'beta': 36.7552,
    'gamma': 0.0204,
    'lambda_param': 0.0169,
    'theta': 37.1913,
}
#%%
# First, determine pain threshold (theta) for each subject
subject_thresholds_from_data = {}
for subject in data_df['subject'].unique():
    subject_data = data_df[data_df['subject'] == subject]
    try:
        threshold = extract_threshold_from_data(subject_data, vas_threshold=5)
        subject_thresholds_from_data[subject] = threshold
    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
        subject_thresholds_from_data[subject] = np.nan

subject_thresholds_from_limits = {}
for subject in data_df['subject'].unique():
    try:
        subject_limits = limits_data[limits_data['subj'] == subject]
        if not subject_limits.empty:
            limits = ['limits1','limits2','limits3']
            threshold = subject_limits[limits].mean(axis=1).values[0]
            subject_thresholds_from_limits[subject] = threshold
        else:
            subject_thresholds_from_limits[subject] = np.nan
            print(f"Subject {subject}: No limits data available")
    except Exception as e:
        print(f"Error processing limits for subject {subject}: {e}")
        subject_thresholds_from_limits[subject] = np.nan

# Compare the two methods
comparison_data = []
for subject in sorted(data_df['subject'].unique()):
    data_threshold = subject_thresholds_from_data.get(subject, np.nan)
    limits_threshold = subject_thresholds_from_limits.get(subject, np.nan)
    comparison_entry = {
        'subject': subject,
        'threshold_from_data': data_threshold,
        'threshold_from_limits': limits_threshold,
        'difference': limits_threshold - data_threshold
    }
    comparison_data.append(comparison_entry)
comparison_df = pd.DataFrame(comparison_data)

# Visualize the differences
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
# Histogram of thresholds from trial data
ax1.hist(comparison_df['threshold_from_data'], bins=15, alpha=0.7, color='blue', edgecolor='black')
ax1.set_xlabel('Threshold Temperature (°C)')
ax1.set_ylabel('Number of Subjects')
ax1.set_title('Thresholds from Trial Data')
ax1.grid(True, alpha=0.3)

# Histogram of thresholds from limits data
ax2.hist(comparison_df['threshold_from_limits'], bins=15, alpha=0.7, color='red', edgecolor='black')
ax2.set_xlabel('Threshold Temperature (°C)')
ax2.set_ylabel('Number of Subjects')
ax2.set_title('Thresholds from Limits Data')
ax2.grid(True, alpha=0.3)

# Scatter plot comparison
correlation = comparison_df['threshold_from_data'].corr(comparison_df['threshold_from_limits'])
ax3.scatter(comparison_df['threshold_from_data'], comparison_df['threshold_from_limits'], alpha=0.7)
min_val = min(comparison_df['threshold_from_data'].min(), comparison_df['threshold_from_limits'].min())
max_val = max(comparison_df['threshold_from_data'].max(), comparison_df['threshold_from_limits'].max())
ax3.set_xlabel('Threshold from Trial Data (°C)')
ax3.set_ylabel('Threshold from Limits Data (°C)')
ax3.set_title('Comparing Thresholds')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Difference histogram
ax4.hist(comparison_df['difference'], bins=15, alpha=0.7, color='purple', edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', label='No Difference')
ax4.axvline(x=comparison_df['difference'].mean(), color='b', linestyle='-', 
            label=f'Mean: {comparison_df["difference"].mean():+.2f}°C')
ax4.set_xlabel('Difference (Limits - Data) (°C)')
ax4.set_ylabel('Number of Subjects')
ax4.set_title('Distribution of Differences')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



#%% 
# Configuration for optimization approach
# 137 subjects maximum
N_SUBJECTS = 5 # Number of subjects to process
USE_MULTIPLE_STARTS = True  # Try multiple random starting points
N_STARTS = 5  # Number of random starts per subject
OPTIMIZE_THETA = False  # Use data-derived thresholds (set to True to optimize theta too)
# Output paths
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
CHECKPOINT_DIR = DATA_PATH
RESULTS_FILE = f"{CHECKPOINT_DIR}optimization_results_{timestamp}.pkl"

print(f"\n{'='*60}")
print(f"🚀 OPTIMIZATION CONFIGURATION")
print(f"{'='*60}")
print(f"Model: Simplified Cecchi 2012")
print(f"Subjects to process: {N_SUBJECTS}")
print(f"Multiple starts: {USE_MULTIPLE_STARTS} (n={N_STARTS})")
print(f"Optimize theta: {OPTIMIZE_THETA}")
print(f"Results will be saved to: {RESULTS_FILE}")
print(f"{'='*60}\n")

# Select subjects
subjects_all = sorted(data_df['subject'].unique())
N_SUBJECTS = min(N_SUBJECTS, len(subjects_all))
random.seed(42)
subjects_to_process = random.sample(subjects_all, k=N_SUBJECTS)
print(f"Selected {len(subjects_to_process)} subjects: {subjects_to_process}\n")

# Initialize results dictionary
optimization_results = {}

# Try to load existing checkpoint
if os.path.exists(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, 'rb') as f:
            checkpoint_data = pickle.load(f)
            optimization_results = checkpoint_data.get('optimization_results', {})
        print(f"📂 Loaded checkpoint with {len(optimization_results)} subjects already processed\n")
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}\n")



#%%
# # Run deep dive on individual subjects
# analyze_subject_data(33, data_df, detailed=True)

#%%
# Process each subject
total_start_time = time.time()
for subject_idx, subject in enumerate(subjects_to_process):
    # Skip if already processed
    if subject in optimization_results:
        print(f"⏭️  Subject {subject} already processed, skipping...\n")
        continue
    
    print(f"\n{'='*60}")
    print(f"📊 Processing subject {subject} ({subject_idx+1}/{len(subjects_to_process)})")
    print(f"{'='*60}")
    
    subject_start_time = time.time()
    
    # Get subject data
    subject_data = data_df[data_df['subject'] == subject]
    
    # Get threshold (use data-derived unless optimizing)
    if OPTIMIZE_THETA:
        threshold = None
    else:
        threshold = subject_thresholds_from_data.get(subject, np.nan)
        if np.isnan(threshold):
            print(f"⚠️  No threshold found for subject {subject}, skipping...")
            continue
        print(f"Using data-derived threshold: {threshold:.2f}°C")
    
    try:
        # Run optimization
        best_params, best_result = optimize_cecchi_simplified(
            subject_data,
            threshold=threshold,
            initial_params=None,  # Use Petre 2017 defaults
            use_multiple_starts=USE_MULTIPLE_STARTS,
            n_starts=N_STARTS,
            verbose=True
        )
        
        # Store results
        if best_params is not None:
            optimization_results[subject] = {
                'params': best_params,
                'result': best_result,
                'timestamp': datetime.now().isoformat()
            }
            
            subject_time = time.time() - subject_start_time
            print(f"\n✅ Subject {subject} completed in {subject_time:.1f}s")
            print(f"   Final MSE: {best_params['mse']:.2f}")
            print(f"   Parameters: α={best_params['alpha']:.4f}, β={best_params['beta']:.4f}, "
                  f"γ={best_params['gamma']:.4f}, λ={best_params['lambda_param']:.4f}")
            
            # Save checkpoint after each subject
            checkpoint_data = {
                'optimization_results': optimization_results,
                'config': {
                    'n_subjects': N_SUBJECTS,
                    'use_multiple_starts': USE_MULTIPLE_STARTS,
                    'n_starts': N_STARTS,
                    'optimize_theta': OPTIMIZE_THETA,
                    'timestamp': timestamp
                }
            }
            
            with open(RESULTS_FILE, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"   💾 Checkpoint saved")
            
        else:
            print(f"\n❌ Subject {subject} optimization failed")
            
    except Exception as e:
        print(f"\n❌ Error processing subject {subject}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Final summary
total_time = time.time() - total_start_time
print(f"\n{'='*60}")
print(f"📊 OPTIMIZATION COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Subjects processed: {len(optimization_results)}/{len(subjects_to_process)}")
print(f"Average time per subject: {total_time/max(len(optimization_results),1):.1f}s")
print(f"Results saved to: {RESULTS_FILE}")
print(f"{'='*60}\n")

#%% 
# Load optimization results and plot
with open(RESULTS_FILE, 'rb') as f:
    checkpoint_data = pickle.load(f)
    optimization_results = checkpoint_data['optimization_results']

# Plot a single subject with all details
plot_optimization_fit(33, optimization_results, 
                                 save_path=f"{FIG_PATH}subject_33_detailed.png")

# Create summary grid for all subjects
plot_multiple_subjects_summary(optimization_results, 
                               save_path=f"{FIG_PATH}all_subjects_summary.png")

# Print summary table
print_optimization_summary(optimization_results)

#%%