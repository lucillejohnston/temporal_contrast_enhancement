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
print(f"Model: Full Cecchi 2012")
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


#%%
def estimate_subject_threshold_robust(subject_data, method='median'):
    """More robust threshold estimation methods"""
    
    print(f"\n🎯 ROBUST THRESHOLD ESTIMATION")
    print(f"{'='*40}")
    
    # Method 1: Use multiple VAS thresholds and take median
    vas_thresholds = [3, 5, 7, 10]
    all_thresholds = []
    
    for vas_thresh in vas_thresholds:
        trial_thetas = []
        for trial in subject_data['trial_num'].unique():
            trial_data = subject_data[subject_data['trial_num'] == trial].copy()
            trial_data = trial_data.sort_values('aligned_time').reset_index(drop=True)
            
            pain_above = trial_data[trial_data['pain'] >= vas_thresh]
            if not pain_above.empty:
                trial_thetas.append(pain_above.iloc[0]['temperature'])
        
        if trial_thetas:
            median_thresh = np.median(trial_thetas)
            all_thresholds.extend(trial_thetas)
            print(f"VAS≥{vas_thresh}: {len(trial_thetas)} trials, median threshold = {median_thresh:.2f}°C")
    
    if method == 'median':
        robust_threshold = np.median(all_thresholds)
    elif method == 'percentile_25':
        robust_threshold = np.percentile(all_thresholds, 25)  # More sensitive
    elif method == 'mode':
        # Find most common threshold (rounded to nearest 0.5°C)
        rounded_thresholds = np.round(np.array(all_thresholds) * 2) / 2
        from scipy import stats
        robust_threshold = stats.mode(rounded_thresholds)[0][0]
    
    print(f"\n🎯 Robust threshold ({method}): {robust_threshold:.2f}°C")
    print(f"   Original mean method: {np.mean(all_thresholds):.2f}°C")
    print(f"   Difference: {robust_threshold - np.mean(all_thresholds):.2f}°C")
    
    return robust_threshold


# Test the simplified model approach from the paper
def test_simplified_model_approach(subject_data):
    """Test using the simplified first-order model from Cecchi 2012"""
    
    print(f"\n🎯 TESTING SIMPLIFIED MODEL (First-Order)")
    print(f"{'='*50}")
    
    # From the paper: p'(t) = ᾱF(T,θ) - c̄p(t)
    # Where ᾱ = α/β and c̄ = cλ/β
    
    robust_threshold = estimate_subject_threshold_robust(subject_data, method='percentile_25')
    
    # Calculate simplified parameters from full model parameters
    original_alpha = 2.4932
    original_beta = 36.7552
    original_gamma = 0.0204
    original_lambda = 0.0169
    
    alpha_bar = original_alpha / original_beta  # ᾱ = α/β
    c_bar = original_gamma * original_lambda / original_beta  # c̄ = γλ/β
    
    print(f"Derived simplified parameters:")
    print(f"   ᾱ = α/β = {original_alpha:.4f}/{original_beta:.2f} = {alpha_bar:.6f}")
    print(f"   c̄ = γλ/β = {original_gamma:.4f}×{original_lambda:.4f}/{original_beta:.2f} = {c_bar:.8f}")
    print(f"   θ = {robust_threshold:.2f}°C")
    
    # Test with scaled versions
    simplified_params = [
        {
            'name': 'Original Simplified',
            'alpha_bar': alpha_bar,
            'gamma_bar': c_bar,
            'theta': robust_threshold
        },
        {
            'name': 'Scaled Simplified (10x)',
            'alpha_bar': alpha_bar * 10,
            'gamma_bar': c_bar,
            'theta': robust_threshold
        },
        {
            'name': 'Scaled Simplified (100x)',
            'alpha_bar': alpha_bar * 100,
            'gamma_bar': c_bar,
            'theta': robust_threshold
        }
    ]
    
    time_data, temp_data, pain_data, _, _ = prepare_data_for_optimization(subject_data)
    temp_func = interp1d(time_data, temp_data, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    for params in simplified_params:
        print(f"\n🧪 Testing: {params['name']}")
        print(f"   ᾱ={params['alpha_bar']:.6f}, c̄={params['gamma_bar']:.8f}")
        
        # Test integration with simplified model
        t_span = (time_data[0], time_data[-1])
        y0 = [0.0]  # Only need initial pain for first-order
        
        try:
            sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                           args=(params, temp_func),
                           t_eval=time_data[:400], method='RK45',
                           rtol=1e-2, atol=1e-4)
            
            if sol.success:
                model_pain = np.maximum(sol.y[0], 0.0)
                mse = np.mean((pain_data[:400] - model_pain) ** 2)
                
                print(f"   ✅ Model pain range: {model_pain.min():.1f} - {model_pain.max():.1f}")
                print(f"   📊 Observed range: {pain_data[:400].min():.1f} - {pain_data[:400].max():.1f}")
                print(f"   📈 MSE: {mse:.1f}")
                
                if model_pain.max() > 10:
                    print(f"   🎉 Simplified model SUCCESS!")
                    
            else:
                print(f"   ❌ Integration failed: {sol.message}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")

# Run both tests
print("Testing scaled full model...")
successful_full = test_scaled_parameters(subject_data)

print("\n" + "="*60)
print("Testing simplified model...")
test_simplified_model_approach(subject_data)


# %%
def optimize_simplified_parameters(subject_data):
    """Find better parameters for the simplified model"""
    
    print(f"\n🎯 OPTIMIZING SIMPLIFIED MODEL PARAMETERS")
    print(f"{'='*60}")
    
    time_data, temp_data, pain_data, _, _ = prepare_data_for_optimization(subject_data)
    temp_func = interp1d(time_data, temp_data, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    
    # Test different parameter combinations
    alpha_bar_values = [0.1, 0.2, 0.5, 1.0, 2.0]  # Force scaling
    gamma_bar_values = [0.005, 0.01, 0.02, 0.05, 0.1]  # Decay rate
    
    best_params = None
    best_score = float('inf')
    results = []
    
    for alpha_bar in alpha_bar_values:
        for gamma_bar in gamma_bar_values:
            params = {
                'alpha_bar': alpha_bar,
                'gamma_bar': gamma_bar,
                'theta': 39.31
            }
            
            # Test integration
            t_span = (time_data[0], time_data[-1])
            y0 = [0.0]
            
            try:
                sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                               args=(params, temp_func),
                               t_eval=time_data[:400], method='RK45',
                               rtol=1e-2, atol=1e-4)
                
                if sol.success:
                    model_pain = np.maximum(sol.y[0], 0.0)
                    section_pain = pain_data[:400]
                    
                    # Calculate fit quality
                    mse = np.mean((section_pain - model_pain) ** 2)
                    max_model = model_pain.max()
                    max_observed = section_pain.max()
                    
                    # Penalty for being too far from observed range
                    range_penalty = abs(max_model - max_observed) * 10
                    
                    # Check decay behavior
                    above_threshold = temp_data[:400] > params['theta']
                    transitions = np.diff(above_threshold.astype(int))
                    fall_transitions = np.where(transitions == -1)[0]
                    
                    decay_score = 0
                    if len(fall_transitions) > 0:
                        fall_idx = fall_transitions[0]
                        if fall_idx + 20 < len(model_pain):
                            pain_before = model_pain[fall_idx]
                            pain_after = model_pain[fall_idx + 20]
                            decay_ratio = pain_after / pain_before if pain_before > 0 else 1.0
                            
                            # Penalty for poor decay (want ratio around 0.5-0.8)
                            if decay_ratio > 0.9:
                                decay_score = (decay_ratio - 0.9) * 1000  # Heavy penalty for no decay
                            elif decay_ratio < 0.3:
                                decay_score = (0.3 - decay_ratio) * 500   # Penalty for too fast decay
                            else:
                                decay_score = 0  # Good decay
                    
                    total_score = mse + range_penalty + decay_score
                    
                    results.append({
                        'alpha_bar': alpha_bar,
                        'gamma_bar': gamma_bar,
                        'mse': mse,
                        'max_model': max_model,
                        'max_observed': max_observed,
                        'decay_ratio': decay_ratio if len(fall_transitions) > 0 else 1.0,
                        'total_score': total_score
                    })
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_params = params.copy()
                        
                    print(f"ᾱ={alpha_bar:4.1f}, c̄={gamma_bar:5.3f}: max={max_model:5.1f}, decay={decay_ratio:.3f}, score={total_score:8.1f}")
                
            except Exception as e:
                print(f"ᾱ={alpha_bar:4.1f}, c̄={gamma_bar:5.3f}: ERROR - {e}")
    
    # Show best results
    print(f"\n🏆 BEST PARAMETERS:")
    if best_params:
        print(f"   ᾱ = {best_params['alpha_bar']:.3f}")
        print(f"   c̄ = {best_params['gamma_bar']:.3f}")
        print(f"   Decay time constant: {1/best_params['gamma_bar']:.1f}s")
        print(f"   Best score: {best_score:.1f}")
        
        # Test the best parameters
        sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                       args=(best_params, temp_func),
                       t_eval=time_data[:400], method='RK45',
                       rtol=1e-2, atol=1e-4)
        
        if sol.success:
            model_pain = np.maximum(sol.y[0], 0.0)
            section_time = time_data[:400]
            section_temp = temp_data[:400]
            section_pain = pain_data[:400]
            
            # Plot the best result
            plt.figure(figsize=(14, 10))
            
            plt.subplot(4,1,1)
            plt.plot(section_time, section_temp, 'orange', linewidth=2, label='Temperature')
            plt.axhline(best_params['theta'], color='red', linestyle='--', label='Threshold')
            plt.fill_between(section_time, best_params['theta'], section_temp,
                            where=(section_temp > best_params['theta']), alpha=0.3, color='red')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Best Simplified Model: ᾱ={best_params["alpha_bar"]:.3f}, c̄={best_params["gamma_bar"]:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4,1,2)
            force_term = np.maximum(section_temp - best_params['theta'], 0) * best_params['alpha_bar']
            decay_term = model_pain * best_params['gamma_bar']
            net_rate = force_term - decay_term
            plt.plot(section_time, force_term, 'g-', linewidth=2, label='ᾱF(T,θ) (driving)')
            plt.plot(section_time, decay_term, 'purple', linewidth=2, label='c̄p(t) (decay)')
            plt.plot(section_time, net_rate, 'black', linewidth=2, label="p'(t) (net rate)")
            plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
            plt.ylabel('Rate Terms')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4,1,3)
            plt.plot(section_time, section_pain, 'b-', alpha=0.7, linewidth=2, label='Observed')
            plt.plot(section_time, model_pain, 'r-', linewidth=2, label='Model')
            plt.ylabel('Pain (VAS)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4,1,4)
            residuals = section_pain - model_pain
            plt.plot(section_time, residuals, 'purple', alpha=0.6, linewidth=1)
            plt.axhline(0, color='black', linestyle='--', alpha=0.5)
            plt.fill_between(section_time, residuals, 0, alpha=0.3, color='purple')
            plt.ylabel('Residuals')
            plt.xlabel('Time (s)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Calculate final statistics
            mse = np.mean((section_pain - model_pain) ** 2)
            rmse = np.sqrt(mse)
            correlation = np.corrcoef(section_pain, model_pain)[0,1]
            
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   MSE: {mse:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   Correlation: {correlation:.3f}")
            print(f"   Model range: {model_pain.min():.1f} - {model_pain.max():.1f}")
            print(f"   Observed range: {section_pain.min():.1f} - {section_pain.max():.1f}")
    
    return best_params, results

# Also test with even higher force scaling
def test_high_force_scaling(subject_data):
    """Test with much higher alpha_bar values"""
    
    print(f"\n🚀 TESTING HIGH FORCE SCALING")
    print(f"{'='*40}")
    
    time_data, temp_data, pain_data, _, _ = prepare_data_for_optimization(subject_data)
    temp_func = interp1d(time_data, temp_data, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    
    # Test very high alpha values
    high_alpha_params = [
        {'alpha_bar': 5.0, 'gamma_bar': 0.05, 'name': 'High Force, Fast Decay'},
        {'alpha_bar': 10.0, 'gamma_bar': 0.1, 'name': 'Very High Force, Very Fast Decay'},
        {'alpha_bar': 3.0, 'gamma_bar': 0.03, 'name': 'Moderate High Force'},
    ]
    
    for params_set in high_alpha_params:
        params = {
            'alpha_bar': params_set['alpha_bar'],
            'gamma_bar': params_set['gamma_bar'],
            'theta': 39.31
        }
        
        print(f"\n🧪 {params_set['name']}:")
        print(f"   ᾱ={params['alpha_bar']:.1f}, c̄={params['gamma_bar']:.3f}")
        print(f"   Decay time: {1/params['gamma_bar']:.1f}s")
        
        t_span = (time_data[0], time_data[-1])
        y0 = [0.0]
        
        try:
            sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                           args=(params, temp_func),
                           t_eval=time_data[:300], method='RK45',
                           rtol=1e-2, atol=1e-4)
            
            if sol.success:
                model_pain = np.maximum(sol.y[0], 0.0)
                section_pain = pain_data[:300]
                
                # Check for reasonable behavior
                max_model = model_pain.max()
                max_observed = section_pain.max()
                
                print(f"   Model range: {model_pain.min():.1f} - {max_model:.1f}")
                print(f"   Observed range: {section_pain.min():.1f} - {max_observed:.1f}")
                
                if max_model > 100:
                    print(f"   ⚠️  Model pain too high!")
                elif max_model < 15:
                    print(f"   ⚠️  Model pain too low")
                else:
                    print(f"   ✅ Reasonable pain levels!")
                    
                    # Check decay
                    above_threshold = temp_data[:300] > params['theta']
                    transitions = np.diff(above_threshold.astype(int))
                    fall_transitions = np.where(transitions == -1)[0]
                    
                    if len(fall_transitions) > 0:
                        fall_idx = fall_transitions[0]
                        if fall_idx + 15 < len(model_pain):
                            pain_before = model_pain[fall_idx]
                            pain_after = model_pain[fall_idx + 15]
                            decay_ratio = pain_after / pain_before if pain_before > 0 else 1.0
                            print(f"   Decay ratio (15 steps): {decay_ratio:.3f}")
                            
                            if 0.4 < decay_ratio < 0.8:
                                print(f"   ✅ Good decay behavior!")
            else:
                print(f"   ❌ Integration failed")
                
        except Exception as e:
            print(f"   💥 Error: {e}")

# Run the optimization
best_params, all_results = optimize_simplified_parameters(subject_data)
test_high_force_scaling(subject_data)

#%%
def test_gradient_sensitivity(subject_data):
    """Test if small parameter changes affect the cost"""
    
    time_data, temp_data, pain_data, _, _ = prepare_data_for_optimization(subject_data)
    temp_func = interp1d(time_data, temp_data, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Test parameters around the "optimal" values
    base_params = [1.0, 0.08]  # alpha_bar, gamma_bar
    
    print("Testing parameter sensitivity:")
    print("alpha_bar  gamma_bar    MSE")
    print("-" * 30)
    
    for alpha_mult in [0.8, 1.0, 1.2]:
        for gamma_mult in [0.8, 1.0, 1.2]:
            test_alpha = base_params[0] * alpha_mult
            test_gamma = base_params[1] * gamma_mult
            
            # Test this parameter combination
            model_params = {
                'alpha_bar': test_alpha,
                'gamma_bar': test_gamma,
                'theta': 42.0  # Use average threshold
            }
            
            t_span = (time_data[0], time_data[-1])
            y0 = [0.0]
            
            try:
                sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                               args=(model_params, temp_func),
                               t_eval=time_data, method='RK45',
                               rtol=1e-3, atol=1e-6)
                
                if sol.success:
                    model_pain = np.maximum(sol.y[0], 0.0)
                    mse = np.mean((pain_data - model_pain) ** 2)
                    print(f"{test_alpha:8.3f}  {test_gamma:8.3f}  {mse:8.1f}")
                else:
                    print(f"{test_alpha:8.3f}  {test_gamma:8.3f}     FAIL")
            except:
                print(f"{test_alpha:8.3f}  {test_gamma:8.3f}     ERROR")

# Test on subject 33
subject_33_data = data_df[data_df['subject'] == 33]
test_gradient_sensitivity(subject_33_data)



def manual_parameter_search(subject_data, alpha_range, gamma_range):
    """Manually search parameter space"""
    
    time_data, temp_data, pain_data, _, _ = prepare_data_for_optimization(subject_data)
    temp_func = interp1d(time_data, temp_data, kind='linear', bounds_error=False, fill_value='extrapolate')
    threshold = extract_threshold_from_data(subject_data, vas_threshold=5)
    
    best_mse = float('inf')
    best_params = None
    results = []
    
    for alpha_bar in alpha_range:
        for gamma_bar in gamma_range:
            model_params = {
                'alpha_bar': alpha_bar,
                'gamma_bar': gamma_bar,
                'theta': threshold
            }
            
            t_span = (time_data[0], time_data[-1])
            y0 = [0.0]
            
            try:
                sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                               args=(model_params, temp_func),
                               t_eval=time_data[:1000],  # Use subset for speed
                               method='RK45', rtol=1e-3, atol=1e-6)
                
                if sol.success:
                    model_pain = np.maximum(sol.y[0], 0.0)
                    mse = np.mean((pain_data[:1000] - model_pain) ** 2)
                    
                    results.append({
                        'alpha_bar': alpha_bar,
                        'gamma_bar': gamma_bar,
                        'mse': mse
                    })
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = model_params.copy()
                        
            except:
                pass
    
    # Sort results by MSE
    results.sort(key=lambda x: x['mse'])
    
    print("Top 10 parameter combinations:")
    print("alpha_bar  gamma_bar    MSE")
    print("-" * 30)
    for r in results[:10]:
        print(f"{r['alpha_bar']:8.2f}  {r['gamma_bar']:8.3f}  {r['mse']:8.1f}")
    
    return best_params, results

# Test with wider parameter ranges
alpha_range = [0.5, 1.0, 2.0, 5.0, 10.0]
gamma_range = [0.01, 0.05, 0.1, 0.2, 0.5]

best_manual, all_results = manual_parameter_search(subject_33_data, alpha_range, gamma_range)

