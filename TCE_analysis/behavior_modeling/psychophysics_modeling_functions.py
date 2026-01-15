"""
Author: Lucille Johnston
Date Updated: 2026-01-06

Functions to help analyze the OH/OA dataset.
Focusing on psychophysics / second-order differential / dynamical systems modeling.
Cecchi 2012, Petre 2017, etc. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import signal

def plot_autocorrelations(df, lags=50):
    """
    Plot autocorrelation of a time series.
    
    Parameters:
    df : DataFrame
        DataFrame containing the time series data.
    lags : int
        Number of lags to include in the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sm.graphics.tsa.plot_acf(df, lags=lags, ax=ax)
    sm.graphics.tsa.plot_pacf(df, lags=lags, ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Plot')
    plt.show()

def extract_threshold_from_data(pain_data, vas_threshold=5): 
    """
    Extract temperature pain threshold (theta) from data by finding where VAS first exceeds threshold (set to 5 for now).
    Extracts that temperature value from each trial, averages across trials per subject to get subject-specific theta.

    Parameters:
    pain_data : DataFrame
        Data for one subject with columns: 'trial_num', 'aligned_time', 'temperature', 'pain'
    vas_threshold : float
        VAS threshold to define pain threshold (default=5) 
        When VAS crosses that threshold, you can safely say that there is pain

    Returns:
    subject_theta : float
        value of the subject's pain threshold (theta)
    """
    trial_thetas = {}
    for trial in pain_data['trial_num'].unique():
        trial_data = pain_data[pain_data['trial_num'] == trial].copy()
        trial_data = trial_data.sort_values('aligned_time').reset_index(drop=True)
        # Find first temperature where pain exceeds threshold
        above_threshold = trial_data[trial_data['pain'] >= vas_threshold]
        if not above_threshold.empty:
            threshold_temp = above_threshold.iloc[0]['temperature']
            trial_thetas[trial] = threshold_temp
        else:
            trial_thetas[trial] = np.nan
    # Average across trials, ignoring NaNs
    subject_theta = np.nanmean(list(trial_thetas.values()))
    return subject_theta



################# Cecchi 2012 Model Functions #################
# def cecchi2012_full(t, y, params, T_func, temp_derivative_func=None):
#     """
#     Cecchi 2012 model: p''(t) = αF(T(t),θ) - βp'(t) + γ(T'(t) - λ)p(t)
    
#     Parameters:
#     t : float
#         Time variable.
#     y : list
#         List containing [pain, pain_rate].
#     params : dict
#         Dictionary containing model parameters:
#         - alpha: force scaling coefficient (subject-specific)
#         - beta: damping coefficient (subject-specific)
#         - gamma: temperature rate coefficient (subject-specific)
#         - lambda_param: temperature rate threshold (subject-specific)
#         - theta: temperature threshold for F(T,θ)
#     T_func : callable
#         Function that returns the temperature T(t) at time t

#     temp_derivative_func : callable
#         Function that returns the dT/dt at time t

#     Returns:
#     dydt : list
#         List containing [pain_rate, pain_acceleration].
#     """
#     pain, pain_rate = y # Model's current predictions
#     alpha = params['alpha']
#     beta = params['beta'] 
#     gamma = params['gamma']
#     lambda_param = params['lambda_param']  # Using lambda_param since lambda is a Python keyword
#     theta = params['theta']
    
#     # Current temperature
#     T = T_func(t)
    
#     # Temperature derivative 
#     if temp_derivative_func is not None:
#         T_dt = temp_derivative_func(t)
#     else:
#         # Fallback to finite difference
#         dt = 1e-6
#         T_dt = (T_func(t + dt) - T_func(t)) / dt

#     # Linear function F(T,θ) - pain level based on temperature above threshold theta
#     if T > theta:
#         F_T = T - theta
#     else:
#         F_T = 0.0 # no pain below threshold

#     # Pain acceleration: p''(t) = αF(T(t),θ) - βp'(t) + γ(T'(t) - λ)p(t)
#     pain_acceleration = ((alpha * F_T) - 
#                          (beta * pain_rate) + 
#                          (gamma * (T_dt - lambda_param) * pain))

#     return [pain_rate, pain_acceleration]

def cecchi2012_simplified(t, y, params, T_func):
    """
    Simplified Cecchi 2012: p'(t) = ᾱF(T(t),T₀) - γ̄p(t)
    """
    pain = y[0]
    alpha_bar = params['alpha_bar']  # α/β 
    gamma_bar = params['gamma_bar']  # γλ/β
    theta = params['theta']  # temperature threshold
    
    T = T_func(t)
    
    # Step function F(T,T₀)
    F_T = max(0, T - theta)
    
    # First-order equation: p'(t) = ᾱF(T,T₀) - γ̄p(t)
    pain_rate = alpha_bar * F_T - gamma_bar * pain

    if pain >= 100.0 and pain_rate > 0:
        pain_rate = 0.0  # Stop increasing when at max
    elif pain < 0.0 and pain_rate < 0:
        pain_rate = 0.0  # Stop decreasing when at min
    
    return [pain_rate]

def prepare_data_for_optimization(subject_data):
    # Prepare concatenated trial data
    trials = sorted(subject_data['trial_num'].unique())
    concatenated_data = []
    time_offset = 0.0
    for trial_num in trials:
        trial_data = subject_data[subject_data['trial_num'] == trial_num].copy()
        clean_data = trial_data.dropna(subset=['aligned_time', 'temperature', 'pain']).copy()
        clean_data = clean_data.sort_values('aligned_time').reset_index(drop=True)

        trial_start = clean_data['aligned_time'].min()
        clean_data['continuous_time'] = clean_data['aligned_time'] - trial_start + time_offset
        concatenated_data.append(clean_data)
        trial_duration = clean_data['aligned_time'].max() - clean_data['aligned_time'].min()
        time_offset += trial_duration + 1.0  # 1s gap between trials
    if not concatenated_data:
        print("No valid trials found for optimization.")
        return None, None
    combined_data = pd.concat(concatenated_data, ignore_index=True)
    time_data = combined_data['continuous_time'].values
    temp_data = combined_data['temperature'].values
    pain_data = combined_data['pain'].values

    # Remove duplicate time points
    unique_mask = np.concatenate(([True], np.diff(time_data) > 1e-10))
    time_data = time_data[unique_mask]
    temp_data = temp_data[unique_mask]
    pain_data = pain_data[unique_mask]

    # Pre-compute temperature derivatives
    temp_derivatives = np.gradient(temp_data, time_data)
    temp_deriv_func = interp1d(time_data, temp_derivatives, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
    return time_data, temp_data, pain_data, concatenated_data, temp_deriv_func

################# Parameter Optimization Functions #################
# def optimize_cecchi_full(subject_data, threshold=None, initial_params=None,
#                          use_multiple_starts=False, n_starts=5, verbose=True):
#     """
#     Parameter optimization for the Full Cecchi 2012 model.
#     Full model: p''(t) = αF(T(t),θ) - βp'(t) + γ(T'(t) - λ)p(t)
#     Parameters to optimize: [alpha, beta, gamma, lambda_param] (and optionally theta)

#     Parameters:
#     subject_data : DataFrame
#         Subject's task data with columns: 'aligned_time', 'temperature', 'pain', 'trial_num'
#         Note: I'm going to have cut each trial off at 60s and downsampled to 5Hz before passing in here
#     threshold : float, optional
#         Subject's pain threshold (theta). If None, it will be optimized as well
#     initial_params : dict, optional
#         Starting parameter values. If None, uses Petre 2017 values.
#     use_multiple_starts : bool
#         If True, tries multiple random starting points
#     n_starts : int
#         Number of random starting points to try
#     verbose : bool
#         Print optimization progress
    
#     Returns:
#     best_params : dict
#         Optimized parameters with keys: alpha, beta, gamma, lambda_param, theta, mse, success
#     best_result : OptimizeResult
#         Full scipy optimization result object
#     """
#     from scipy.optimize import minimize
#     # Default Petre 2017 parameters
#     if initial_params is None:
#         initial_params = {
#             'alpha': 2.4932,
#             'beta': 36.7552,
#             'gamma': 0.0204,
#             'lambda_param': 0.0169,
#             'theta': 37.1913 if threshold is None else threshold
#         }

#     # Determine if we're optimizing theta
#     optimize_theta = (threshold is None)
#     if not optimize_theta and threshold is not None:
#         initial_params['theta'] = threshold

#     if verbose:
#         print(f"\n FULL MODEL OPTIMIZAION")
#         print(f"   Optimize theta: {optimize_theta}")
#         print(f"   Multiple starts: {use_multiple_starts} (n={n_starts if use_multiple_starts else 1})")

#     # Prepare concatenated trial data
#     time_data, temp_data, pain_data, concatenated_data, temp_deriv_func = prepare_data_for_optimization(subject_data)

#     if verbose:
#         print(f"    Data: {len(time_data)} time points across {len(concatenated_data)} trials")
#         print(f"    Pain range: {pain_data.min():.1f} to {pain_data.max():.1f}")

#     # Create interpolation function for ODE solver
#     temp_func = interp1d(time_data, temp_data, kind='linear',
#                          bounds_error=False, fill_value='extrapolate')

#     # Define objective function
#     def objective(params_array):
#         """ MSE between observed and predicted pain"""
#         if optimize_theta:
#             alpha, beta, gamma, lambda_param, theta = params_array
#         else:
#             alpha, beta, gamma, lambda_param = params_array
#             theta = initial_params['theta']
#         model_params = {
#             'alpha': alpha,
#             'beta': beta,
#             'gamma': gamma,
#             'lambda_param': lambda_param,
#             'theta': theta
#         }
        
#         t_span = (time_data[0], time_data[-1])
#         y0 = [0.0, 0.0] # Initial [pain, pain rate]
#         # Quick sanity check before expensive ODE solve
#         if alpha <= 0 or beta <= 0 or gamma <= 0:
#             return 1e6

#         try:
#             sol = solve_ivp(cecchi2012_full, t_span, y0,
#                             args=(model_params, temp_func, temp_deriv_func),
#                             t_eval=time_data, method='RK45', # Trying RK45 to see if it is faster than LSODA
#                             rtol=1e-2, atol=1e-4) # Play with these tolerances to balance speed and accuracy
#             if sol.success:
#                 model_pain = np.maximum(sol.y[0], 0.0) # No negative pain
#                 mse = np.mean((pain_data - model_pain) ** 2)
#                 return mse
#             else:
#                 return 1e6
#         except Exception as e:
#             return 1e6
    
#     # Parameter bounds
#     if optimize_theta:
#         bounds = [
#             (0.5, 10.0), # alpha
#             (10.0, 80.0), # beta
#             (0.0005, 0.1), # gamma
#             (-0.05, 0.05), # lambda_param
#             (35.0, 50.0) # theta
#         ]
#         x0_default = [initial_params['alpha'], initial_params['beta'],
#                       initial_params['gamma'], initial_params['lambda_param'],
#                       initial_params['theta']]
#     else:
#         bounds = [
#             (0.5, 10.0), # alpha
#             (10.0, 80.0), # beta
#             (0.0005, 0.1), # gamma
#             (-0.05, 0.05) # lambda_param
#         ]
#         x0_default = [initial_params['alpha'], initial_params['beta'],
#                       initial_params['gamma'], initial_params['lambda_param']]
    
#     # Generate starting points
#     np.random.seed()
#     if use_multiple_starts:
#         starting_points = [x0_default] # Always include default
#         for _ in range(n_starts - 1):
#             random_start = [np.random.uniform(b[0],b[1]) for b in bounds]
#             starting_points.append(random_start)
#     else:
#         starting_points = [x0_default]
    
#     # Try each starting point
#     best_result = None
#     best_cost = np.inf
#     best_start_idx = -1
#     for idx, x0 in enumerate(starting_points):
#         if verbose:
#             print(f"    Starting point {idx+1}/{len(starting_points)}...", end='', flush=True)
#         # Test initial cost
#         try:
#             initial_cost = objective(x0)
#             result = minimize(objective, x0, method='L-BFGS-B',
#                             bounds=bounds, options={'maxiter':1000, 
#                                                     'maxfun': 15000, 
#                                                     'ftol':1e-12,
#                                                     'gtol': 1e-10,
#                                                     'eps': 1e-8,
#                                                     'disp': True})
            
#             if verbose:
#                 status = "✓" if result.success else "✗"
#                 print(f"{status} cost: {initial_cost:.1f} → {result.fun:.1f} "
#                     f"({result.nfev} evals, {result.nit} iters)")
            
#             if result.fun < best_cost:
#                 best_cost = result.fun
#                 best_result = result
#                 best_start_idx = idx
#         except Exception as e:
#             if verbose:
#                 print(f"❌ Error during optimization: {e}", end='', flush=True)
#             continue
    
#     # Package results
#     if best_result is not None and best_result.fun < 1e5:
#         if optimize_theta:
#             alpha, beta, gamma, lambda_param, theta = best_result.x
#         else:
#             alpha, beta, gamma, lambda_param = best_result.x
#             theta = initial_params['theta']
        
#         best_params = {
#             'alpha': alpha,
#             'beta': beta,
#             'gamma': gamma,
#             'lambda_param': lambda_param,
#             'theta': theta,
#             'mse': best_result.fun,
#             'success': True,
#             'n_trials': len(concatenated_data),
#             'n_points': len(time_data),
#             'n_evals': best_result.nfev,
#             'n_iters': best_result.nit,
#             'best_start_index': best_start_idx
#         }

#         # Re-run model once to get predictions for storage
#         model_params = {
#             'alpha': alpha,
#             'beta': beta,
#             'gamma': gamma, 
#             'lambda_param': lambda_param,
#             'theta': theta
#         }
#         t_span = (time_data[0], time_data[-1])
#         y0 = [0.0, 0.0] # Initial [pain, pain rate]

#         try:
#             sol = solve_ivp(cecchi2012_full, t_span, y0,
#                             args=(model_params, temp_func, temp_deriv_func),
#                             t_eval=time_data, method='LSODA',
#                             rtol=1e-6, atol=1e-9)
#             if sol.success:
#                 model_pain = np.maximum(sol.y[0], 0.0) # No negative pain
#                 best_params['model_data'] = {
#                     'time': time_data,
#                     'predicted_pain': model_pain,
#                     'observed_pain': pain_data,
#                     'temperature': temp_data
#                 }
#             if verbose:
#                 print(f"\n   ✅ Optimization successful (start #{best_start_idx+1}):")
#                 print(f"      α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, λ={lambda_param:.4f}, θ={theta:.2f}")
#                 print(f"      MSE={best_result.fun:.2f} ({best_result.nfev} evals, {best_result.nit} iters)")

#         except Exception as e:
#             if verbose:
#                 print(f"❌ Error during final solve: {e}", end='', flush=True)
#     return best_params, best_result


def optimize_cecchi_simplified(subject_data, threshold=None, initial_params=None,
                               use_multiple_starts=False, n_starts=5, verbose=True):
    """
    Parameter optimization for the Simplified Cecchi 2012 model.
    Simplified model: p'(t) = ᾱF(T,θ) - γ̄p(t)
    Parameters to optimize: [alpha_bar, gamma_bar] (and optionally theta)
    Where: ᾱ = α/β and γ̄ = γλ/β (reduced parameters)
    
    Parameters:
    -----------
    subject_data : DataFrame
        Subject's pain data with columns: 'aligned_time', 'temperature', 'pain', 'trial_num'
    threshold : float, optional
        Subject's pain threshold (θ). If None, will be optimized as well.
    initial_params : dict, optional
        Starting parameter values. If None, derives from Petre 2017 full model values.
    use_multiple_starts : bool
        If True, tries multiple random starting points
    n_starts : int
        Number of random starting points to try
    verbose : bool
        Print optimization progress
        
    Returns:
    --------
    best_params : dict
        Optimized parameters with keys: alpha_bar, gamma_bar, theta, mse, success
    best_result : OptimizeResult
        Full scipy optimization result object
    """
    from scipy.optimize import minimize
    # Default parameters derived from searching parameter space
    if initial_params is None:
        initial_params = {
            'alpha_bar': 1.0,  # α/β
            'gamma_bar': 0.1,  # γλ/β
            'theta': 39.31 if threshold is None else threshold
        }
    
    # Determine if we're optimizing theta
    optimize_theta = (threshold is None)
    if not optimize_theta and threshold is not None:
        initial_params['theta'] = threshold

    if verbose:
        print(f"\n🔧 SIMPLIFIED MODEL OPTIMIZATION")
        print(f"   Optimize theta: {optimize_theta}")
        print(f"   Multiple starts: {use_multiple_starts} (n={n_starts if use_multiple_starts else 1})")
    
    # Prepare concatenated trial data
    time_data, temp_data, pain_data, concatenated_data, temp_deriv_func = prepare_data_for_optimization(subject_data)

    if verbose:
        print(f"    Data: {len(time_data)} time points across {len(concatenated_data)} trials")
        print(f"    Pain range: {pain_data.min():.1f} to {pain_data.max():.1f}")

    # Create interpolation function for ODE solver
    temp_func = interp1d(time_data, temp_data, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    
    # Define objective function
    def objective(params_array):
        """ MSE between observed and predicted pain"""
        if optimize_theta:
            alpha_bar, gamma_bar, theta = params_array
        else:
            alpha_bar, gamma_bar = params_array
            theta = initial_params['theta']
        model_params = {
            'alpha_bar': alpha_bar,
            'gamma_bar': gamma_bar,
            'theta': theta
        }
        
        t_span = (time_data[0], time_data[-1])
        y0 = [0.0] # Initial pain

        try:
            sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                            args=(model_params, temp_func),
                            t_eval=time_data, method='LSODA',
                            rtol=1e-6, atol=1e-9)
            if sol.success:
                model_pain = np.maximum(sol.y[0], 0.0) # No negative pain
                mse = np.mean((pain_data - model_pain) ** 2)
                return mse
            else:
                return 1e6
        except:
            return 1e6
        
    # Parameter bounds
    if optimize_theta:
        bounds = [
            (0.5, 8.0), # alpha_bar: based on parameter search
            (0.01, 0.6), # gamma_bar: based on parameter search
            (37.0, 50.0) # theta 
        ]
        x0_default = [initial_params['alpha_bar'],
                      initial_params['gamma_bar'],
                      initial_params['theta']]
    else:
        bounds = [
            (0.5, 10.0), # alpha_bar: based on parameter search
            (0.01, 0.6) # gamma_bar: based on parameter search
        ]
        x0_default = [initial_params['alpha_bar'],
                      initial_params['gamma_bar']]
    
    # Generate starting points
    if use_multiple_starts:
        starting_points = []
        for _ in range(n_starts):
            random_start = [np.random.uniform(b[0],b[1]) for b in bounds]
            starting_points.append(random_start)
    else:
        starting_points = [[np.random.uniform(0.8, 1.2), np.random.uniform(0.08, 0.12)]]

    # Try each starting point
    best_result = None
    best_cost = np.inf
    best_start_idx = -1
    for idx, x0 in enumerate(starting_points):
        if verbose:
            print(f"    Starting point {idx+1}/{len(starting_points)}...", end='')
        
        # Test initial cost
        initial_cost = objective(x0)
        result = minimize(objective, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter':1000,
                                                  'maxfun': 5000, 
                                                  'ftol':1e-12,      # Tighter tolerance
                                                  'gtol': 1e-10,     # Tighter gradient tolerance
                                                  'eps': 1e-6,       # Larger step size for gradient estimation
                                                  'finite_diff_rel_step': 1e-4})     # Larger relative step
        
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} cost: {initial_cost:.1f} → {result.fun:.1f} "
                  f"({result.nfev} evals, {result.nit} iters)")
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            best_start_idx = idx
    
    # Package results
    if best_result is not None and best_result.success:
        if optimize_theta:
            alpha_bar, gamma_bar, theta = best_result.x
        else:
            alpha_bar, gamma_bar = best_result.x
            theta = initial_params['theta']
        
        best_params = {
            'alpha_bar': alpha_bar,
            'gamma_bar': gamma_bar,
            'theta': theta,
            'mse': best_result.fun,
            'success': True,
            'n_trials': len(concatenated_data),
            'n_points': len(time_data),
            'n_evals': best_result.nfev,
            'n_iters': best_result.nit,
            'best_start_index': best_start_idx
        }

        # Re-run model once to get predictions for storage
        model_params = {
        'alpha_bar': alpha_bar,
        'gamma_bar': gamma_bar,
        'theta': theta
        }
        t_span = (time_data[0], time_data[-1])
        y0 = [0.0]

        try:
            print(f"    Re-running model to save predictions...")
            sol = solve_ivp(cecchi2012_simplified, t_span, y0,
                            args=(model_params, temp_func),
                            t_eval=time_data, method='RK45',
                            rtol=1e-4, atol=1e-6)
            if sol.success:
                model_pain = np.maximum(sol.y[0], 0.0)
                model_pain[model_pain > 100.0] = 100.0 # Cap at 100
                best_params['model_data'] = {
                    'time': time_data,
                    'predicted_pain': model_pain,
                    'observed_pain': pain_data,
                    'temperature': temp_data
                }
                print(f"    ✅ Model data saved successfully!")
            else:
                print(f"    ❌ Final model solve failed: {sol.message}")
        except Exception as e:
            print(f"    ❌ Error during final solve: {e}")
            import traceback
            traceback.print_exc()

        if verbose:
            print(f"\n   ✅ Optimization successful (start #{best_start_idx+1}):")
            print(f"      ᾱ={alpha_bar:.4f}, γ̄={gamma_bar:.4f}, θ={theta:.2f}")
            print(f"      MSE={best_result.fun:.2f} ({best_result.nfev} evals, {best_result.nit} iters)")
    else:
        if verbose:
            print(f"\n   ❌ Optimization failed.")
        best_params = None
    return best_params, best_result








################## Plotting Functions ##################
def plot_optimization_fit(subject, optimization_results, save_path=None, figsize=(14,10)):
    """
    Plot observed vs. predicted pain using saved optimization results.
    Updated for simplified model parameters.
    
    Parameters:
    subject : int
        Subject ID
    optimization_results : dict
        Dictionary of optimization results with saved model_data
    save_path : str, optional
        Path to save the figure. If None, it just displays the figure
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    fig : matplotlib.figure.Figure
        The figure object
    """
    from sklearn.metrics import r2_score
    if subject not in optimization_results:
        print(f"No optimization results found for subject {subject}.")
        return None
    
    params = optimization_results[subject]['params']

    # Check for saved model data
    if 'model_data' not in params:
        print(f"No model data found in optimization results for subject {subject}.")
        return None
    
    # Extract saved data
    data = params['model_data']
    time = data['time']
    temp = data['temperature']
    observed_pain = data['observed_pain']
    model_pain = data['predicted_pain']

    # Calculate fit statistics
    residuals = observed_pain - model_pain
    mse = params['mse']
    rmse = np.sqrt(mse)
    r2 = r2_score(observed_pain, model_pain)
    mae = np.mean(np.abs(residuals))
    correlation = np.corrcoef(observed_pain, model_pain)[0,1]

    # Create figure with shared x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Top plot: Temperature with threshold
    ax1.plot(time, temp, 'o-', label='Temperature (°C)', 
             color='orange', linewidth=2, markersize=3, alpha=0.7)
    ax1.axhline(params['theta'], color='red', linestyle='--', label='Threshold (θ)')
    ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Subject {subject} - Simplified Model Fit ({params["n_trials"]} trials, {params["n_points"]} points)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_ylim(30, 50)

    # Middle plot: Observed vs. Predicted Pain
    ax2.plot(time, observed_pain, label='Observed Pain',
             color='red', linewidth=2, alpha=0.7, linestyle='-')
    ax2.plot(time, model_pain, label='Predicted Pain',
             color='blue', linewidth=2, alpha=0.7, linestyle='-')
    ax2.set_ylabel('Pain (VAS)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_ylim(0, 105)

    # Bottom plot: Residuals over time
    ax3.plot(time, residuals, color='purple', alpha=0.6, linewidth=1)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(time, residuals, 0, alpha=0.3, color='purple')
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residuals (Observed - Predicted)', fontsize=12, fontweight='bold')
    ax3.set_ylim(-100, 100)
    ax3.set_title('Model Residuals', fontsize=11)

    # Add text box with SIMPLIFIED MODEL parameters and fit statistics
    param_text = (f'Simplified Model Parameters:\n'
                  f'ᾱ = {params["alpha_bar"]:.4f}\n'
                  f'γ̄ = {params["gamma_bar"]:.4f}\n'
                  f'θ = {params["theta"]:.2f}°C\n'
                  f'\nFit Statistics:\n'
                  f'R² = {r2:.3f}\n'
                  f'RMSE = {rmse:.2f}\n'
                  f'MAE = {mae:.2f}\n'
                  f'Corr = {correlation:.3f}')
    ax2.text(0.98, 0.97, param_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    return fig
    

def print_optimization_summary(optimization_results):
    """
    Print a summary table of all optimization results.
    Updated for simplified model parameters.
    
    Parameters:
    -----------
    optimization_results : dict
        Dictionary of optimization results
    """
    from sklearn.metrics import r2_score
    
    print(f"\n{'='*80}")
    print(f"SIMPLIFIED MODEL OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # Collect statistics
    summary_data = []
    
    for subject in sorted(optimization_results.keys()):
        params = optimization_results[subject]['params']
        
        if 'model_data' in params:
            data = params['model_data']
            r2 = r2_score(data['observed_pain'], data['predicted_pain'])
        else:
            r2 = np.nan
        
        summary_data.append({
            'subject': subject,
            'theta': params['theta'],
            'alpha_bar': params['alpha_bar'],  # Changed from 'alpha'
            'gamma_bar': params['gamma_bar'],  # Changed from 'gamma'
            'mse': params['mse'],
            'r2': r2,
            'n_trials': params['n_trials'],
            'n_points': params['n_points'],
            'n_evals': params['n_evals'],
            'n_iters': params['n_iters'],
            'best_start': params['best_start_index'] + 1
        })
    
    # Print table header (updated for simplified model)
    print(f"{'Subj':<6} {'θ':>7} {'ᾱ':>8} {'γ̄':>8} {'MSE':>8} {'R²':>6} {'Trials':>7} {'Pts':>6} {'Evals':>6} {'Iters':>6} {'Start':>6}")
    print(f"{'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    
    for s in summary_data:
        print(f"{s['subject']:<6} "
              f"{s['theta']:>7.2f} "
              f"{s['alpha_bar']:>8.4f} "
              f"{s['gamma_bar']:>8.4f} "
              f"{s['mse']:>8.1f} "
              f"{s['r2']:>6.3f} "
              f"{s['n_trials']:>7} "
              f"{s['n_points']:>6} "
              f"{s['n_evals']:>6} "
              f"{s['n_iters']:>6} "
              f"{s['best_start']:>6}")
    
    # Print summary statistics (updated for simplified model)
    print(f"\n{'-'*80}")
    print(f"SUMMARY STATISTICS (n={len(summary_data)} subjects)")
    print(f"{'-'*80}")
    
    for param in ['theta', 'alpha_bar', 'gamma_bar', 'mse', 'r2']:
        values = [s[param] for s in summary_data if not np.isnan(s[param])]
        if values:
            print(f"{param:>9}: mean={np.mean(values):>8.4f}, "
                  f"std={np.std(values):>8.4f}, "
                  f"min={np.min(values):>8.4f}, "
                  f"max={np.max(values):>8.4f}")
    
    print(f"{'='*80}\n")






################# Side functions to check data quality #################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_subject_data(subject_id, data_df, detailed=True):
    """
    Comprehensive analysis of a subject's data to identify potential issues
    """
    subject_data = data_df[data_df['subject'] == subject_id].copy()
    
    if subject_data.empty:
        print(f"No data found for subject {subject_id}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 DEEP DIVE ANALYSIS: SUBJECT {subject_id}")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total rows: {len(subject_data)}")
    print(f"   Unique trials: {subject_data['trial_num'].nunique()}")
    print(f"   Trial numbers: {sorted(subject_data['trial_num'].unique())}")
    print(f"   Time range: {subject_data['aligned_time'].min():.2f} to {subject_data['aligned_time'].max():.2f}s")
    print(f"   Temperature range: {subject_data['temperature'].min():.2f} to {subject_data['temperature'].max():.2f}°C")
    print(f"   Pain range: {subject_data['pain'].min():.2f} to {subject_data['pain'].max():.2f}")
    
    # Check for missing data
    print(f"\n🚨 MISSING DATA CHECK:")
    missing_temp = subject_data['temperature'].isna().sum()
    missing_pain = subject_data['pain'].isna().sum()
    missing_time = subject_data['aligned_time'].isna().sum()
    print(f"   Missing temperature: {missing_temp} ({missing_temp/len(subject_data)*100:.1f}%)")
    print(f"   Missing pain: {missing_pain} ({missing_pain/len(subject_data)*100:.1f}%)")
    print(f"   Missing time: {missing_time} ({missing_time/len(subject_data)*100:.1f}%)")
    
    # Check for extreme values
    print(f"\n⚠️  EXTREME VALUES:")
    temp_q99 = subject_data['temperature'].quantile(0.99)
    temp_q01 = subject_data['temperature'].quantile(0.01)
    pain_q99 = subject_data['pain'].quantile(0.99)
    pain_q01 = subject_data['pain'].quantile(0.01)
    
    extreme_temp = subject_data[(subject_data['temperature'] > temp_q99) | 
                               (subject_data['temperature'] < temp_q01)]
    extreme_pain = subject_data[(subject_data['pain'] > pain_q99) | 
                               (subject_data['pain'] < pain_q01)]
    
    print(f"   Extreme temperatures (>99th or <1st percentile): {len(extreme_temp)} points")
    print(f"   Extreme pain (>99th or <1st percentile): {len(extreme_pain)} points")
    
    if len(extreme_temp) > 0:
        print(f"   Extreme temp values: {extreme_temp['temperature'].values[:10]}")
    if len(extreme_pain) > 0:
        print(f"   Extreme pain values: {extreme_pain['pain'].values[:10]}")
    
    # Check for duplicates and time issues
    print(f"\n🕐 TIME SERIES ISSUES:")
    time_diffs = subject_data.groupby('trial_num')['aligned_time'].apply(lambda x: np.diff(x.sort_values()))
    
    all_diffs = []
    for trial, diffs in time_diffs.items():
        all_diffs.extend(diffs)
    
    all_diffs = np.array(all_diffs)
    print(f"   Time step stats: min={all_diffs.min():.4f}s, max={all_diffs.max():.4f}s, mean={all_diffs.mean():.4f}s")
    print(f"   Zero time steps: {np.sum(all_diffs == 0)}")
    print(f"   Negative time steps: {np.sum(all_diffs < 0)}")
    print(f"   Very small time steps (<0.01s): {np.sum(all_diffs < 0.01)}")
    print(f"   Very large time steps (>1s): {np.sum(all_diffs > 1.0)}")
    
    # Analyze temperature derivatives
    print(f"\n🌡️  TEMPERATURE DERIVATIVE ANALYSIS:")
    temp_derivatives = []
    for trial in sorted(subject_data['trial_num'].unique()):
        trial_data = subject_data[subject_data['trial_num'] == trial].sort_values('aligned_time')
        if len(trial_data) > 1:
            temp_deriv = np.gradient(trial_data['temperature'].values, 
                                   trial_data['aligned_time'].values)
            temp_derivatives.extend(temp_deriv)
    
    temp_derivatives = np.array(temp_derivatives)
    temp_derivatives = temp_derivatives[np.isfinite(temp_derivatives)]  # Remove inf/nan
    
    if len(temp_derivatives) > 0:
        print(f"   Temp derivative stats: min={temp_derivatives.min():.2f}, max={temp_derivatives.max():.2f}")
        print(f"   Temp derivative mean={temp_derivatives.mean():.2f}, std={temp_derivatives.std():.2f}")
        print(f"   Extreme derivatives (>10°C/s): {np.sum(np.abs(temp_derivatives) > 10)}")
        print(f"   Very extreme derivatives (>50°C/s): {np.sum(np.abs(temp_derivatives) > 50)}")
        print(f"   Insane derivatives (>100°C/s): {np.sum(np.abs(temp_derivatives) > 100)}")
    
    # Check pain-temperature relationship
    print(f"\n🔗 PAIN-TEMPERATURE RELATIONSHIP:")
    correlation = subject_data['temperature'].corr(subject_data['pain'])
    print(f"   Overall correlation: {correlation:.3f}")
    
    # Check for weird patterns by trial
    print(f"\n📋 PER-TRIAL ANALYSIS:")
    for trial in sorted(subject_data['trial_num'].unique())[:5]:  # First 5 trials
        trial_data = subject_data[subject_data['trial_num'] == trial]
        trial_temp_range = trial_data['temperature'].max() - trial_data['temperature'].min()
        trial_pain_range = trial_data['pain'].max() - trial_data['pain'].min()
        trial_duration = trial_data['aligned_time'].max() - trial_data['aligned_time'].min()
        
        print(f"   Trial {trial}: {len(trial_data)} points, {trial_duration:.1f}s, "
              f"temp Δ={trial_temp_range:.1f}°C, pain Δ={trial_pain_range:.1f}")
    
    if detailed:
        # Create diagnostic plots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Subject {subject_id} - Detailed Diagnostic Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Temperature over time (first 3 trials)
        ax1 = fig.add_subplot(gs[0, 0])
        for trial in sorted(subject_data['trial_num'].unique())[:3]:
            trial_data = subject_data[subject_data['trial_num'] == trial].sort_values('aligned_time')
            ax1.plot(trial_data['aligned_time'], trial_data['temperature'], 
                    'o-', label=f'Trial {trial}', markersize=3, alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Profiles (First 3 Trials)')
        ax1.set_xlim([subject_data['aligned_time'].min(), subject_data['aligned_time'].max()])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pain over time (first 3 trials)
        ax2 = fig.add_subplot(gs[0, 1])
        for trial in sorted(subject_data['trial_num'].unique())[:3]:
            trial_data = subject_data[subject_data['trial_num'] == trial].sort_values('aligned_time')
            ax2.plot(trial_data['aligned_time'], trial_data['pain'], 
                    'o-', label=f'Trial {trial}', markersize=3, alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pain (VAS)')
        ax2.set_title('Pain Ratings (First 3 Trials)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(subject_data['temperature'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(temp_q01, color='red', linestyle='--', label='1st percentile')
        ax3.axvline(temp_q99, color='red', linestyle='--', label='99th percentile')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Count')
        ax3.set_title('Temperature Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pain histogram
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(subject_data['pain'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax4.axvline(pain_q01, color='red', linestyle='--', label='1st percentile')
        ax4.axvline(pain_q99, color='red', linestyle='--', label='99th percentile')
        ax4.set_xlabel('Pain (VAS)')
        ax4.set_ylabel('Count')
        ax4.set_title('Pain Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Temperature derivatives
        ax5 = fig.add_subplot(gs[1, 1])
        if len(temp_derivatives) > 0:
            ax5.hist(temp_derivatives, bins=100, edgecolor='black', alpha=0.7)
            ax5.axvline(10, color='red', linestyle='--', label='±10°C/s')
            ax5.axvline(-10, color='red', linestyle='--')
            ax5.set_xlabel('Temperature Derivative (°C/s)')
            ax5.set_ylabel('Count')
            ax5.set_title('Temperature Rate of Change')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_xlim([-50, 50])  # Focus on reasonable range
        
        # Plot 6: Time step histogram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(all_diffs[all_diffs < 1.0], bins=100, edgecolor='black', alpha=0.7)
        ax6.axvline(0.01, color='red', linestyle='--', label='0.01s')
        ax6.set_xlabel('Time Step (s)')
        ax6.set_ylabel('Count')
        ax6.set_title('Time Step Distribution (<1s)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Pain vs Temperature scatter
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(subject_data['temperature'], subject_data['pain'], 
                   alpha=0.3, s=10, edgecolors='none')
        ax7.set_xlabel('Temperature (°C)')
        ax7.set_ylabel('Pain (VAS)')
        ax7.set_title(f'Pain vs Temperature (r={correlation:.3f})')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Trial durations
        ax8 = fig.add_subplot(gs[2, 1])
        trial_durations = []
        trial_nums = []
        for trial in sorted(subject_data['trial_num'].unique()):
            trial_data = subject_data[subject_data['trial_num'] == trial]
            duration = trial_data['aligned_time'].max() - trial_data['aligned_time'].min()
            trial_durations.append(duration)
            trial_nums.append(trial)
        ax8.bar(range(len(trial_nums)), trial_durations, edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Trial Index')
        ax8.set_ylabel('Duration (s)')
        ax8.set_title('Trial Durations')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Plot 9: Points per trial
        ax9 = fig.add_subplot(gs[2, 2])
        trial_counts = subject_data.groupby('trial_num').size()
        ax9.bar(range(len(trial_counts)), trial_counts.values, edgecolor='black', alpha=0.7)
        ax9.set_xlabel('Trial Index')
        ax9.set_ylabel('Number of Points')
        ax9.set_title('Data Points per Trial')
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    return subject_data


