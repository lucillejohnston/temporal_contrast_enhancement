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
from sklearn.metrics import r2_score

def plot_autocorrelations(df, lags=50):
    """
    Plot autocorrelation of a time series.
    
    Parameters:
    df : DataFrame
        DataFrame containing the time series data.
    lags : int
        Number of lags to include in the plot.
    """
    # One axis each: drawing both onto the same axes makes the second
    # overwrite the first, which is what this previously did.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    sm.graphics.tsa.plot_acf(df, lags=lags, ax=ax1)
    sm.graphics.tsa.plot_pacf(df, lags=lags, ax=ax2)
    ax1.set(title='Autocorrelation', ylabel='ACF')
    ax2.set(title='Partial autocorrelation', xlabel='Lag', ylabel='PACF')
    plt.tight_layout()
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
    """
    SUPERSEDED -- kept only because parameter_search.py still calls it.

    Concatenates all of a subject's trials into one continuous series with a
    1s gap, for a single ODE solve across the whole thing. The modelling
    pipeline uses prepare_trials_for_optimization() instead, which keeps the
    trials separate so the model does not carry one trial's pain into the
    start of the next.

    parameter_search.py is itself superseded: it runs its own LSODA solves
    with no max_step, which is the bug that made earlier fits meaningless
    (see simulate_trials). Treat anything it produces as invalid, including
    the parameter bounds it was once used to choose.
    """
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


def prepare_trials_for_optimization(subject_data):
    """
    Split one subject's data into independent per-trial blocks.

    This is the per-trial counterpart to prepare_data_for_optimization(), which
    concatenates every trial into one continuous time series with a 1s gap and
    integrates a single ODE across the whole thing. That carries the model's
    pain state from the end of one trial into the start of the next, and since
    gamma_bar is small almost nothing decays in 1s. About 18% of trials end
    above 10 VAS (mostly the hold trials, where the stimulus is still on when
    the trace is cut), so for those the model would start the next trial with
    pain the subject does not have, and the error compounds down the session.

    Integrating each trial from p=0 removes that. Each trial keeps its own
    aligned_time, so integration starts in the pre-stimulus baseline where
    T < theta and pain is genuinely 0.

    Parameters:
    -----------
    subject_data : DataFrame
        One subject's data with columns 'trial_num', 'aligned_time',
        'temperature', 'pain'.

    Returns:
    --------
    trials : list of dict
        One entry per usable trial, each with 'trial_num', 'time', 'temperature',
        'pain', 'temp_func' and 'continuous_time' (for plotting only).
    """
    trials = []
    time_offset = 0.0
    for trial_num in sorted(subject_data['trial_num'].unique()):
        trial_data = subject_data[subject_data['trial_num'] == trial_num]
        clean = trial_data.dropna(subset=['aligned_time', 'temperature', 'pain'])
        clean = clean.sort_values('aligned_time')
        # Strictly increasing time is required by interp1d and solve_ivp
        clean = clean[~clean['aligned_time'].duplicated(keep='first')]
        if len(clean) < 3:
            continue

        t = clean['aligned_time'].values
        temp = clean['temperature'].values
        pain = clean['pain'].values

        trials.append({
            'trial_num': int(trial_num),
            # Carried through so per-trial fits can be split by trial type
            # (offset / onset / t1_hold / t2_hold) without a re-merge later
            'trial_type': (clean['trial_type'].iloc[0]
                           if 'trial_type' in clean.columns else None),
            'time': t,
            'temperature': temp,
            'pain': pain,
            'temp_func': interp1d(t, temp, kind='linear',
                                  bounds_error=False, fill_value='extrapolate'),
            # Only used to lay trials end-to-end for plotting; the ODE never
            # sees this, so the 1s gap is cosmetic here.
            'continuous_time': t - t[0] + time_offset,
        })
        time_offset += (t[-1] - t[0]) + 1.0

    return trials


def simulate_trials(params, trials, method='RK45', rtol=1e-6, atol=1e-9,
                    max_step=1.0, initial_pain='observed'):
    """
    Integrate the simplified Cecchi model separately for each trial, always
    starting from p=0.

    Returns a list of predicted pain arrays (one per trial, aligned to that
    trial's 'time'), or None if any trial fails to solve.

    Solver choice matters here, and the defaults are not arbitrary. Every trial
    opens with several seconds of baseline at ~32C, well below theta, where
    F(T,theta)=0 and so dp/dt=0. Given a flat start and no step ceiling, LSODA
    concludes the solution is constant, takes about four derivative
    evaluations, and steps straight over the stimulus -- returning p=0 for the
    whole trial no matter how strong the forcing. That makes the optimizer's
    objective flat and its fitted parameters meaningless.

    Capping max_step at the 1 Hz sample interval prevents the solver from
    stepping across the ramp. RK45 is used because it is both the fastest of
    the safe options and the one the others converge toward.
    """
    predictions = []
    for trial in trials:
        t = trial['time']
        p0 = (float(np.clip(trial['pain'][0], 0.0, 100.0))
              if isinstance(initial_pain, str) and initial_pain == 'observed'
              else float(initial_pain))
        sol = solve_ivp(cecchi2012_simplified, (t[0], t[-1]), [p0],
                        args=(params, trial['temp_func']),
                        t_eval=t, method=method, rtol=rtol, atol=atol,
                        max_step=max_step)
        if not sol.success:
            return None
        predictions.append(np.clip(sol.y[0], 0.0, 100.0))
    return predictions


def simulate_trials_analytic(params, trials, cap=100.0, initial_pain='observed'):
    """
    Exact solution of the simplified Cecchi model, with no ODE solver.

    Eq. 2 of Cecchi 2012,

        p'(t) = alpha_bar * F(T(t), theta) - gamma_bar * p(t),

    is first-order and linear in p, so it can be integrated in closed form
    instead of stepped through numerically. Over one sample interval h, with
    F varying linearly between F_k and F_k+1 (which is exactly how the 1 Hz
    temperature is interpolated),

        p_{k+1} = p_k e^{-gh} + a [ F_k (1-e^{-gh})/g
                                    + ((F_{k+1}-F_k)/h)(h/g - (1-e^{-gh})/g^2) ]

    This is the same equation and the same answer as solve_ivp, just arrived at
    directly. It is ~100x faster, carries no integration error (which otherwise
    corrupts the optimizer's finite-difference gradients), and cannot suffer the
    step-skipping failure described in simulate_trials().

    The step above is only exact where F is a straight line across the whole
    interval, and F = max(0, T - theta) has a kink wherever the temperature
    crosses theta: it sits at 0 for part of that second and then ramps. Those
    crossing times are therefore inserted into the grid first, so that F is
    genuinely piecewise linear on every interval and the formula is exact
    throughout. Without this the prediction is off by a few VAS around each
    crossing, which is precisely where the interesting dynamics are.

    initial_pain controls where each trial's integration starts:

      'observed' (default) -- start at the subject's own rating at t=0.
      0.0 (or any number)  -- start from rest, as the paper assumes.

    The paper starts from rest because its trials do: plosONE and kneeOA trials
    open with the subject at VAS 0 in 99.9% of cases. cLBP trials do not --
    29% of them begin above VAS 5 and 6.5% above VAS 25, because those traces
    have no pre-stimulus baseline and the recording starts at stimulus onset.
    Forcing p=0 on a trial where the subject is already at 83 makes the fit
    impossible for reasons that have nothing to do with the model.

    Either way this stays a free-running prediction: the model is given one
    number at t=0 and then runs on temperature alone. It never sees the rest
    of the observed pain, so this is not one-step-ahead prediction.

    Returns a list of predicted pain arrays, one per trial, sampled at that
    trial's original time points.
    """
    a = params['alpha_bar']
    g = params['gamma_bar']
    theta = params['theta']

    predictions = []
    for trial in trials:
        t = trial['time']
        T = trial['temperature']

        # Insert the times where T crosses theta, so F has no kink inside any
        # interval of the grid we integrate over.
        d = T - theta
        cross = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
        if len(cross):
            t_cross = t[cross] + (t[cross + 1] - t[cross]) * (
                -d[cross] / (d[cross + 1] - d[cross]))
            t_all = np.concatenate([t, t_cross])
            order = np.argsort(t_all, kind='stable')
            t_all = t_all[order]
            is_original = np.concatenate(
                [np.ones(len(t), bool), np.zeros(len(t_cross), bool)])[order]
            F = np.maximum(0.0, np.interp(t_all, t, T) - theta)
        else:
            t_all = t
            is_original = np.ones(len(t), bool)
            F = np.maximum(0.0, d)

        h = np.diff(t_all)
        E = np.exp(-g * h)

        # Contribution of the forcing over each interval
        step = (F[:-1] * (1.0 - E) / g
                + ((F[1:] - F[:-1]) / h) * (h / g - (1.0 - E) / g ** 2))

        p = np.empty_like(t_all, dtype=float)
        if isinstance(initial_pain, str) and initial_pain == 'observed':
            p[0] = float(np.clip(trial['pain'][0], 0.0, cap))
        else:
            p[0] = float(initial_pain)
        for k in range(len(h)):
            p[k + 1] = p[k] * E[k] + a * step[k]
            if p[k + 1] > cap:      # ratings are bounded above by the VAS
                p[k + 1] = cap
            elif p[k + 1] < 0.0:    # p >= 0 is imposed in the paper
                p[k + 1] = 0.0
        predictions.append(p[is_original])

    return predictions


################# Full (second-order) Cecchi 2012 model #################
def pack_trials(trials):
    """
    Pad a subject's trials into rectangular arrays, so the full model can be
    integrated for every trial at once instead of one at a time.

    That matters more than it sounds. The full model is stiff -- Petre 2017's
    fitted beta of 36.76 puts a 0.027s process inside it -- so it needs a very
    small integration step, and a per-trial call to solve_ivp costs ~2.3s per
    subject. At the thousands of evaluations a five-parameter search needs,
    that is ~790 hours for this dataset. Stepping all of a subject's trials
    forward together in numpy is ~765x faster and makes the fit feasible.

    Padding is filled with the 32C baseline so the padded columns stay inert
    (F = 0 there); 'mask' marks the real samples.
    """
    lens = [len(t['time']) for t in trials]
    n, L = len(trials), max(lens)
    T = np.full((n, L), 32.0)
    pain = np.full((n, L), np.nan)
    mask = np.zeros((n, L), dtype=bool)
    for i, tr in enumerate(trials):
        T[i, :lens[i]] = tr['temperature']
        pain[i, :lens[i]] = tr['pain']
        mask[i, :lens[i]] = True
    return {'T': T, 'pain': pain, 'mask': mask, 'lens': lens,
            'p0': np.array([float(np.clip(t['pain'][0], 0.0, 100.0))
                            for t in trials]),
            'observed_flat': np.concatenate([t['pain'] for t in trials])}


def cecchi2012_full_substeps(beta, floor=16, safety=1.2):
    """
    Integration substeps per 1s sample needed for RK4 to stay stable.

    Explicit RK4 is stable only while the step is below about 2.78/beta. With
    a whole-second step and beta near 37 the solution diverges to infinity
    rather than merely losing accuracy, so this is a hard requirement, not a
    tuning knob. 'safety' keeps the step about 2.3x inside the limit.
    """
    return max(floor, int(np.ceil(beta / safety)))


def simulate_trials_full(params, packed, cap=100.0, initial_pain='observed'):
    """
    Integrate the full second-order Cecchi 2012 model (their Eq. 1):

        p''(t) = alpha * F(T,theta) - beta * p'(t) + gamma * (T'(t) - lambda) * p(t)

    The third term is what the simplified first-order model loses. It responds
    to the RATE of temperature change: when temperature falls quickly it acts
    as a restoring force that pushes pain down faster than the decay term
    alone could, which is the mechanism behind offset analgesia. The
    simplification to Eq. 2 assumes lambda >> 1, which collapses (T' - lambda)
    to a constant and removes that mechanism entirely.

    T is linearly interpolated between 1 Hz samples, so T' is constant within
    each interval and is taken as the per-second difference.

    params : dict with alpha, beta, gamma, lam, theta
    packed : output of pack_trials()

    Returns the predicted pain as an (n_trials, L) array.
    """
    T = packed['T']
    n, L = T.shape
    a, b = params['alpha'], params['beta']
    g, lam, theta = params['gamma'], params['lam'], params['theta']

    sub = cecchi2012_full_substeps(b)
    h = 1.0 / sub

    if isinstance(initial_pain, str) and initial_pain == 'observed':
        p = packed['p0'].copy()
    else:
        p = np.full(n, float(initial_pain))
    v = np.zeros(n)                      # pain is not changing at trial onset

    out = np.zeros((n, L))
    out[:, 0] = p
    for k in range(L - 1):
        T0 = T[:, k]
        rate = T[:, k + 1] - T0          # dT/dt, constant across this second
        c = g * (rate - lam)             # the rate term's coefficient
        for j in range(sub):
            s0 = j * h

            def deriv(p_, v_, s):
                F = np.maximum(0.0, T0 + rate * s - theta)
                return v_, a * F - b * v_ + c * p_

            k1p, k1v = deriv(p, v, s0)
            k2p, k2v = deriv(p + h / 2 * k1p, v + h / 2 * k1v, s0 + h / 2)
            k3p, k3v = deriv(p + h / 2 * k2p, v + h / 2 * k2v, s0 + h / 2)
            k4p, k4v = deriv(p + h * k3p, v + h * k3v, s0 + h)
            p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            # Cecchi impose p >= 0 by setting p' = 0 when p would go negative;
            # clipping the state each substep is the discrete equivalent.
            np.clip(p, 0.0, cap, out=p)
        out[:, k + 1] = p

    return out


def full_predictions_flat(params, packed, **kwargs):
    """Predicted pain for a subject, concatenated trial by trial (padding removed)."""
    out = simulate_trials_full(params, packed, **kwargs)
    return np.concatenate([out[i, :packed['lens'][i]]
                           for i in range(len(packed['lens']))])


def fit_metrics(observed, predicted):
    """
    Goodness-of-fit measures for one trial or one subject.

    'r' is the zero-lag Pearson correlation, which is what Cecchi 2012 reports
    as model accuracy (0.92 simple / 0.88 complex stimuli for the first-order
    model), so it is the number to compare against the paper. r2 and sse are
    kept alongside it because r only judges shape: a prediction with the right
    shape but the wrong amplitude scores a high r and a poor r2, and knowing
    which of the two is failing tells you what to fix.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sse = float(np.sum((observed - predicted) ** 2))
    if np.std(observed) == 0 or np.std(predicted) == 0:
        r = np.nan          # a flat trace has no correlation to speak of
    else:
        r = float(np.corrcoef(observed, predicted)[0, 1])
    return {'r': r,
            'r2': float(r2_score(observed, predicted)),
            'mse': float(np.mean((observed - predicted) ** 2)),
            'sse': sse}


################# Parameter Optimization Functions #################
def optimize_cecchi_simplified(subject_data, threshold=None, initial_params=None,
                               use_multiple_starts=False, n_starts=5, verbose=True,
                               optimizer='de', random_state=0, bounds=None):
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
        If True, tries multiple random starting points (optimizer='multistart')
    n_starts : int
        Number of random starting points to try (optimizer='multistart')
    verbose : bool
        Print optimization progress
    optimizer : {'de', 'multistart'}
        'de' (default) uses differential evolution, a global search. Preferred
        because the objective has flat regions -- any theta above the hottest
        stimulus makes the forcing zero everywhere, so the cost stops
        responding to the parameters and a gradient-following search stalls
        there. 'multistart' is the older multi-start L-BFGS-B path.
    random_state : int or None
        Seed, so a given subject's fit is reproducible.

    Returns:
    --------
    best_params : dict
        Optimized parameters with keys: alpha_bar, gamma_bar, theta, mse,
        r (zero-lag correlation, the measure Cecchi 2012 reports), r2, sse,
        success, and trial_fits (the same measures per trial)
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
    
    # Split into independent per-trial blocks. Each trial is integrated from
    # p=0 rather than inheriting the previous trial's pain -- see
    # prepare_trials_for_optimization() for why.
    trials = prepare_trials_for_optimization(subject_data)
    if len(trials) == 0:
        if verbose:
            print("    ❌ No usable trials for this subject.")
        return None, None

    # Laid end-to-end purely so results/plots keep their existing shape
    time_data = np.concatenate([t['continuous_time'] for t in trials])
    temp_data = np.concatenate([t['temperature'] for t in trials])
    pain_data = np.concatenate([t['pain'] for t in trials])
    trial_index = np.concatenate([np.full(len(t['time']), t['trial_num']) for t in trials])
    concatenated_data = trials

    if verbose:
        print(f"    Data: {len(time_data)} time points across {len(trials)} trials")
        print(f"    Pain range: {pain_data.min():.1f} to {pain_data.max():.1f}")

    # Define objective function
    def objective(params_array):
        """ MSE between observed and predicted pain, pooled over trials"""
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

        try:
            predictions = simulate_trials_analytic(model_params, trials)
            if predictions is None:
                return 1e6
            model_pain = np.concatenate(predictions)
            mse = np.mean((pain_data - model_pain) ** 2)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    # Parameter bounds.
    #
    # The previous limits -- alpha_bar (0.5, 8.0), gamma_bar (0.01, 0.6) --
    # came from parameter_search.py, which ran against the LSODA objective
    # that silently returned a flat zero prediction. They were therefore
    # derived from fits that carried no information, and in practice most
    # subjects came back pinned exactly against them, meaning the bound rather
    # than the data was choosing the answer. These are wide enough to be
    # inactive for a well-behaved subject; landing on one is now recorded in
    # 'at_bounds' and should be read as a warning about that fit.
    #
    # theta stops at 30C because the stimulus baseline is ~32C: a threshold
    # below baseline means the forcing never switches off, which is not
    # physiologically meaningful and signals a failed fit rather than a low
    # pain threshold.
    ALPHA_BOUNDS = (0.001, 50.0)
    GAMMA_BOUNDS = (0.001, 3.0)
    THETA_BOUNDS = (30.0, 52.0)

    if optimize_theta:
        default_bounds = [ALPHA_BOUNDS, GAMMA_BOUNDS, THETA_BOUNDS]
        param_names = ['alpha_bar', 'gamma_bar', 'theta']
        x0_default = [initial_params['alpha_bar'],
                      initial_params['gamma_bar'],
                      initial_params['theta']]
    else:
        default_bounds = [ALPHA_BOUNDS, GAMMA_BOUNDS]
        param_names = ['alpha_bar', 'gamma_bar']
        x0_default = [initial_params['alpha_bar'],
                      initial_params['gamma_bar']]

    # A caller can widen the limits -- e.g. to refit a subject whose parameters
    # landed on a bound, to tell "the bound was too tight" apart from "the fit
    # failed". The length is checked because a caller passing three bounds to a
    # two-parameter fit would otherwise map theta's range onto nothing and fail
    # silently rather than loudly.
    bounds = default_bounds if bounds is None else [tuple(b) for b in bounds]
    if len(bounds) != len(param_names):
        raise ValueError(
            f'bounds has {len(bounds)} entries but this fit has '
            f'{len(param_names)} parameters {param_names}; theta is '
            f'{"fitted" if optimize_theta else "fixed, so pass only 2"}')
    
    best_result = None
    best_cost = np.inf
    best_start_idx = -1

    if optimizer == 'de':
        # Differential evolution: a global search that never asks "which way is
        # downhill". That matters because this objective has large dead flat
        # regions -- whenever theta sits above the hottest stimulus, F is 0
        # everywhere, the model predicts a flat zero, and the cost is constant.
        # A gradient follower that steps into one of those plateaus stops dead,
        # which is how subjects were previously coming back with theta pinned
        # at the upper bound and a flat prediction. polish=True finishes with a
        # local L-BFGS-B refinement, so precision is not sacrificed.
        from scipy.optimize import differential_evolution
        best_result = differential_evolution(
            objective, bounds,
            seed=random_state, polish=True, tol=1e-8,
            maxiter=1000, popsize=20, init='sobol')
        best_cost = best_result.fun
        best_start_idx = 0
        if verbose:
            print(f"    Global search: cost → {best_result.fun:.1f} "
                  f"({best_result.nfev} evals)")
    else:
        # Generate starting points
        if use_multiple_starts:
            rng = np.random.default_rng(random_state)
            starting_points = [[rng.uniform(b[0], b[1]) for b in bounds]
                               for _ in range(n_starts)]
        else:
            starting_points = [x0_default]

        # Try each starting point
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
            'best_start_index': best_start_idx,
            # Any parameter sitting on a bound means the bound, not the data,
            # picked that value -- treat such a fit as suspect.
            'at_bounds': [name for name, val, (lo, hi)
                          in zip(param_names, best_result.x, bounds)
                          if abs(val - lo) < 1e-6 * max(1.0, abs(lo))
                          or abs(val - hi) < 1e-6 * max(1.0, abs(hi))]
        }

        # Re-run model once to get predictions for storage
        model_params = {
        'alpha_bar': alpha_bar,
        'gamma_bar': gamma_bar,
        'theta': theta
        }

        try:
            if verbose:
                print(f"    Re-running model to save predictions...")
            predictions = simulate_trials_analytic(model_params, trials)
            if predictions is not None:
                model_pain = np.concatenate(predictions)
                best_params['model_data'] = {
                    'time': time_data,
                    'predicted_pain': model_pain,
                    'observed_pain': pain_data,
                    'temperature': temp_data,
                    'trial_num': trial_index
                }
                # Whole-subject fit. 'r' is the zero-lag correlation Cecchi
                # 2012 reports, so it is the number comparable to their
                # 0.92 / 0.88 for the first-order model.
                best_params.update(fit_metrics(pain_data, model_pain))

                # Per-trial fit quality: the input for asking whether the
                # model fails on particular trial types, in particular groups,
                # or progressively across repeated trials.
                best_params['trial_fits'] = [
                    {'trial_num': trial['trial_num'],
                     'trial_type': trial.get('trial_type'),
                     'n_points': len(pred),
                     **fit_metrics(trial['pain'], pred)}
                    for trial, pred in zip(trials, predictions)
                ]
                if verbose:
                    print(f"    ✅ Model data saved successfully!")
            else:
                print(f"    ❌ Final model solve failed")
        except Exception as e:
            print(f"    ❌ Error during final solve: {e}")
            import traceback
            traceback.print_exc()

        if verbose:
            print(f"\n   ✅ Optimization successful (start #{best_start_idx+1}):")
            print(f"      ᾱ={alpha_bar:.4f}, γ̄={gamma_bar:.4f}, θ={theta:.2f}")
            print(f"      r={best_params.get('r', np.nan):.3f} "
                  f"(Cecchi 2012 report 0.92/0.88 for this model), "
                  f"r²={best_params.get('r2', np.nan):.3f}")
            print(f"      MSE={best_result.fun:.2f} ({best_result.nfev} evals, {best_result.nit} iters)")
    else:
        if verbose:
            print(f"\n   ❌ Optimization failed.")
        best_params = None
    return best_params, best_result








def seed_full_from_simplified(alpha_bar, gamma_bar, theta, scale=30.0):
    """
    Express a fitted Eq. 2 (first-order) parameter set as an equivalent point
    in the full Eq. 1 parameter space.

    Eq. 1 NESTS Eq. 2: making beta and lambda both large recovers the
    simplification exactly. Setting beta = lambda = scale and

        alpha = alpha_bar * scale,   gamma = gamma_bar

    gives gamma*lambda/beta = gamma_bar and (T' - lambda) ~ -lambda, which is
    the reduction. Verified against the analytic Eq. 2 solution: r agrees to
    3-4 decimal places at scale = 30.

    This is used for two things. It is a sensible starting point for the
    five-parameter search, and more importantly it is a FLOOR: evaluating it
    guarantees the full model is never scored worse than the simplified one.
    That matters because the 5D search does not reliably converge on its own
    -- widening the bounds once produced a *worse* fit for a subject, which is
    impossible at a true optimum and so diagnostic of the optimiser stalling.
    Without the floor, a reported "the full model is worse here" could be
    optimiser noise rather than a fact about the data.

    Petre 2017's published values are deliberately NOT used as the seed. Their
    gamma*lambda is 3.45e-4, implying a decay time constant of ~106,000s -- on
    a 60s trial that model rises and never returns, which is not what these
    traces do. Those values come from a different study's group mean and sit
    in a region of parameter space this data rules out.
    """
    return {'alpha': alpha_bar * scale,
            'beta': scale,
            'gamma': gamma_bar,
            'lam': scale,
            'theta': theta}


def optimize_cecchi_full(subject_data, initial_params=None, bounds=None,
                         popsize=12, maxiter=60, seed=0, verbose=True):
    """
    Fit the full second-order Cecchi 2012 model (Eq. 1) to one subject.

    Parameters are [alpha, beta, gamma, lam, theta], fitted by differential
    evolution as for Eq. 2 -- the objective has the same flat regions wherever
    theta exceeds the hottest stimulus, so a gradient-following search stalls.

    Two identifiability cautions worth carrying into the interpretation:

      - gamma and lambda enter only as 'gamma*lambda' (the decay rate) and
        'gamma' (the scale of the rate term). They can only be told apart
        where the rate term measurably matters, so on trials with little
        temperature movement they trade off against each other.
      - beta sets a timescale of 1/beta, which for plausible values is ~0.03s.
        The data are sampled at 1 Hz, so beta is constrained only indirectly,
        through the shape of the slow envelope.

    Returns (best_params, result) with the same keys as
    optimize_cecchi_simplified, so downstream code can treat them alike.
    """
    from scipy.optimize import differential_evolution, minimize

    trials = prepare_trials_for_optimization(subject_data)
    if not trials:
        if verbose:
            print('    No usable trials for this subject.')
        return None, None
    packed = pack_trials(trials)
    observed = packed['observed_flat']

    if bounds is None:
        bounds = [
            (0.001, 500.0),   # alpha: drive. alpha/beta sets the rise rate
            # beta: damping. The lower end matters -- small beta is the weakly
            # damped, genuinely oscillatory regime, and in piloting that was
            # where the full model actually beat Eq. 2 (one subject sat on a
            # 0.5 floor and gained the most), so the floor is well below it.
            # The ceiling is practical: integration substeps scale with beta,
            # and 1 Hz data cannot constrain a timescale that fast regardless.
            (0.05, 60.0),
            (1e-4, 50.0),     # gamma: scale of the rate term
            # lam: temperature rate above which a change counts as alarming
            # (degC/s). Large lam makes (T' - lambda) effectively constant,
            # which IS Cecchi's assumption (b) -- i.e. the full model
            # collapsing back to Eq. 2. The ceiling is set high so that
            # reaching it reads as the data preferring the simplified form,
            # rather than as the optimiser running out of room. Fits that land
            # there should be interpreted, not discarded: excluding them would
            # keep only the subjects where the full model wins and inflate the
            # model comparison.
            (1e-4, 50.0),
            (30.0, 52.0),     # theta: below the 32C baseline is meaningless
        ]
    names = ['alpha', 'beta', 'gamma', 'lam', 'theta']

    def objective(x):
        params = dict(zip(names, x))
        try:
            pred = full_predictions_flat(params, packed)
            mse = np.mean((observed - pred) ** 2)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    kwargs = dict(seed=seed, polish=True, tol=1e-6, init='sobol',
                  popsize=popsize, maxiter=maxiter)
    x0 = None
    if initial_params is not None:
        x0 = np.array([float(np.clip(initial_params[n], b[0], b[1]))
                       for n, b in zip(names, bounds)])
        try:
            result = differential_evolution(objective, bounds, x0=x0, **kwargs)
        except TypeError:      # older scipy has no x0
            result = differential_evolution(objective, bounds, **kwargs)
    else:
        result = differential_evolution(objective, bounds, **kwargs)

    if result is None or not np.isfinite(result.fun):
        return None, result

    # Eq. 1 nests Eq. 2, so the point equivalent to the subject's first-order
    # fit is always available and the full model can never truly do worse.
    # The 5D global search does not reliably reach the optimum on its own, so
    # that point is polished locally and kept if it wins -- otherwise an
    # apparent "the full model is worse for this subject" would just be the
    # optimiser having stalled. Both candidates are scored on the same
    # objective, so this only ever improves the answer.
    best_x, best_fun = result.x, result.fun
    if x0 is not None:
        polished = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        for cand_x, cand_fun in ((x0, objective(x0)),
                                 (polished.x, polished.fun)):
            if np.isfinite(cand_fun) and cand_fun < best_fun:
                best_x, best_fun = np.asarray(cand_x), float(cand_fun)
        if verbose and best_fun < result.fun - 1e-9:
            print(f'    (global search stalled; kept the Eq.2-seeded local fit, '
                  f'MSE {result.fun:.1f} -> {best_fun:.1f})')

    params = dict(zip(names, best_x))
    predictions = simulate_trials_full(params, packed)
    pred_flat = np.concatenate([predictions[i, :packed['lens'][i]]
                                for i in range(len(trials))])

    best = dict(params)
    best.update(fit_metrics(observed, pred_flat))
    best.update({
        'success': bool(result.success),
        'n_trials': len(trials),
        'n_points': len(observed),
        'n_evals': int(result.nfev),
        'at_bounds': [n for n, v, (lo, hi) in zip(names, result.x, bounds)
                      if abs(v - lo) < 1e-6 * max(1.0, abs(lo))
                      or abs(v - hi) < 1e-6 * max(1.0, abs(hi))],
        'trial_fits': [
            {'trial_num': tr['trial_num'], 'trial_type': tr.get('trial_type'),
             'n_points': packed['lens'][i],
             **fit_metrics(tr['pain'], predictions[i, :packed['lens'][i]])}
            for i, tr in enumerate(trials)],
        'model_data': {
            'time': np.concatenate([t['continuous_time'] for t in trials]),
            'predicted_pain': pred_flat,
            'observed_pain': observed,
            'temperature': np.concatenate([t['temperature'] for t in trials]),
            'trial_num': np.concatenate([np.full(len(t['time']), t['trial_num'])
                                         for t in trials]),
        },
    })

    if verbose:
        print(f"    α={best['alpha']:.3f} β={best['beta']:.3f} "
              f"γ={best['gamma']:.4f} λ={best['lam']:.4f} θ={best['theta']:.2f}")
        print(f"    r={best['r']:.3f}  r²={best['r2']:.3f}  MSE={best['mse']:.1f} "
              f"({result.nfev} evals)")
        if best['at_bounds']:
            print(f"    ⚠️  on bounds: {best['at_bounds']}")

    return best, result


################# Plotting Functions ##################
def plot_subject_trials(subject_data, params, subject_uid=None, max_trials=12,
                        ncols=4, save_path=None, show_temperature=True):
    """
    Plot observed vs model-predicted pain for each of a subject's trials.

    One panel per trial, so the per-trial fit is visible rather than being
    averaged away. Correlation is shown per panel because that is the measure
    Cecchi 2012 reports, and it is computed per trial for the same reason --
    pooling a subject's trials into one series also asks the model to get the
    relative amplitude between trials right, which is a different (and harder)
    question than whether it captures the shape within a trial.

    Parameters:
    -----------
    subject_data : DataFrame
        One subject's rows from the combined traces table.
    params : dict
        Fitted parameters with keys alpha_bar, gamma_bar, theta.
    subject_uid : str, optional
        Label for the figure title.
    max_trials : int
        Cap on panels, so a 24-trial subject does not produce an unreadable grid.
    show_temperature : bool
        Overlay the stimulus on a secondary axis.
    """
    trials = prepare_trials_for_optimization(subject_data)
    if not trials:
        print(f"No usable trials for {subject_uid}")
        return None

    model_params = {k: params[k] for k in ('alpha_bar', 'gamma_bar', 'theta')}
    predictions = simulate_trials_analytic(model_params, trials)

    n = min(len(trials), max_trials)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows),
                             squeeze=False, sharey=True)

    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis('off')
            continue

        trial, pred = trials[i], predictions[i]
        m = fit_metrics(trial['pain'], pred)

        ax.plot(trial['time'], trial['pain'], color='k', lw=1.6, label='observed')
        ax.plot(trial['time'], pred, color='crimson', lw=1.6, label='model')
        ax.set_ylim(-5, 105)

        if show_temperature:
            ax2 = ax.twinx()
            ax2.plot(trial['time'], trial['temperature'], color='0.6',
                     lw=1.0, ls='--', zorder=0)
            ax2.axhline(model_params['theta'], color='steelblue', lw=0.8,
                        ls=':', zorder=0)
            ax2.set_ylim(30, 52)
            ax2.set_yticks([] if (i % ncols) != ncols - 1 else [32, 40, 48])
            if (i % ncols) == ncols - 1:
                ax2.set_ylabel('°C', color='0.5', fontsize=8)
                ax2.tick_params(labelsize=7, colors='0.5')

        title = f"trial {trial['trial_num']}"
        if trial.get('trial_type'):
            title += f" · {trial['trial_type']}"
        ax.set_title(f"{title}\nr={m['r']:.2f}  r²={m['r2']:.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
        if i % ncols == 0:
            ax.set_ylabel('VAS')
        if i // ncols == nrows - 1:
            ax.set_xlabel('time (s)')
        if i == 0:
            ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

    overall = fit_metrics(np.concatenate([t['pain'] for t in trials]),
                          np.concatenate(predictions))
    per_trial_r = np.nanmedian([fit_metrics(t['pain'], p)['r']
                                for t, p in zip(trials, predictions)])
    fig.suptitle(
        f"{subject_uid or 'subject'}  —  "
        f"ᾱ={model_params['alpha_bar']:.2f}, γ̄={model_params['gamma_bar']:.3f}, "
        f"θ={model_params['theta']:.1f}°C   |   "
        f"median per-trial r={per_trial_r:.3f}, pooled r={overall['r']:.3f}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   saved {save_path}")
    return fig



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
def analyze_subject_data(subject_id, data_df, detailed=True):
    """
    Comprehensive analysis of a subject's data to identify potential issues
    """
    # subject_uid is the unique key; 'subject' is an offset-numeric id that
    # does not identify a person on its own. Fall back for older tables.
    key = 'subject_uid' if 'subject_uid' in data_df.columns else 'subject'
    subject_data = data_df[data_df[key] == subject_id].copy()

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


