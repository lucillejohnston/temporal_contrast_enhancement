import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy import stats

def plot_temp_pain(df, title=None, temp_col='temperature', pain_col='pain', time_col=None):
    """
    Plot temperature (left y-axis) and pain (right y-axis) from a DataFrame.
    
    Parameters:
        df: DataFrame containing the data.
        title: Optional plot title.
        temp_col: Name of the temperature column.
        pain_col: Name of the pain column.
        time_col: Name of the time column (if not index). If None, use df.index.
    """
    if time_col is not None:
        x = df[time_col]
    else:
        x = df.index

    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Temperature on left y-axis
    ax1.plot(x, df[temp_col], color='blue', label='Temperature')
    ax1.set_xlabel("Aligned Time (s)")
    ax1.set_ylabel("Temperature", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Pain on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, df[pain_col], color='red', label='Pain')
    ax2.set_ylabel("Pain", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    if title:
        plt.title(title)
    fig.tight_layout()
    plt.show()


def plot_temp_pain_separate(df, title=None, temp_col='temperature', pain_col='pain', time_col=None):
    """
    Plot temperature and pain separately from a DataFrame.
    
    Parameters:
        df: DataFrame containing the data.
        title: Optional plot title.
        temp_col: Name of the temperature column.
        pain_col: Name of the pain column.
        time_col: Name of the time column (if not index). If None, use df.index.
    """
    if time_col is not None:
        x = df[time_col]
    else:
        x = df.index

    fig, (ax_temp, ax_pain) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Temperature plot
    ax_temp.plot(x, df[temp_col], color='blue', label='Temperature')
    ax_temp.set_ylabel("Temperature", color='blue')
    ax_temp.tick_params(axis='y', labelcolor='blue')
    
    # Pain plot
    ax_pain.plot(x, df[pain_col], color='red', label='Pain')
    ax_pain.set_ylabel("Pain", color='red')
    ax_pain.tick_params(axis='y', labelcolor='red')
    
    if title:
        plt.suptitle(title)
    
    fig.tight_layout()
    plt.show()


def create_correlation_scatter(df, x_col, y_col, title=None, xlabel=None, ylabel=None,
                               filter_col=None, filter_val=None, figsize=(8,6)):
    """
    Create a correlation scatter plot with regression line and statistics
    Parameters:
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    title : str, optional
        Plot title (auto-generated if None)
    xlabel : str, optional
        X-axis label (uses column name if None)
    ylabel : str, optional
        Y-axis label (uses column name if None)
    filter_col : str, optional
        Column name to filter data
    filter_val : any, optional
        Value to filter data
    figsize : tuple, optional
        Figure size

    Returns:
    dict: correlation statistics (r_value, p_value, slope, intercept)
    """
    # Apply filter if specified
    if filter_col and filter_val:
        if isinstance(filter_val, list):
            subset = df[df[filter_col].isin(filter_val)]
        else:
            subset = df[df[filter_col] == filter_val]
    else:
        subset = df.copy()

    # Remove null values
    mask = subset[x_col].notnull() & subset[y_col].notnull()
    x = subset.loc[mask, x_col]
    y = subset.loc[mask, y_col]
    
    # Calculate correlation
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"{x_col} vs. {y_col}:")
    print(f"  Correlation r: {r_value:.3f}, p-value: {p_value:.3g}")

    # Create scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.7)
    if p_value < 0.05:                                          # If significant, add regression line
        x_vals = np.array([x.min(), x.max()])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, color='red')
    plt.text(0.05, 0.95,                                        # Add statistics text
             f"p = {p_value:.3g}\n$R$ = {r_value:.2f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.title(title if title else f"{y_col} vs. {x_col}")       # Add titles and labels
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.tight_layout()
    plt.show()
    return {
        "r_value": r_value,
        "p_value": p_value,
        "slope": slope,
        "intercept": intercept
    }



def plot_trial_comparison(structured_data, df, stepped_type, control_type, reference_suffix, 
                         specific_subject=None, specific_control_trial=None):
    """
    Plot comparison between stepped trial and its corresponding control trial.
    
    Args:
        structured_data: Dictionary with trial metrics organized by subject/trial_num
        df: DataFrame with raw trial data
        stepped_type: 'offset' or 'inv'
        control_type: 't1_hold' or 't2_hold'
        reference_suffix: 'offset', 'inv', etc. for column naming
        specific_subject: Optional specific subject ID to plot (if None, picks random)
        specific_control_trial: Optional specific control trial number to plot (if None, picks first)
    """
    
    if specific_subject is not None:
        # Use specific subject if provided
        if specific_subject not in structured_data:
            print(f"Subject {specific_subject} not found!")
            print(f"Available subjects: {list(structured_data.keys())}")
            return
        
        subject_trials = structured_data[specific_subject]
        trial_types = [trial['trial_type'] for trial in subject_trials.values()]
        has_stepped = stepped_type in trial_types
        has_control = control_type in trial_types
        
        if not (has_stepped and has_control):
            print(f"Subject {specific_subject} doesn't have both {stepped_type} and {control_type} trials!")
            return
        
        subject = specific_subject
        print(f"Using specified subject: {subject}")
    else:
        # Find subjects with both trial types (original logic)
        subjects_with_both = []
        for subject_id, trials in structured_data.items():
            trial_types = [trial['trial_type'] for trial in trials.values()]
            has_stepped = stepped_type in trial_types
            has_control = control_type in trial_types
            if has_stepped and has_control:
                subjects_with_both.append(subject_id)
        
        if not subjects_with_both:
            print(f"No subjects found with both {stepped_type} and {control_type} trials!")
            return
        
        # Pick random subject
        subject = np.random.choice(subjects_with_both)
        print(f"Randomly selected subject: {subject}")
    
    # Get control trials for this subject
    control_trials = []
    for trial_num, trial_data in structured_data[subject].items():
        if trial_data['trial_type'] == control_type:
            control_trials.append((trial_num, trial_data))
    
    if len(control_trials) == 0:
        print(f"No {control_type} trials found!")
        return
    
    if specific_control_trial is not None:
        # Use specific control trial if provided
        control_trial_found = False
        for trial_num, trial_data in control_trials:
            if trial_num == specific_control_trial:
                control_trial_num, control_trial = trial_num, trial_data
                control_trial_found = True
                print(f"Using specified control trial: {control_trial_num}")
                break
        
        if not control_trial_found:
            print(f"Control trial {specific_control_trial} not found for subject {subject}!")
            available_trials = [t[0] for t in control_trials]
            print(f"Available {control_type} trials: {available_trials}")
            return
    else:
        # Pick first control trial (original logic)
        control_trial_num, control_trial = control_trials[0]
        print(f"Using first available control trial: {control_trial_num}")
    
    # Get reference column name
    if control_type == 't1_hold':
        ref_col = f'reference_trial_num_{reference_suffix}'
    else:  # t2_hold
        ref_col = 'reference_trial_num'
    
    referenced_stepped_num = control_trial.get(ref_col)
    print(f"{control_type} trial {control_trial_num} references {stepped_type} trial {referenced_stepped_num}")
    
    if referenced_stepped_num is None or pd.isna(referenced_stepped_num):
        print(f"{control_type} trial has no {stepped_type} reference!")
        return
    
    # Find the referenced stepped trial
    referenced_stepped_num = int(referenced_stepped_num)
    if referenced_stepped_num not in structured_data[subject]:
        print(f"Referenced {stepped_type} trial {referenced_stepped_num} not found!")
        return
    
    stepped_trial = structured_data[subject][referenced_stepped_num]
    if stepped_trial['trial_type'] != stepped_type:
        print(f"Referenced trial {referenced_stepped_num} is not a {stepped_type} trial!")
        return
    
    stepped_trial_num = referenced_stepped_num
    
    print(f"Found matching pair: {stepped_type} {stepped_trial_num} ↔ {control_type} {control_trial_num}")
    
    # Get time series data for both trials
    stepped_data = df[(df['subject'] == subject) & 
                     (df['trial_num'] == stepped_trial_num)].sort_values('aligned_time')
    control_data = df[(df['subject'] == subject) & 
                     (df['trial_num'] == control_trial_num)].sort_values('aligned_time')
    
    # Create subplot with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Function to plot a single trial
    def plot_single_trial(ax, trial_data, trial_metrics, trial_name, is_stepped=True):
        # Create twin axis for temperature
        ax_temp = ax.twinx()
        
        # Plot pain on left axis (blue)
        pain_line = ax.plot(trial_data['aligned_time'], trial_data['pain'], 
                           color='blue', linewidth=2, label='Pain Rating')[0]
        
        # Plot temperature on right axis (red)
        temp_line = ax_temp.plot(trial_data['aligned_time'], trial_data['temperature'], 
                                color='red', alpha=0.7, linewidth=2, label='Temperature')[0]
        
        # Set axis limits and labels
        ax.set_ylim(0, 100)
        ax.set_ylabel('Pain Rating', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        ax_temp.set_ylim(29, 50)
        ax_temp.set_ylabel('Temperature (°C)', color='red')
        ax_temp.tick_params(axis='y', labelcolor='red')
        
        # Mark periods A, B, C
        for period in ['A', 'B', 'C']:
            start = trial_metrics[f'{period}_start']
            end = trial_metrics[f'{period}_end']
            color = {'A': 'yellow',
                     'B': 'lightgreen',
                     'C': 'lightblue'}[period]
            ax.axvspan(start, end, color=color, alpha=0.1, label=f'Period {period}')
        
        if is_stepped:
            # Mark local extrema for stepped trials (using abs_ values)
            if trial_metrics.get('abs_min_time') is not None:
                ax.axvline(trial_metrics['abs_min_time'], color='green', linestyle='--', 
                          linewidth=2, label=f"Local Min ({trial_metrics['abs_min_val']:.1f})")
                ax.plot(trial_metrics['abs_min_time'], trial_metrics['abs_min_val'], 
                       'go', markersize=8)
            
            if trial_metrics.get('abs_max_time') is not None:
                ax.axvline(trial_metrics['abs_max_time'], color='orange', linestyle='--', 
                          linewidth=2, label=f"Local Max ({trial_metrics['abs_max_val']:.1f})")
                ax.plot(trial_metrics['abs_max_time'], trial_metrics['abs_max_val'], 
                       'o', color='orange', markersize=8)
        else:
            # Mark time-yoked extrema for control trials
            min_time_col = f'time_yoked_min_time_{reference_suffix}'
            min_val_col = f'time_yoked_min_val_{reference_suffix}'
            max_time_col = f'time_yoked_max_time_{reference_suffix}'
            max_val_col = f'time_yoked_max_val_{reference_suffix}'
            
            if trial_metrics.get(min_time_col) is not None:
                ax.axvline(trial_metrics[min_time_col], color='green', 
                          linestyle='--', linewidth=2, alpha=0.7,
                          label=f"Time-yoked at 'Min' time ({trial_metrics[min_val_col]:.1f})")
                ax.plot(trial_metrics[min_time_col], trial_metrics[min_val_col], 
                       'go', markersize=6, alpha=0.7)
            
            if trial_metrics.get(max_time_col) is not None:
                ax.axvline(trial_metrics[max_time_col], color='orange', 
                          linestyle='--', linewidth=2, alpha=0.7,
                          label=f"Time-yoked at 'Max' time ({trial_metrics[max_val_col]:.1f})")
                ax.plot(trial_metrics[max_time_col], trial_metrics[max_val_col], 
                       'o', color='orange', markersize=6, alpha=0.7)
            
            # Mark absolute extrema
            if trial_metrics.get('abs_max_time') is not None:
                ax.axvline(trial_metrics['abs_max_time'], color='purple', 
                          linestyle='--', linewidth=2,
                          label=f"True Max ({trial_metrics['abs_max_val']:.1f})")
                ax.plot(trial_metrics['abs_max_time'], trial_metrics['abs_max_val'], 
                       'rs', markersize=8)
            
            if trial_metrics.get('abs_min_time') is not None:
                ax.axvline(trial_metrics['abs_min_time'], color='yellow', 
                          linestyle='--', linewidth=2,
                          label=f"True Min ({trial_metrics['abs_min_val']:.1f})")
                ax.plot(trial_metrics['abs_min_time'], trial_metrics['abs_min_val'], 
                       's', color='yellow', markersize=8)
        
        # Set title
        ax.set_title(f'Subject {subject} - {trial_name}')
        
        # Create combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_temp.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    # Plot both trials
    plot_single_trial(ax1, stepped_data, stepped_trial, 
                     f'{stepped_type.title()} Trial {stepped_trial_num}', is_stepped=True)
    
    plot_single_trial(ax2, control_data, control_trial, 
                     f'{control_type.upper()} Trial {control_trial_num} (ref: {stepped_type} {stepped_trial_num})', 
                     is_stepped=False)
    
    # Set shared x-label
    ax2.set_xlabel('Aligned Time (s)')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary metrics
    print(f"\n=== METRICS SUMMARY ===")
    print(f"{stepped_type.upper()} TRIAL {stepped_trial_num}:")
    print(f"  Local Min: {stepped_trial['abs_min_val']:.2f} at t={stepped_trial['abs_min_time']:.2f}s")
    print(f"  Local Max: {stepped_trial['abs_max_val']:.2f} at t={stepped_trial['abs_max_time']:.2f}s")
    print(f"  Peak-to-peak: {stepped_trial['abs_peak_to_peak']:.2f}")
    print(f"  AUC Total: {stepped_trial['auc_total']:.2f}")
    
    print(f"\n{control_type.upper()} TRIAL {control_trial_num}:")
    
    # Print time-yoked metrics
    min_val_col = f'time_yoked_min_val_{reference_suffix}'
    max_val_col = f'time_yoked_max_val_{reference_suffix}'
    min_time_col = f'time_yoked_min_time_{reference_suffix}'
    max_time_col = f'time_yoked_max_time_{reference_suffix}'
    pp_col = f'time_yoked_peak_to_peak_{reference_suffix}'
    
    if control_trial.get(min_val_col) is not None:
        print(f"  Time-yoked at 'Min' time: {control_trial[min_val_col]:.2f} at t={control_trial[min_time_col]:.2f}s")
        print(f"  Time-yoked at 'Max' time: {control_trial[max_val_col]:.2f} at t={control_trial[max_time_col]:.2f}s")
        print(f"  Time-yoked Peak-to-peak: {control_trial[pp_col]:.2f}")
    
    if control_trial.get('abs_max_val') is not None:
        print(f"  True Max: {control_trial['abs_max_val']:.2f} at t={control_trial['abs_max_time']:.2f}s")
        print(f"  True Min: {control_trial['abs_min_val']:.2f} at t={control_trial['abs_min_time']:.2f}s")
    
    print(f"  AUC Total: {control_trial['auc_total']:.2f}")
    print(f"  Time-yoked Normalized Pain Change: {control_trial.get('time_yoked_normalized_pain_change', 'N/A'):.1f}%")
    print(f"  Absolute Normalized Pain Change: {control_trial.get('abs_normalized_pain_change', 'N/A'):.1f}%")

