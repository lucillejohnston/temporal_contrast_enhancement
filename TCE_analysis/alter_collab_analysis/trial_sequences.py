#%%
# ========================================================
# CONFIGURATION
# ========================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os, json
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_functions import *  

# File paths
TRIAL_METRICS_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_metrics.json'
TRIAL_DATA_PATH = '/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/trial_data_cleaned_aligned.json'

# Load the metrics data
with open(TRIAL_METRICS_PATH, 'r') as f:
    metrics_data = json.load(f)

with open(TRIAL_DATA_PATH, 'r') as f:
    trial_data = json.load(f)

# Convert to DataFrame
trial_metrics_df = {}
records = []
for subject_id, trials in metrics_data.items():
    for trial_num, trial_data in trials.items():
        record = {
            'subject': int(subject_id),
            'trial_num': int(trial_num),
            **trial_data
        }
        records.append(record)
trial_metrics_df = pd.DataFrame(records)


#%%
# ========================================================
# Calculate preceding trial metrics 
# ========================================================
def get_preceding_value(row, col):
    prev_trial_num = row['trial_num'] - 1
    subject = row['subject']
    prev_row = trial_metrics_df[
        (trial_metrics_df['subject'] == subject) & 
        (trial_metrics_df['trial_num'] == prev_trial_num)
    ]
    if not prev_row.empty:
        return prev_row.iloc[0][col]
    return None

# Define the preceding metrics to calculate
preceding_metrics = {
    'preceding_trial_type': 'trial_type',
    'preceding_abs_max_val': 'abs_max_val',
    'preceding_abs_min_val': 'abs_min_val',
    'preceding_abs_peak_to_peak': 'abs_peak_to_peak',
    'preceding_time_yoked_max_val_offset': 'time_yoked_max_val_offset',
    'preceding_time_yoked_min_val_offset': 'time_yoked_min_val_inv',
    'preceding_time_yoked_peak_to_peak_offset': 'time_yoked_peak_to_peak_offset',
    'preceding_time_yoked_min_val_inv': 'time_yoked_min_val_inv',
    'preceding_auc_A': 'auc_A',
    'preceding_auc_B': 'auc_B',
    'preceding_auc_C': 'auc_C',
    'preceding_auc_total': 'auc_total',
    'preceding_abs_normalized_pain_change': 'abs_normalized_pain_change'
}

# Add time-yoked metrics for control trials
for trial_type in trial_metrics_df['trial_type'].unique():
    if trial_type in ['t1_hold', 't2_hold']:
        # Add time-yoked metrics based on trial type
        if trial_type == 't1_hold':
            # t1_hold can reference offset or stepdown
            for ref_type in ['offset', 'stepdown']:
                preceding_metrics[f'preceding_time_yoked_max_val_{ref_type}'] = f'time_yoked_max_val_{ref_type}'
                preceding_metrics[f'preceding_time_yoked_min_val_{ref_type}'] = f'time_yoked_min_val_{ref_type}'
                preceding_metrics[f'preceding_time_yoked_peak_to_peak_{ref_type}'] = f'time_yoked_peak_to_peak_{ref_type}'
        elif trial_type == 't2_hold':
            # t2_hold references inv
            preceding_metrics['preceding_time_yoked_max_val_inv'] = 'time_yoked_max_val_inv'
            preceding_metrics['preceding_time_yoked_min_val_inv'] = 'time_yoked_min_val_inv'
            preceding_metrics['preceding_time_yoked_peak_to_peak_inv'] = 'time_yoked_peak_to_peak_inv'

for new_col, source_col in preceding_metrics.items():
    trial_metrics_df[new_col] = trial_metrics_df.apply(
        lambda row: get_preceding_value(row, source_col), axis=1
    )


# %%
# ========================================================
# Preliminary plots of OA/OH trials by preceding trial type
# ========================================================
# Filter for OA/OH trials
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# Violin plot of abs_max_val in OA/OH trials, separated by preceding trial type
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=oa_oh_trials,
    x='preceding_trial_type',
    y='abs_max_val',
    hue='trial_type',
    split=True
)
plt.title('OA/OH trial max_val by preceding trial type')
plt.xlabel('Preceding Trial Type')
plt.ylabel('Max Value')
plt.tight_layout()
plt.show()

# Plot normalized pain change for 'offset' trials
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=oa_oh_trials[oa_oh_trials['trial_type'] == 'offset'],
    x='preceding_trial_type',
    y='abs_normalized_pain_change'
)
plt.title('Offset trials: Normalized pain change by preceding trial type')
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Plot normalized pain change for 'inv' trials
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=oa_oh_trials[oa_oh_trials['trial_type'] == 'inv'],
    x='preceding_trial_type',
    y='abs_normalized_pain_change'
)
plt.title('Inv trials: Normalized pain change by preceding trial type')
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# %%
# ========================================================================================
# Compare Preceding Trial Metrics to Current Trial Metrics 
# normalized_pain_change (as metric of OH/OA) vs auc_total, abs_max_val, abs_min_val
# ========================================================================================
# Filter for OA/OH trials
oa_oh_trials = trial_metrics_df[trial_metrics_df['trial_type'].isin(['offset', 'inv'])].copy()

# Store all correlation results for multiple comparisons correction
correlation_results = []

print("=" * 80)
print("PRECEDING TRIAL CONTEXT EFFECTS ON CONTRAST ENHANCEMENT")
print("=" * 80)

################################################################################################################### Define all the correlations to test
correlations_to_test = [
    # Format: (trial_type, x_col, y_col, description, xlabel, ylabel)
    ('offset', 'preceding_auc_total', 'abs_normalized_pain_change', 
     'OA: Preceding AUC → Current OA', 'Preceding AUC Total', 'Offset Analgesia (%)'),
    ('inv', 'preceding_auc_total', 'abs_normalized_pain_change', 
     'OH: Preceding AUC → Current OH', 'Preceding AUC Total', 'Onset Hyperalgesia (%)'),
    ('offset', 'preceding_abs_max_val', 'abs_normalized_pain_change', 
     'OA: Preceding Max → Current OA', 'Preceding Max Pain', 'Offset Analgesia (%)'),
    ('inv', 'preceding_abs_max_val', 'abs_normalized_pain_change', 
     'OH: Preceding Max → Current OH', 'Preceding Max Pain', 'Onset Hyperalgesia (%)'),
    ('offset', 'preceding_abs_peak_to_peak', 'abs_normalized_pain_change', 
     'OA: Preceding P2P → Current OA', 'Preceding Peak-to-Peak', 'Offset Analgesia (%)'),
    ('inv', 'preceding_abs_peak_to_peak', 'abs_normalized_pain_change', 
     'OH: Preceding P2P → Current OH', 'Preceding Peak-to-Peak', 'Onset Hyperalgesia (%)'),
    ('offset', 'preceding_abs_min_val', 'abs_normalized_pain_change', 
     'OA: Preceding Min → Current OA', 'Preceding Min Pain', 'Offset Analgesia (%)'),
    ('inv', 'preceding_abs_min_val', 'abs_normalized_pain_change', 
     'OH: Preceding Min → Current OH', 'Preceding Min Pain', 'Onset Hyperalgesia (%)'),
    ('offset', 'preceding_abs_normalized_pain_change', 'abs_normalized_pain_change', 
     'OA: Preceding Normalized Pain Change → Current OA', 'Preceding Normalized Pain Change (%)', 'Current Offset Analgesia (%)'),
    ('inv', 'preceding_abs_normalized_pain_change', 'abs_normalized_pain_change', 
     'OH: Preceding Normalized Pain Change → Current OH', 'Preceding Normalized Pain Change (%)', 'Current Onset Hyperalgesia (%)')
]

################################################################################################################### First pass: collect all correlation statistics
print("COLLECTING CORRELATION STATISTICS...")
print("-" * 50)
for trial_type, x_col, y_col, description, xlabel, ylabel in correlations_to_test:
    # Filter data for this specific test
    subset = oa_oh_trials[oa_oh_trials['trial_type'] == trial_type]
    clean_data = subset.dropna(subset=[x_col, y_col])
    
    if len(clean_data) >= 5:  # Need minimum data points
        # Calculate correlation
        r, p_uncorrected = stats.pearsonr(clean_data[x_col], clean_data[y_col])
        
        # Store results
        correlation_results.append({
            'description': description,
            'trial_type': trial_type,
            'x_col': x_col,
            'y_col': y_col,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'r': r,
            'p_uncorrected': p_uncorrected,
            'n': len(clean_data)
        })
        
        print(f"{description}: r={r:.3f}, p={p_uncorrected:.4f}, n={len(clean_data)}")
    else:
        print(f"INSUFFICIENT DATA for {description}: n={len(clean_data)}")

################################################################################################################### Apply multiple comparisons correction
if correlation_results:
    p_values = [result['p_uncorrected'] for result in correlation_results]
    # Benjamini-Hochberg FDR correction (recommended)
    rejected_fdr, p_corrected_fdr, alpha_sidak, alpha_bonf = multipletests(
        p_values, method='fdr_bh', alpha=0.05
    )
    # Bonferroni correction (more conservative)
    rejected_bonf, p_corrected_bonf, _, _ = multipletests(
        p_values, method='bonferroni', alpha=0.05
    )
    # Add corrected p-values to results
    for i, result in enumerate(correlation_results):
        result['p_fdr'] = p_corrected_fdr[i]
        result['p_bonferroni'] = p_corrected_bonf[i]
        result['significant_fdr'] = rejected_fdr[i]
        result['significant_bonferroni'] = rejected_bonf[i]

################################################################################################################### Print results table
print(f"\n" + "=" * 80)
print(f"MULTIPLE COMPARISONS CORRECTION RESULTS (n = {len(correlation_results)} tests)")
print("=" * 80)
print(f"{'Description':<35} {'r':<8} {'p_raw':<8} {'p_FDR':<8} {'p_Bonf':<8} {'n':<5} {'FDR_Sig':<8}")
print("-" * 80)

significant_tests = []
for result in correlation_results:
    sig_fdr = "***" if result['p_fdr'] < 0.001 else "**" if result['p_fdr'] < 0.01 else "*" if result['p_fdr'] < 0.05 else "ns"
    
    print(f"{result['description']:<35} {result['r']:<8.3f} {result['p_uncorrected']:<8.4f} "
          f"{result['p_fdr']:<8.4f} {result['p_bonferroni']:<8.4f} {result['n']:<5} {sig_fdr:<8}")
    
    if result['significant_fdr']:
        significant_tests.append(result)

################################################################################################################## Create plots 

if significant_tests:
    print(f"Found {len(significant_tests)} significant results after FDR correction:")
    for result in significant_tests:
        print(f"  - {result['description']}: r={result['r']:.3f}, p_FDR={result['p_fdr']:.4f}")
else:
    print("No significant results after FDR correction.")
print(f"\nGenerating all plots (significant results will show regression lines)...")

# Generate plots for all correlations
for result in correlation_results:
    print(f"\n{'-'*50}")
    print(f"PLOTTING: {result['description']}")
    print(f"{'-'*50}")
    
    # Create the plot using your modified function
    plot_result = create_correlation_scatter_corrected(
        df=oa_oh_trials,
        x_col=result['x_col'],
        y_col=result['y_col'],
        title=result['description'],
        xlabel=result['xlabel'],
        ylabel=result['ylabel'],
        filter_col='trial_type',
        filter_val=result['trial_type'],
        figsize=(8, 6),
        p_corrected=result['p_fdr'],
        p_uncorrected=result['p_uncorrected'],
        r_value=result['r']
    )

# %%
# ========================================================================
# Compare Preceding Trial Metrics for FIRST OA/OH Trial Only
# ========================================================================

########################################################################### OFFSET TRIAL 
# Find the first 'offset' trial for each subject
first_offset_trials = (
    oa_oh_trials[oa_oh_trials['trial_type'] == 'offset']
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

# Replace NaN in preceding_trial_type with 'None' for plotting
first_offset_trials['preceding_trial_type'] = first_offset_trials['preceding_trial_type'].fillna('None')

# Violin plot of normalized pain change by preceding trial type
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=first_offset_trials,
    x='preceding_trial_type',
    y='abs_normalized_pain_change'
)
plt.title("First 'offset' trial: Normalized pain change by preceding trial type")
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()


# Scatter plot: preceding_auc_total vs. normalized pain change for first offset trial
create_correlation_scatter(
    first_offset_trials,
    x_col='preceding_auc_total',
    y_col='abs_normalized_pain_change',
    title="First Offset trial: Normalized pain change vs. preceding AUC Total",
    xlabel='Preceding AUC Total',
    ylabel='Normalized Pain Change (%)'
)


# Scatter plot: preceding_max_val vs. normalized pain change for first offset trial
create_correlation_scatter(
    first_offset_trials,
    x_col='preceding_abs_max_val',
    y_col='abs_normalized_pain_change',
    title="First Offset trial: Normalized pain change vs. preceding Max Val",
    xlabel='Preceding Max Val',
    ylabel='Normalized Pain Change (%)'
)

# Scatter plot: preceding_min_val vs. normalized pain change for first offset trial
create_correlation_scatter(
    first_offset_trials,
    x_col='preceding_abs_min_val',
    y_col='abs_normalized_pain_change',
    title="First Offset trial: Normalized pain change vs. preceding Min Val",
    xlabel='Preceding Min Val',
    ylabel='Normalized Pain Change (%)'
)

########################################################################### ONSET/INV TRIAL 

# Find the first 'inv' trial for each subject
first_inv_trials = (
    oa_oh_trials[oa_oh_trials['trial_type'] == 'inv']
    .sort_values(['subject', 'trial_num'])
    .groupby('subject')
    .first()
    .reset_index()
)

# Replace NaN in preceding_trial_type with 'None' for plotting
first_inv_trials['preceding_trial_type'] = first_inv_trials['preceding_trial_type'].fillna('None')

# Violin plot of normalized pain change by preceding trial type
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=first_inv_trials,
    x='preceding_trial_type',
    y='abs_normalized_pain_change'
)
plt.title("First 'inv' trial: Normalized pain change by preceding trial type")
plt.xlabel('Preceding Trial Type')
plt.ylabel('Normalized Pain Change (%)')
plt.tight_layout()
plt.show()

# Scatter plot: preceding_auc_total vs. normalized pain change for first inv trial
create_correlation_scatter(
    first_inv_trials,
    x_col='preceding_auc_total',
    y_col='abs_normalized_pain_change',
    title="First Inv trial: Normalized pain change vs. preceding AUC Total",
    xlabel='Preceding AUC Total',
    ylabel='Normalized Pain Change (%)'
)

# Scatter plot: preceding_max_val vs. normalized pain change for first inv trial
create_correlation_scatter(
    first_inv_trials,
    x_col='preceding_abs_max_val',
    y_col='abs_normalized_pain_change',
    title="First Inv trial: Normalized pain change vs. preceding Max Val",
    xlabel='Preceding Max Val',
    ylabel='Normalized Pain Change (%)'
)

# Scatter plot: preceding_min_val vs. normalized pain change for first inv trial
create_correlation_scatter(
    first_inv_trials,
    x_col='preceding_abs_min_val',
    y_col='abs_normalized_pain_change',
    title="First Inv trial: Normalized pain change vs. preceding Min Val",
    xlabel='Preceding Min Val',
    ylabel='Normalized Pain Change (%)'
)


# %%
# ========================================================
# Comprehensive grid plot of normalized pain change vs. preceding min and max values for OA and OH trials
# ========================================================

# PLOT NORMALIZED PAIN CHANGE V MIN / MAX AS A GRID FOR OFFSET AND INV TRIALS
fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex='col', sharey='row')

# Define trial types and metrics
trial_types = ['offset', 'inv']
metrics = ['preceding_abs_min_val', 'preceding_abs_max_val']
metric_labels = ['Preceding Min Value', 'Preceding Max Value']

# Color scheme
colors = {'offset': 'blue', 'inv': 'red'}

for row_idx, ttype in enumerate(trial_types):
    subset = oa_oh_trials[oa_oh_trials['trial_type'] == ttype]
    
    for col_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row_idx, col_idx]
        
        # Get data
        mask = subset[metric].notnull() & subset['abs_normalized_pain_change'].notnull()
        x = subset.loc[mask, metric]
        y = subset.loc[mask, 'abs_normalized_pain_change']
        
        if len(x) > 1:
            # Compute regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=50, color=colors[ttype], edgecolors='black', linewidth=0.5)
            
            # Add regression line if significant
            if p_value < 0.05:
                x_vals = np.array([x.min(), x.max()])
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color='black', linewidth=2, linestyle='--')
            
            # Add stats text box
            ax.text(0.05, 0.95,
                   f"$R$ = {r_value:.2f}\np = {p_value:.3g}\nn = {len(x)}",
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor='black'))
        
        # Set labels
        if row_idx == 1:  # Bottom row
            ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
        if col_idx == 0:  # Left column
            ax.set_ylabel('Normalized Pain Change (%)', fontsize=12, fontweight='bold')
        
        # Add title
        title = f"{ttype.capitalize()}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Set different y-axis limits for top and bottom rows
        if row_idx == 0:  # Top row (offset)
            ax.set_ylim(-102, 0) # make -102 to give some space below -100
        else:  # Bottom row (inv)
            ax.set_ylim(0, 102) # make 102 to give some space above 100

# Add overall title
fig.suptitle('Normalized Pain Change vs Preceding Trial Metrics', 
             fontsize=16, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])
# # Save figure
# plt.savefig('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/figures/pain_change_vs_preceding_metrics_grid.svg', 
#             dpi=300, bbox_inches='tight')
plt.show()

# %%
"""====================================================================================================
SUMMARY: PRECEDING TRIAL CONTEXT EFFECTS ON CONTRAST ENHANCEMENT
====================================================================================================
Test                                     r        p_raw    p_FDR    p_Bonf   n     FDR   Bonf  Effect   Direction
----------------------------------------------------------------------------------------------------
OA: Preceding AUC → Current OA           0.208    0.0027   0.0054   0.0271   206   **    *     Small    Positive
OH: Preceding AUC → Current OH           -0.366   0.0000   0.0001   0.0002   130   ***   ***   Medium   Negative
OA: Preceding Max → Current OA           0.008    0.9090   0.9090   1.0000   206   ns    ns    Negligible Positive
OH: Preceding Max → Current OH           -0.231   0.0083   0.0138   0.0827   130   *     ns    Small    Negative
OA: Preceding P2P → Current OA           -0.289   0.0000   0.0001   0.0002   206   ***   ***   Small    Negative
OH: Preceding P2P → Current OH           0.050    0.5712   0.7140   1.0000   130   ns    ns    Negligible Positive
OA: Preceding Min → Current OA           0.345    0.0000   0.0000   0.0000   206   ***   ***   Medium   Positive
OH: Preceding Min → Current OH           -0.264   0.0024   0.0054   0.0238   130   **    *     Small    Negative
OA: Preceding Normalized Pain Change → Current OA 0.031    0.6613   0.7348   1.0000   206   ns    ns    Negligible Positive
OH: Preceding Normalized Pain Change → Current OH -0.201   0.0221   0.0316   0.2214   130   *     ns    Small    Negative

====================================================================================================
SIGNIFICANCE SUMMARY
====================================================================================================
Total tests performed: 10
Significant after FDR correction (α = 0.05): 7/10 (70.0%)
Significant after Bonferroni correction (α = 0.05): 5/10 (50.0%)

====================================================================================================
PATTERN ANALYSIS
====================================================================================================

OFFSET ANALGESIA (OA) - 3 significant effects:
  ↑ OA: Preceding AUC → Current OA: r = 0.208, p_FDR = 0.0054
  ↓ OA: Preceding P2P → Current OA: r = -0.289, p_FDR = 0.0001
  ↑ OA: Preceding Min → Current OA: r = 0.345, p_FDR = 0.0000

ONSET HYPERALGESIA (OH) - 4 significant effects:
  ↓ OH: Preceding AUC → Current OH: r = -0.366, p_FDR = 0.0001
  ↓ OH: Preceding Max → Current OH: r = -0.231, p_FDR = 0.0138
  ↓ OH: Preceding Min → Current OH: r = -0.264, p_FDR = 0.0054
  ↓ OH: Preceding Normalized Pain Change → Current OH: r = -0.201, p_FDR = 0.0316

====================================================================================================
BIOLOGICAL INTERPRETATION
====================================================================================================

KEY FINDINGS:

1. OFFSET ANALGESIA ENHANCEMENT:
   • Previous pain experiences STRENGTHEN subsequent offset analgesia:
     - OA: Preceding AUC: r = 0.208
     - OA: Preceding Min: r = 0.345
   • Previous pain experiences WEAKEN subsequent offset analgesia:
     - OA: Preceding P2P: r = -0.289

2. ONSET HYPERALGESIA HABITUATION:
   • Previous pain experiences WEAKEN subsequent onset hyperalgesia:
     - OH: Preceding AUC: r = -0.366
     - OH: Preceding Max: r = -0.231
     - OH: Preceding Min: r = -0.264
     - OH: Preceding Normalized Pain Change: r = -0.201

3. MECHANISTIC IMPLICATIONS:
   • OPPOSITE ADAPTATION PATTERNS: OA strengthens while OH weakens
   • Suggests different neural circuits with different plasticity rules
   • OA system: Adaptive enhancement (gets better with experience)
   • OH system: Protective habituation (prevents runaway sensitization)

4. CLINICAL RELEVANCE:
   • Trial-to-trial context effects reveal dynamic pain processing
   • Individual differences in these patterns may predict pain outcomes
   • Contrast enhancement mechanisms are not independent between trials

====================================================================================================
STATISTICAL NOTES
====================================================================================================
• FDR correction controls false discovery rate at 5% among significant results
• Bonferroni correction controls family-wise error rate at 5%
• Effect sizes: Small (0.1-0.3), Medium (0.3-0.5), Large (≥0.5)
• All correlations calculated using Pearson's r with complete case analysis
"""
