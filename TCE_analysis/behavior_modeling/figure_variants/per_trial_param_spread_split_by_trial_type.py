"""Same SD figure, split hold vs stepped, written to a separate file so the
approved pooled version is not overwritten until she decides."""
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

BM = ('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/'
      'temporal_contrast_enhancement/TCE_analysis/behavior_modeling/')
pt = pd.read_csv(sorted(glob.glob(
    BM + 'model_fit_results/*_per_trial_fits_full_fixedtheta.csv'))[-1])
pt['abar'] = pt.alpha / pt.beta
pt['gbar'] = pt.gamma * pt.lam / pt.beta

KIND = {'offset': 'stepped', 'onset': 'stepped',
        't1_hold': 'hold', 't2_hold': 'hold'}
PLOT_QTY = ['alpha', 'beta', 'gamma', 'lam', 'abar', 'gbar']
SYMBOL = {'alpha': r'$\alpha$', 'beta': r'$\beta$', 'gamma': r'$\gamma$',
          'lam': r'$\lambda$', 'abar': r'$\alpha\,/\,\beta$',
          'gbar': r'$\gamma\lambda\,/\,\beta$'}
KIND_COLORS = {'hold': '#9AA7B0', 'stepped': '#2B6CA3'}

tt = pt[pt.trial_type.isin(KIND)].copy()
tt['kind'] = tt.trial_type.map(KIND)
cell = (tt.groupby(['subject_uid', 'trial_type', 'kind'])
        .agg(n_trials=('alpha', 'size'),
             **{q: (q, lambda v: v.std(ddof=1)) for q in PLOT_QTY})
        .reset_index())
cell = cell[cell.n_trials >= 3]

print(f'{len(cell)} cells, {cell.subject_uid.nunique()} subjects '
      f'({(cell.kind == "hold").sum()} hold, '
      f'{(cell.kind == "stepped").sum()} stepped)\n')
print(f'{"":16s} {"hold":>10s} {"stepped":>10s} {"ratio":>8s} {"p (paired)":>12s}')
for q in PLOT_QTY:
    w = cell.pivot_table(index='subject_uid', columns='kind', values=q)
    pair = w[['hold', 'stepped']].dropna()
    p = st.wilcoxon(pair.stepped, pair.hold)[1]
    h = cell.loc[cell.kind == 'hold', q].median()
    s = cell.loc[cell.kind == 'stepped', q].median()
    plain = {'abar': 'alpha/beta', 'gbar': 'gamma*lam/beta'}.get(q, q)
    print(f'{plain:16s} {h:10.4g} {s:10.4g} {s / h:8.2f} {p:12.3g}  n={len(pair)}')

fig, axes = plt.subplots(1, len(PLOT_QTY), figsize=(2.9 * len(PLOT_QTY), 4.8))
for ax, q in zip(np.atleast_1d(axes), PLOT_QTY):
    v = cell[q].dropna()
    q1, q3 = v.quantile(.25), v.quantile(.75)
    hi = min(v.max(), q3 + 1.5 * (q3 - q1))
    sub = cell[cell[q] <= hi]
    sns.violinplot(data=sub, x='kind', y=q, order=['hold', 'stepped'],
                   hue='kind', hue_order=['hold', 'stepped'],
                   palette=KIND_COLORS, legend=False, cut=0, inner='box',
                   linewidth=1.0, saturation=1.0, width=.72, ax=ax)
    for coll in ax.collections:
        coll.set_alpha(.85)
    lab = []
    for k in ('hold', 'stepped'):
        lab.append(f'{k} {cell.loc[cell.kind == k, q].median():.3g}')
    n_over = int((v > hi).sum())
    ax.text(.5, .98, '   '.join(lab) + (f'\n{n_over} above axis' if n_over else ''),
            transform=ax.transAxes, ha='center', va='top', fontsize=8.5,
            color='0.15')
    ax.set_title(SYMBOL[q], fontsize=15)
    ax.set(xlabel='', ylabel='')
    ax.set_ylim(bottom=0, top=hi * 1.25)
    ax.tick_params(labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel(
    "SD across a subject's own trials  (parameter's own units)")
fig.suptitle("Trial-to-trial spread of each subject's own parameters, "
             "split by trial type.   One point = one subject within one type.",
             y=1.0)
plt.tight_layout()
out = BM + 'figures/20261101_per_trial_param_spread_SPLIT.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('\nwrote', out)
