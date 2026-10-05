import numpy as np, pandas as pd, sys
sys.path.append('.')
df = pd.read_pickle('/Users/ljohnston1/Library/CloudStorage/OneDrive-UCSF/Desktop/Python/temporal_contrast_enhancement/data/alter_collab_data/combined_traces_1Hz.pkl')
MAXLAG=12

def best_lag(x, y, maxlag=MAXLAG):
    '''lag k maximising corr(y[t+k], x[t]) -- how far y trails x, in seconds'''
    out=[]
    for k in range(0, maxlag+1):
        a, b = x[:len(x)-k], y[k:]
        if len(a)<8 or np.std(a)==0 or np.std(b)==0: out.append(np.nan); continue
        out.append(np.corrcoef(a,b)[0,1])
    out=np.array(out)
    if np.all(np.isnan(out)): return np.nan, np.nan, np.nan
    k=int(np.nanargmax(out))
    return k, out[k], out[0]

rows=[]
for (uid, tnum), g in df.sort_values('aligned_time').groupby(['subject_uid','trial_num']):
    k, r_best, r_zero = best_lag(g.temperature.values, g.pain.values)
    if np.isnan(k): continue
    rows.append({'subject_uid':uid,'trial_num':tnum,'dataset':g.dataset.iloc[0],
                 'trial_type':g.trial_type.iloc[0],'lag':k,'r_best':r_best,'r_zero':r_zero})
L=pd.DataFrame(rows)
print(f'{len(L):,} trials, {L.subject_uid.nunique()} subjects')
print()
print('=== STIMULUS -> RESPONSE LAG (Petre 2017 procedure; they report 3.36 +/- 1.6 s) ===')
per_subj = L.groupby(['subject_uid','dataset']).lag.median().reset_index()
print(f'across all trials : median {L.lag.median():.1f}s, mean {L.lag.mean():.2f}s, IQR [{L.lag.quantile(.25):.0f}, {L.lag.quantile(.75):.0f}]')
print(f'per-subject median: mean {per_subj.lag.mean():.2f}s +/- {per_subj.lag.std():.2f} (mean +/- SD, comparable to Petre)')
print()
print('by dataset:')
print(per_subj.groupby('dataset').lag.agg(['mean','std','median','count']).round(2).to_string())
print()
print('by trial type (median lag, seconds):')
print(L.groupby(['dataset','trial_type']).lag.agg(['median','count']).round(1).to_string())
print()
print('=== HOW MUCH CORRELATION IS LOST BY IGNORING THE LAG? ===')
print(f'median corr(temp, pain) at zero lag : {L.r_zero.median():.3f}')
print(f'median corr(temp, pain) at best lag : {L.r_best.median():.3f}')
print(f'median gain                          : +{(L.r_best-L.r_zero).median():.3f}')
print()
print('fraction of trials whose best lag is 0s:', f'{(L.lag==0).mean():.1%}')
print('fraction with best lag >= 2s         :', f'{(L.lag>=2).mean():.1%}')
