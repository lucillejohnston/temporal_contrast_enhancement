"""
There are 3 ways to extract OA/OH:
1. Normalized pain change (min-max)/max or (max-min)/max 
2. AUC hold period C - AUC stepped period C
3. Max pain in period A - curve of period C 

This code:
- extracts each metric (some are already calculated in trial_metrics) 
- 
"""