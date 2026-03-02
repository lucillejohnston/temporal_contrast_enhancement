# temporal_contrast_enhancement

### For alter_collab_analysis:
1. Either extract_kneeOA_data.py or extract_PLOSONE_data.py (or other script) 
- pull the data from the given format
- put into the SQL database (combined_data.sqlite)
2. import_from_SQL.py
- pull from the SQL database
- create {dataset}_trial_data.json
3. preprocessing.py
- resample to 10Hz
- clean out weird trials
- align trials in time
- create {dataset}_trial_data_cleaned_aligned.json 
- resample further to 5Hz
- create {dataset}_trial_data_trimmed_dowmsampled.json
4. extract_metrics.py 
- extracts specific metrics from all trials
- creates {dataset}_trial_metrics.json
5. task_plots.py
- creates nice figures of all averaged trial data
5. basic_stats.py
- uses extracted trial metrics and does some basic stats (including replicating figures from Ben's published work)
6. across_session.py
- looks at across-session metrics 
7. plot_over_time.py
- nice plot of all trials over time for subjects to visualize overall trends
8. trial_sequences.py
- looks at the ordering of trials to see if there is any carryover effect from the previous trial
9. habituators_sensitizers.py
- building on trial_sequences.py, are there common trajectories of people across trials? 
10. baseline_metrics.py
- building on trial_sequences.py and habituators_sensizters.py, are there differences at baseline (or at the first trial) that predict the trajectory of people over time 