# -*- coding: utf-8 -*-
"""
Period A/B/C windows measured from each trial's own temperature trace.

This replaces the experimenter-defined windows in extract_metrics.py
(define_timewindows), which counts forward from one anchor on an assumed
5 s / 0.67 s / 5 s / 0.67 s / 20 s schedule. In practice the thermode does not
follow that schedule exactly, so those windows drift away from the real steps.

Ported from the neural pipeline (utils/data_loading.py: find_plateau_level,
measure_temperature_edges, get_epoch_timing). The idea there is the one kept
here: never ask "is the temperature changing?" -- that means differentiating,
and thermode jitter of a few hundredths of a degree between samples looks like
a large rate of change. Ask instead "what level is the thermode sitting at?",
which is immune to jitter because it compares against a value, not a slope.

Three things differ from the neural version, all because the behavioural rig is
not the neural rig:

  baseline     the neural probe rests at 29 C, so `temp > 30` skips the
               baseline. Here it rests at 32 C and holds there for ~8 s, which
               that test would return as "the first plateau". Baseline is taken
               from the trace instead.
  ramp down    the neural version finds the end of C with `temp < 40`. kneeOA
               `innocuous` trials plateau at ~42 C and cLBP traces are trimmed
               before the ramp down ever happens. Both are handled relative to
               the plateau the trial actually held.
  trial types  the behavioural datasets add `innocuous` (one plateau, like a
               hold) and plosONE `stepdown` (two plateaus, 46 -> 45, with no
               return), neither of which exists in the neural task.

Tolerance is 0.3 C rather than 0.15 C because these traces are 1 Hz averages of
the 10 Hz log, so a sample spanning a step lands between the two levels and the
thermode overshoots by up to ~0.26 C on settling. The two plateaus are 1.0 C
apart, so +/-0.3 C bands are still disjoint with 0.4 C to spare.

Author: Lucille Johnston
Updated: 10/27/26
"""

import numpy as np
import pandas as pd

# Tolerances. AT_LEVEL is "close enough to be sitting at this level"; FLAT_TOL
# is the half-width allowed within a candidate plateau when identifying its
# temperature. The ramp runs at ~1.45 C/s, so over MIN_PLATEAU_SEC it covers
# ~4.4 C and can never be mistaken for a plateau.
AT_LEVEL = 0.3
FLAT_TOL = 0.3
MIN_PLATEAU_SEC = 3.0
HOLD_PERIOD_SEC = 5.0

# How each trial type moves after its first plateau.
#   STEP_RETURN  three plateaus, P -> P+step -> P   (A, B, C)
#   STEP_ONCE    two plateaus,   P -> P+step        (A and B split P, C is the second)
#   HOLD_TYPES   one plateau                        (A and B are 5 s each, C is the rest)
# 'inv' is plosONE's name for what kneeOA and cLBP call 'onset'.
STEP_RETURN = {'offset': +1.0, 'onset': -1.0, 'inv': -1.0}
STEP_ONCE = {'stepdown': -1.0}
HOLD_TYPES = {'t1_hold', 't2_hold', 'innocuous'}

PERIOD_COLS = ['A_start', 'A_end', 'B_start', 'B_end', 'C_start', 'C_end']


class TrialWindowError(ValueError):
    """A trial's temperature trace does not show the plateaus its type implies."""


def find_plateau_level(t, temp, flat_tol=FLAT_TOL, min_plateau_sec=MIN_PLATEAU_SEC,
                       baseline_margin=1.0):
    """
    Temperature (C) of the first plateau in one trial.

    Walks forward from the first sample clear of baseline and returns the median
    of the first window of min_plateau_sec whose 5th-95th percentile spread is
    within +/-flat_tol. Percentiles rather than min/max so a single noisy sample
    does not reject a real plateau.

        offset / t1_hold        first plateau = T1
        onset / t2_hold         first plateau = T2
        innocuous / stepdown    first plateau = the only / upper level

    Gives the LEVEL only; the timing comes from measure_temperature_edges.
    Returns nan if no window qualifies.
    """
    t = np.asarray(t, dtype=float)
    temp = np.asarray(temp, dtype=float)
    baseline = float(np.nanmin(temp))

    for i in np.flatnonzero(temp > baseline + baseline_margin):
        if t[i] + min_plateau_sec > t[-1]:
            break  # not enough trace left for a plateau this long
        seg = temp[(t >= t[i]) & (t <= t[i] + min_plateau_sec)]
        if len(seg) < 3:
            continue
        lo, hi = np.percentile(seg, [5, 95])
        if hi - lo <= 2 * flat_tol:
            return round(float(np.median(seg)), 1)
    return np.nan


def _measure_c_end(t, temp, i_c_start, settle_sec=1.0, window_sec=5.0,
                   drop=1.0, tol=0.5):
    """
    End of period C: the last sample still at the level C actually held.

    `final_level` is measured from C itself rather than assumed, because the
    thermode settles a little low or high. The search for the ramp down is
    relative to that level, not an absolute 40 C, so it works for the ~42 C
    innocuous trials as well as the ~48 C noxious ones.

    cLBP traces are trimmed before the ramp down, so when no descent is found
    C ends at the last sample rather than raising.
    """
    tail = (t > t[i_c_start] + settle_sec) & (t < t[i_c_start] + settle_sec + window_sec)
    final_level = float(np.median(temp[tail])) if tail.any() else float(temp[i_c_start])

    descending = np.flatnonzero((t > t[i_c_start]) & (temp < final_level - drop))
    if descending.size == 0:
        return t[-1], final_level
    i_down = descending[0]
    held = np.flatnonzero(temp[:i_down] >= final_level - tol)
    return (t[held[-1]] if held.size else t[i_down - 1]), final_level


def measure_temperature_edges(t, temp, trial_type, at_level=AT_LEVEL,
                              hold_period_sec=HOLD_PERIOD_SEC,
                              flat_tol=FLAT_TOL, min_plateau_sec=MIN_PLATEAU_SEC):
    """
    Period A/B/C windows of one trial, measured from its temperature trace.

        A = first plateau, B = second plateau, C = last plateau.
        Ramps between plateaus fall in the gaps and belong to no window.

    'At a level' means within +/-at_level C of it.

    offset / onset (three plateaus, P -> P+/-1 -> P):
        A_start  first sample at the first plateau
        A_end    last sample at it before the temperature moves to the second
        B_start  first sample at the second plateau
        B_end    last sample at it before the temperature moves back
        C_start  first sample back at the first plateau
        C_end    last sample before the ramp down (or the last sample, if the
                 trace is trimmed before the ramp down)

    stepdown (two plateaus, P -> P-1, no return):
        A and B split the first plateau (A is hold_period_sec, B runs to the
        step), and C is the second plateau.

    t1_hold / t2_hold / innocuous (one plateau):
        A and B are hold_period_sec each from A_start, and C runs from B_end to
        the measured ramp down. There are no steps to mark A and B, so those two
        boundaries remain experimenter-defined.

    Returns dict: P1 (first plateau, C), P2 (second plateau, C, or nan),
    C_level (the level C actually held) and the six window columns.
    Raises TrialWindowError if the trace does not show the expected plateaus.
    """
    t = np.asarray(t, dtype=float)
    temp = np.asarray(temp, dtype=float)
    # 18 plosONE trials carry NaN temperature samples in otherwise sound traces.
    # Dropping them from both arrays together keeps the real sample times, so
    # the measured edges stay on the trial clock.
    usable = np.isfinite(t) & np.isfinite(temp)
    t, temp = t[usable], temp[usable]
    if len(t) < 5:
        raise TrialWindowError(f'only {len(t)} usable samples')

    P = find_plateau_level(t, temp, flat_tol=flat_tol, min_plateau_sec=min_plateau_sec)
    if not np.isfinite(P):
        raise TrialWindowError('no plateau found')

    on_P = np.abs(temp - P) <= at_level
    i0 = int(np.flatnonzero(on_P)[0])  # first sample at the first plateau
    out = {'P1': P, 'P2': np.nan, 'A_start': t[i0]}

    if trial_type in HOLD_TYPES:
        out['A_end'] = out['A_start'] + hold_period_sec
        out['B_start'] = out['A_end']
        out['B_end'] = out['B_start'] + hold_period_sec
        out['C_start'] = out['B_end']
        i_last = i0
    elif trial_type in STEP_RETURN or trial_type in STEP_ONCE:
        step = STEP_RETURN.get(trial_type, STEP_ONCE.get(trial_type))
        Q = P + step
        on_Q = np.abs(temp - Q) <= at_level
        reached_Q = np.flatnonzero(on_Q & (t > t[i0]))
        if reached_Q.size == 0:
            raise TrialWindowError(f'never reached the second plateau ({Q:.1f} C)')
        iq = int(reached_Q[0])
        out['P2'] = Q

        if trial_type in STEP_RETURN:
            returned = np.flatnonzero(on_P & (t > t[iq]))
            if returned.size == 0:
                raise TrialWindowError(f'never returned to the first plateau ({P:.1f} C)')
            ip = int(returned[0])
            out['A_end'] = t[np.flatnonzero(on_P[:iq])[-1]]
            out['B_start'] = t[iq]
            out['B_end'] = t[np.flatnonzero(on_Q[:ip])[-1]]
            out['C_start'] = t[ip]
            i_last = ip
        else:  # STEP_ONCE: one long first plateau, then the step and no return
            out['A_end'] = out['A_start'] + hold_period_sec
            out['B_start'] = out['A_end']
            out['B_end'] = t[np.flatnonzero(on_P[:iq])[-1]]
            out['C_start'] = t[iq]
            i_last = iq
    else:
        raise TrialWindowError(f'unrecognised trial type {trial_type!r}')

    # Windows must be ordered. Hold trials put A and B at fixed offsets, so a
    # short trace can push B_end past the end of the trial.
    i_c = int(np.argmin(np.abs(t - out['C_start'])))
    out['C_end'], out['C_level'] = _measure_c_end(t, temp, i_c)
    edges = [out[c] for c in PERIOD_COLS]
    if any(b < a for a, b in zip(edges, edges[1:])):
        raise TrialWindowError('windows out of order (trace too short?)')
    return out


def measure_all_trials(traces, group_cols=('dataset', 'subject_uid', 'site', 'trial_num'),
                       time_col='aligned_time', temp_col='temperature',
                       trial_type_col='trial_type', **kwargs):
    """
    Measured windows for every trial in a traces table.

    traces : DataFrame with the group columns plus time, temperature, trial_type
             (i.e. combined_traces_1Hz.pkl).

    Returns one row per trial with the group columns, trial_type, P1, P2,
    C_level, the six window columns, each period's duration, and `status`
    ('ok', or why the trial could not be measured). Unmeasurable trials keep
    their row with nan windows rather than disappearing.
    """
    group_cols = [c for c in group_cols if c in traces.columns]
    rows = []
    for keys, g in traces.groupby(list(group_cols), sort=False):
        g = g.sort_values(time_col)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        trial_type = g[trial_type_col].iloc[0]
        row['trial_type'] = trial_type
        row['n_samples'] = len(g)
        try:
            row.update(measure_temperature_edges(
                g[time_col].to_numpy(dtype=float),
                g[temp_col].to_numpy(dtype=float),
                trial_type, **kwargs))
            row['status'] = 'ok'
        except TrialWindowError as e:
            row['status'] = str(e)
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in ['P1', 'P2', 'C_level'] + PERIOD_COLS:
        if col not in out:
            out[col] = np.nan
    for period in ['A', 'B', 'C']:
        out[f'{period}_dur'] = out[f'{period}_end'] - out[f'{period}_start']
    return out


def window_mask(t, windows, period):
    """Boolean mask for the samples of `t` inside `period` (inclusive of both edges)."""
    start, end = windows[f'{period}_start'], windows[f'{period}_end']
    if not (np.isfinite(start) and np.isfinite(end)):
        return np.zeros(len(t), dtype=bool)
    return (np.asarray(t, dtype=float) >= start) & (np.asarray(t, dtype=float) <= end)
