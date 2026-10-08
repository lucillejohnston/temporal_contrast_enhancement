"""Rebuild the data embedded in cecchi_model_explorer.html.

Writes two JavaScript lines into the page, replacing the old ones in place:
  var PROFILES = {dataset: {group: {trial_type: {t, T, P, n_trials, n_subj}}}}
  var SUBJECTS = {dataset: [{id, g, fit, r_full, r_simple, prof: {trial_type: {t, T, P, n}}}]}

Forearm trials only. A time point is kept when at least half the trials being
averaged have a sample there, so ragged trial ends do not drag the mean around.

Saved fits shown per subject:
  - full model (Eq. 1): 20260924_model_fits_subject_full.csv. kneeOA fits there
    pooled knee and forearm trials; the page says so.
  - simplified (Eq. 2) r: model_fits_subject_20261005.csv, forearm rows only.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRACES = HERE.parents[1] / "data" / "alter_collab_data" / "combined_traces_1Hz.pkl"
FULL_FITS = HERE / "model_fit_results" / "20260924_model_fits_subject_full.csv"
SIMPLE_FITS = HERE / "model_fit_results" / "model_fits_subject_20261005.csv"
HTML = HERE / "cecchi_model_explorer.html"

TRIAL_TYPES = ["offset", "onset", "t1_hold", "t2_hold"]
GROUP_ORDER = ["Control", "Low", "High"]


def mean_profile(df):
    """Average temperature and pain at each whole second across trials."""
    df = df.assign(sec=df["aligned_time"].round().astype(int))
    n_trials = df[["subject_uid", "trial_num"]].drop_duplicates().shape[0]
    g = df.groupby("sec").agg(
        T=("temperature", "mean"),
        P=("pain", "mean"),
        n=("trial_num", "size"),
    )
    g = g[g["n"] >= n_trials / 2]
    return {
        "t": g.index.tolist(),
        "T": np.round(g["T"].to_numpy(), 2).tolist(),
        "P": np.round(g["P"].to_numpy(), 1).tolist(),
        "n_trials": int(n_trials),
        "n_subj": int(df["subject_uid"].nunique()),
    }


def main():
    tr = pd.read_pickle(TRACES)
    tr = tr[(tr["site"] == "forearm") & tr["trial_type"].isin(TRIAL_TYPES)]
    tr = tr.dropna(subset=["temperature", "pain"])

    profiles = {}
    for ds, dsd in tr.groupby("dataset"):
        groups = ["All"] + [g for g in GROUP_ORDER if g in set(dsd["group_label"].dropna())]
        profiles[ds] = {}
        for grp in groups:
            gd = dsd if grp == "All" else dsd[dsd["group_label"] == grp]
            profiles[ds][grp] = {tt: mean_profile(gd[gd["trial_type"] == tt]) for tt in TRIAL_TYPES}

    full = pd.read_csv(FULL_FITS).set_index("subject_uid")
    simple = pd.read_csv(SIMPLE_FITS)
    simple = simple[simple["site"].fillna("forearm") == "forearm"].set_index("subject_uid")

    subjects = {}
    for ds, dsd in tr.groupby("dataset"):
        rows = []
        for uid, sd in dsd.groupby("subject_uid"):
            prof = {}
            for tt in TRIAL_TYPES:
                td = sd[sd["trial_type"] == tt]
                if td.empty:
                    continue
                p = mean_profile(td)
                prof[tt] = {"t": p["t"], "T": p["T"], "P": p["P"], "n": p["n_trials"]}
            row = {"id": uid, "g": sd["group_label"].iloc[0], "prof": prof,
                   "fit": None, "r_full": None, "r_simple": None}
            if uid in full.index:
                f = full.loc[uid]
                row["fit"] = {k: round(float(f[k]), 4) for k in ["alpha", "beta", "gamma", "lam", "theta"]}
                row["r_full"] = round(float(f["r"]), 3)
            if uid in simple.index:
                row["r_simple"] = round(float(simple.loc[uid, "r"]), 3)
            rows.append(row)
        subjects[ds] = rows

    html = HTML.read_text()
    for name, obj in [("PROFILES", profiles), ("SUBJECTS", subjects)]:
        line = "  var %s = %s;" % (name, json.dumps(obj, separators=(",", ":")))
        html, n = re.subn(r"^  var %s = .*;$" % name, lambda m: line, html, flags=re.M)
        if n != 1:
            raise SystemExit("expected exactly one 'var %s = ...;' line in the HTML, found %d" % (name, n))
    HTML.write_text(html)

    n_subj = sum(len(v) for v in subjects.values())
    print("wrote %s: %d subjects, %.0f KB" % (HTML.name, n_subj, len(html) / 1024))


if __name__ == "__main__":
    main()
