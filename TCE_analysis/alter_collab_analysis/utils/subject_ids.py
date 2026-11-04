# -*- coding: utf-8 -*-
"""
Subject identity across the three datasets.

Lives here rather than inside one pipeline script because both
data_pipeline/build_combined_traces.py and data_pipeline/extract_metrics.py
need the same cLBP mapping. Two copies of an ID rule drift apart silently, and
when they do, two files that look like they describe the same person do not.
"""


def canonical_clbp_id(subject, study):
    """
    Map a within-substudy cLBP subject number onto the lab's DPOP_ID namespace.

    MYP_DPOP_MBPR_DATA_09062023.csv uses DPOP_ID as the master ID for both
    substudies. DPOP subjects keep their own number (1-57); MBPR subjects are
    re-coded by prefixing a '2' onto a zero-padded MBPR ID:

        MBPR 1 -> 201      MBPR 53  -> 253
        MBPR 10 -> 210     MBPR 101 -> 2101
        MBPR 22 -> 222     MBPR 145 -> 2145

    This holds for every MBPR row in that file, and it is what makes DPOP
    subject 1 and MBPR subject 1 distinguishable -- they are DPOP_ID 1 and 201.
    Seven subject numbers are used by both substudies (1, 10, 22, 40, 43, 48,
    53), so without this they are 7 pairs of different people sharing a key.
    """
    if study == 'cLBP_DPOP':
        return int(subject)
    if study == 'cLBP_MBPR':
        return int('2' + str(int(subject)).zfill(2))
    raise ValueError(f'Unexpected cLBP study: {study!r}')
