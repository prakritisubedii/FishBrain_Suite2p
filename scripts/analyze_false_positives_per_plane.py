#!/usr/bin/env python3
"""
analyze_false_positives_per_plane.py — Per-plane breakdown of flagged vs normal cells

Runs on CPU only — safe to run while Optuna jobs use both GPUs.

Reads the false_positive_analysis.csv files already saved by
analyze_false_positives.py and breaks down npix, npix_norm, radius, and skew
by PLANE, separately for flagged vs normal cells.

Purpose:
    The pooled summary showed flagged cells have smaller npix/radius on
    average — but flagged cells are concentrated in specific planes
    (1/2/3/6 of 165925). This script checks whether the size difference
    holds WITHIN each plane, or whether it's just a plane-level confound
    (i.e. those planes have smaller cells regardless of flagging).

Usage:
    python analyze_false_positives_per_plane.py
"""

import os
import csv
import numpy as np
from collections import defaultdict

RESULTS_ROOT = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results"
SESSIONS = {
    "145544": "145544_run1_colmedian_tau07",
    "164302": "164302_run1_colmedian_tau07",
    "164628": "164628_run1_colmedian_tau07",
    "165925": "165925_run1_colmedian_tau07",
}
PP_TAG = "rolling_prctile_win40_pct8"

METRICS = ["npix", "npix_norm", "radius", "skew"]

# data[(session, plane)] = list of row dicts
data = defaultdict(list)

for session_id, run_name in SESSIONS.items():
    csv_path = os.path.join(
        RESULTS_ROOT, run_name, "postprocessing", PP_TAG,
        "false_positive_analysis.csv"
    )
    if not os.path.exists(csv_path):
        print(f"Session {session_id}: CSV not found at {csv_path}, skipping.")
        continue

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plane = int(row["plane"])
            data[(session_id, plane)].append(row)

print("=" * 90)
print("PER-PLANE BREAKDOWN: flagged vs normal cells")
print("=" * 90)

# Track aggregate within-plane differences across all (session, plane) groups
# to see if the pattern is consistent
diff_summary = defaultdict(list)  # metric -> list of (flagged_median - normal_median)

for (session_id, plane), rows in sorted(data.items()):
    flagged = [r for r in rows if r["flagged_increasing"] == "1"]
    normal  = [r for r in rows if r["flagged_increasing"] == "0"]

    n_flagged = len(flagged)
    n_normal  = len(normal)

    if n_flagged == 0:
        continue  # nothing to compare in this plane

    print(f"\nSession {session_id}, Plane {plane}: "
          f"{n_flagged} flagged / {n_normal} normal")

    for metric in METRICS:
        try:
            flagged_vals = [float(r[metric]) for r in flagged
                             if r[metric] not in ("", "nan")]
            normal_vals  = [float(r[metric]) for r in normal
                             if r[metric] not in ("", "nan")]
        except (ValueError, KeyError):
            continue

        if not flagged_vals or not normal_vals:
            continue

        f_med = np.median(flagged_vals)
        n_med = np.median(normal_vals)
        diff  = f_med - n_med

        print(f"  {metric:12s}: flagged median={f_med:8.3f}  "
              f"normal median={n_med:8.3f}  diff={diff:+8.3f}")

        diff_summary[metric].append(diff)

# ─────────────────────────────────────────────────────────────────────────────
# Overall consistency check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("CONSISTENCY CHECK: is the flagged-vs-normal difference consistent")
print("across planes, or does it flip sign (suggesting a plane-level confound)?")
print("=" * 90)

for metric, diffs in diff_summary.items():
    diffs = np.array(diffs)
    n_negative = (diffs < 0).sum()
    n_positive = (diffs > 0).sum()
    n_total = len(diffs)

    print(f"\n{metric}:")
    print(f"  Groups where flagged < normal: {n_negative}/{n_total}")
    print(f"  Groups where flagged > normal: {n_positive}/{n_total}")
    print(f"  Mean diff across groups: {diffs.mean():+.4f}")

    if n_total >= 3:
        if n_negative == n_total or n_positive == n_total:
            print(f"  -> CONSISTENT direction across all groups. "
                  f"Likely a real effect, not a plane-level confound.")
        elif max(n_negative, n_positive) / n_total >= 0.75:
            print(f"  -> MOSTLY consistent ({max(n_negative,n_positive)}/{n_total}), "
                  f"but some groups flip sign. Weak/mixed effect.")
        else:
            print(f"  -> INCONSISTENT (flips sign across groups). "
                  f"Likely a plane-level confound, not a real flagged-vs-normal effect.")
    else:
        print(f"  -> Not enough groups ({n_total}) with flagged cells to assess consistency.")

print("\nDone.")