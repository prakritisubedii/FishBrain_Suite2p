#!/usr/bin/env python3
"""
analyze_false_positives.py — Identify likely false-positive ROIs

Runs on CPU only — safe to run while Optuna jobs use both GPUs.

Usage:
    python analyze_false_positives.py

What this does:
  For each of the 4 valid sessions (145544, 164302, 164628, 165925):
    - Loads the combined postprocessing output (dff_all_planes.npy etc.)
    - For each detected cell, computes the neuropil-corrected trace Fc
    - Flags cells where Fc INCREASES over the recording (last 15% mean >
      first 15% mean by more than 10%) — this is the suspicious pattern
      we saw in 165925 Planes 1/2/3/6 (likely blood vessels / neuropil,
      not real neurons, since real GCaMP signal should bleach DOWNWARD)
    - Cross-references each flagged cell against suite2p's own quality
      metrics from stat.npy (skew, compactness, npix, etc. — whatever
      fields are actually present)
    - Saves a per-session CSV of all cells with their trend + quality
      metrics + flag, and prints a summary comparing flagged vs normal cells

Output:
    results/<run>/postprocessing/<pp>/false_positive_analysis.csv
    (one row per cell: session, plane, roi_idx, trend, flag, + stat fields)

Interpretation:
    If flagged cells consistently have LOWER skew and/or LOWER compactness
    than normal cells, that's evidence the "increasing trace" heuristic is
    catching the same population that suite2p's own metrics consider
    lower-quality — useful supporting evidence for filtering criteria.
"""

import os
import csv
import numpy as np

RESULTS_ROOT = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results"
SESSIONS = {
    "144321": None,  # excluded — corrupted
    "145544": "145544_run1_colmedian_tau07",
    "164302": "164302_run1_colmedian_tau07",
    "164628": "164628_run1_colmedian_tau07",
    "165925": "165925_run1_colmedian_tau07",
}
PP_TAG = "rolling_prctile_win40_pct8"

NEUROPIL_COEFF = 0.7
TREND_FRAC     = 0.15   # use first/last 15% of trace for trend
TREND_THRESH   = 0.10   # flag if (last_mean - first_mean) / |first_mean| > this
N_PLANES       = 7

printed_fields = False
all_rows = []

for session_id, run_name in SESSIONS.items():
    if run_name is None:
        print(f"Session {session_id}: excluded, skipping.")
        continue

    run_dir = os.path.join(RESULTS_ROOT, run_name)
    pp_dir  = os.path.join(run_dir, "postprocessing", PP_TAG)
    s2p_dir = os.path.join(run_dir, "suite2p")

    cell_plane_path   = os.path.join(pp_dir, "cell_plane.npy")
    cell_roi_idx_path = os.path.join(pp_dir, "cell_roi_idx.npy")

    if not os.path.exists(cell_plane_path):
        print(f"Session {session_id}: postprocessing output not found at "
              f"{pp_dir}, skipping.")
        continue

    cell_plane   = np.load(cell_plane_path)
    cell_roi_idx = np.load(cell_roi_idx_path)

    print(f"\nSession {session_id}: {len(cell_plane)} cells total")

    session_rows = []

    for p in range(N_PLANES):
        mask = cell_plane == p
        n_in_plane = mask.sum()
        if n_in_plane == 0:
            continue

        F_path    = os.path.join(s2p_dir, f"plane{p}", "F.npy")
        Fneu_path = os.path.join(s2p_dir, f"plane{p}", "Fneu.npy")
        stat_path = os.path.join(s2p_dir, f"plane{p}", "stat.npy")

        if not (os.path.exists(F_path) and os.path.exists(stat_path)):
            print(f"  Plane {p}: missing F.npy or stat.npy, skipping")
            continue

        F    = np.load(F_path)
        Fneu = np.load(Fneu_path)
        stat = np.load(stat_path, allow_pickle=True)

        if not printed_fields:
            print(f"\nAvailable stat.npy fields (from session {session_id}, "
                  f"plane {p}, cell 0):")
            print(f"  {sorted(stat[0].keys())}\n")
            printed_fields = True

        roi_idxs = cell_roi_idx[mask]

        for roi_idx in roi_idxs:
            roi_idx = int(roi_idx)
            Fc = F[roi_idx].astype(np.float64) - NEUROPIL_COEFF * Fneu[roi_idx].astype(np.float64)
            n = len(Fc)
            seg = max(int(n * TREND_FRAC), 5)

            first_mean = Fc[:seg].mean()
            last_mean  = Fc[-seg:].mean()

            if abs(first_mean) > 1e-6:
                trend = (last_mean - first_mean) / abs(first_mean)
            else:
                trend = 0.0

            flagged = trend > TREND_THRESH

            s = stat[roi_idx]
            row = {
                "session": session_id,
                "plane": p,
                "roi_idx": roi_idx,
                "trend": round(float(trend), 4),
                "flagged_increasing": int(flagged),
                "skew": float(s.get("skew", np.nan)) if "skew" in s else np.nan,
                "compact": float(s.get("compact", np.nan)) if "compact" in s else np.nan,
                "npix": float(s.get("npix", np.nan)) if "npix" in s else np.nan,
                "npix_norm": float(s.get("npix_norm", np.nan)) if "npix_norm" in s else np.nan,
                "radius": float(s.get("radius", np.nan)) if "radius" in s else np.nan,
                "solidity": float(s.get("solidity", np.nan)) if "solidity" in s else np.nan,
                "aspect_ratio": float(s.get("aspect_ratio", np.nan)) if "aspect_ratio" in s else np.nan,
            }
            session_rows.append(row)
            all_rows.append(row)

    n_flagged = sum(r["flagged_increasing"] for r in session_rows)
    print(f"  Flagged as 'increasing' (likely false positive): "
          f"{n_flagged} / {len(session_rows)} "
          f"({100*n_flagged/len(session_rows):.1f}%)")

    # ── Save per-session CSV ──────────────────────────────────────────────
    if session_rows:
        fieldnames = list(session_rows[0].keys())
        csv_path = os.path.join(pp_dir, "false_positive_analysis.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(session_rows)
        print(f"  Saved: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Overall summary across all sessions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OVERALL SUMMARY (all sessions combined)")
print("=" * 60)

if all_rows:
    total = len(all_rows)
    n_flagged = sum(r["flagged_increasing"] for r in all_rows)
    print(f"Total cells          : {total}")
    print(f"Flagged (increasing) : {n_flagged} ({100*n_flagged/total:.1f}%)")

    # Compare quality metrics between flagged and non-flagged, for any
    # field that has non-NaN values
    metric_fields = ["skew", "compact", "npix", "npix_norm", "radius",
                     "solidity", "aspect_ratio"]

    for field in metric_fields:
        flagged_vals = [r[field] for r in all_rows
                        if r["flagged_increasing"] and not np.isnan(r[field])]
        normal_vals  = [r[field] for r in all_rows
                        if not r["flagged_increasing"] and not np.isnan(r[field])]

        if flagged_vals and normal_vals:
            print(f"\n{field}:")
            print(f"  Flagged  (n={len(flagged_vals)}): "
                  f"mean={np.mean(flagged_vals):.4f}, "
                  f"median={np.median(flagged_vals):.4f}")
            print(f"  Normal   (n={len(normal_vals)}): "
                  f"mean={np.mean(normal_vals):.4f}, "
                  f"median={np.median(normal_vals):.4f}")
        elif field in [r for r in all_rows[0]] and np.isnan(all_rows[0][field]):
            print(f"\n{field}: not present in this suite2p version's stat.npy")
else:
    print("No data found across any session.")

print("\nDone. Per-session CSVs saved in each postprocessing/ folder.")
print("Bring the printed stat.npy field list and the summary stats to discuss.")