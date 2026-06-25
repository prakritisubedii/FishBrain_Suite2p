
"""
postprocess.py — dF/F postprocessing for suite2p results
Session: Fish A1, any session from 20iv26 dataset

Usage:
    python postprocess.py <session_id>                           # most recent run
    python postprocess.py <session_id> --run <run_folder_name>   # specific run
    python postprocess.py <session_id> --all                     # all runs for session

Examples:
    python postprocess.py 164628
    python postprocess.py 164628 --run 164628_run1_colmedian_tau07
    python postprocess.py 164628 --all

Output goes inside the run folder:
    results/<run_folder>/postprocessing/<PP_TAG>/
        dff_all_planes.npy     ← main output: (n_cells, n_frames) dF/F array
        cell_plane.npy         ← which plane each cell came from
        cell_roi_idx.npy       ← ROI index within that plane's stat.npy
        traces_plane0.png      ← raw Fc and dF/F side by side for plane 0
        ...
        population_dff.png     ← population average — should be flat if correct

Method:
    1. Neuropil subtract: Fc = F - 0.7 * Fneu
    2. Per-cell rolling 8th-percentile baseline F0(t) over 60-second window
       F0(t) automatically tracks photobleaching because it follows the slow
       floor of each cell's trace over time.
    3. dF/F = (Fc - F0) / F0   ← divide makes it dimensionless (0.1 to 2.0 range)

Note on the /2 factor in F.npy:
    The run script stores values as (raw - col_median) / 2 to fit in int16.
    This /2 appears in both Fc and F0, so it cancels out in the dF/F ratio.
    The final dF/F values are unaffected.
"""

# ── CHANGE THIS to describe the postprocessing method ────────────────────────
# No spaces. This becomes a subfolder name inside postprocessing/.
# Examples: "rolling_prctile_win60_pct8", "rolling_prctile_win30_pct8"
PP_TAG = "rolling_prctile_win40_pct8"
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import argparse
import numpy as np
from scipy.ndimage import percentile_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="dF/F postprocessing for suite2p results.")
parser.add_argument("session_id", help="Session ID, e.g. 164628")
group = parser.add_mutually_exclusive_group()
group.add_argument("--run",  help="Specific run folder, e.g. 164628_run1_colmedian_tau07")
group.add_argument("--all",  action="store_true", help="Process all runs for this session")
args = parser.parse_args()

SESSION_ID   = args.session_id
RESULTS_ROOT = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results"

# ── Find which run folders exist for this session ────────────────────────────
all_run_folders = sorted([
    d for d in os.listdir(RESULTS_ROOT)
    if re.match(rf"^{re.escape(SESSION_ID)}_run(\d+)_", d)
    and os.path.isdir(os.path.join(RESULTS_ROOT, d))
    and os.path.isdir(os.path.join(RESULTS_ROOT, d, "suite2p"))
], key=lambda d: int(re.search(r"run(\d+)", d).group(1)))

if not all_run_folders:
    print(f"ERROR: No completed suite2p run folders found for session {SESSION_ID}")
    print(f"  Looked in: {RESULTS_ROOT}")
    sys.exit(1)

# ── Select which run(s) to postprocess ──────────────────────────────────────
if args.all:
    run_folders = all_run_folders
    print(f"Processing ALL {len(run_folders)} runs for session {SESSION_ID}:")
    for r in run_folders:
        print(f"  {r}")
elif args.run:
    if args.run not in all_run_folders:
        print(f"ERROR: Run folder '{args.run}' not found.")
        print(f"Available runs for {SESSION_ID}:")
        for r in all_run_folders:
            print(f"  {r}")
        sys.exit(1)
    run_folders = [args.run]
else:
    # Default: most recent run (highest run number)
    run_folders = [all_run_folders[-1]]
    print(f"No --run specified. Using most recent: {run_folders[0]}")

print(f"\nSession : {SESSION_ID}")
print(f"PP tag  : {PP_TAG}")

# ── dF/F parameters ──────────────────────────────────────────────────────────
FS             = 5.0            # Hz per plane
NEUROPIL_COEFF = 0.7            # standard suite2p neuropil coefficient
WIN_S          = 40.0           # rolling baseline window in seconds
WIN_FRAMES     = int(WIN_S * FS) # = 300 frames
PRCTILE        = 8              # percentile for baseline (8th = near-floor of signal)
N_PLANES       = 7

# ─────────────────────────────────────────────────────────────────────────────
# Core processing function — runs for one run folder
# ─────────────────────────────────────────────────────────────────────────────
def process_run(run_folder_name):
    run_dir = os.path.join(RESULTS_ROOT, run_folder_name)
    s2p_dir = os.path.join(run_dir, "suite2p")
    pp_dir  = os.path.join(run_dir, "postprocessing", PP_TAG)
    os.makedirs(pp_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Run      : {run_folder_name}")
    print(f"PP method: {PP_TAG}")
    print(f"Output   : postprocessing/{PP_TAG}/")
    print(f"Window   : {WIN_S:.0f}s ({WIN_FRAMES} frames) | Percentile: {PRCTILE}th")
    print("=" * 60)

    all_dff   = []
    all_plane = []
    all_idx   = []

    for p in range(N_PLANES):
        plane_dir   = os.path.join(s2p_dir, f"plane{p}")
        F_path      = os.path.join(plane_dir, "F.npy")
        Fneu_path   = os.path.join(plane_dir, "Fneu.npy")
        iscell_path = os.path.join(plane_dir, "iscell.npy")

        if not os.path.exists(F_path):
            print(f"\nPlane {p}: F.npy not found, skipping.")
            continue

        F      = np.load(F_path)
        Fneu   = np.load(Fneu_path)
        iscell = np.load(iscell_path)

        cell_idx = np.where(iscell[:, 0] == 1)[0]
        n_cells  = len(cell_idx)
        n_frames = F.shape[1]

        print(f"\nPlane {p}: {n_cells} cells, {n_frames} frames")

        if n_cells == 0:
            print(f"  No cells classified. Skipping.")
            continue

        # ── Step 1: Neuropil subtraction ──────────────────────────────────────
        # Fc = F - 0.7 * Fneu removes the contribution of surrounding neuropil
        # Both F and Fneu have the /2 scaling from the run script — it cancels in dF/F
        Fc = (F[cell_idx] - NEUROPIL_COEFF * Fneu[cell_idx]).astype(np.float64)

        # ── Step 2 + 3: Per-cell rolling percentile → dF/F ───────────────────
        #
        # For each cell independently:
        #   F0(t) = 8th percentile of Fc in a sliding 60-second window
        #
        # Why this works for photobleaching:
        #   The 8th percentile of the signal at any moment in time represents
        #   the "quiet floor" — what the cell looks like when it's not firing.
        #   Because we use a SLIDING window (not the whole trace), F0(t) slowly
        #   follows the photobleaching trend. Each cell gets its own F0(t), so
        #   fast-bleaching and slow-bleaching cells are handled independently.
        #
        # dF/F = (Fc - F0) / F0
        #   Dividing by F0 makes the result dimensionless.
        #   At baseline: Fc ≈ F0, so dF/F ≈ 0.
        #   During a spike: Fc > F0, so dF/F > 0 (typically 0.1 to 2.0).
        #
        dff = np.zeros_like(Fc, dtype=np.float32)

        for i in range(n_cells):
            # Rolling percentile baseline
            F0 = percentile_filter(Fc[i], percentile=PRCTILE, size=WIN_FRAMES)
            # Floor at 1 count to prevent divide-by-zero in very dim regions
            F0 = np.maximum(F0, 1.0)
            # Compute dF/F
            dff[i] = ((Fc[i] - F0) / F0).astype(np.float32)

        all_dff.append(dff)
        all_plane.extend([p] * n_cells)
        all_idx.extend(cell_idx.tolist())

        print(f"  dF/F stats:  mean={dff.mean():.3f}  std={dff.std():.3f}  "
              f"min={dff.min():.2f}  max={dff.max():.2f}")
        print(f"  Expected:    mean≈0.0  std≈0.1-0.5  range roughly -0.3 to +2.0")

        # ── Per-plane trace visualization ──────────────────────────────────────
        n_show = min(6, n_cells)
        t      = np.arange(n_frames) / FS
        fig, axes = plt.subplots(n_show, 2, figsize=(18, 2.5 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, 2)

        for i in range(n_show):
            # Left panel: raw neuropil-corrected fluorescence
            axes[i, 0].plot(t, Fc[i], lw=0.8, color='gray')
            axes[i, 0].set_ylabel(f"Cell {cell_idx[i]}", fontsize=8)
            if i == 0:
                axes[i, 0].set_title("Neuropil-corrected Fc (raw counts, /2 scale)",
                                     fontsize=10)

            # Right panel: dF/F
            axes[i, 1].plot(t, dff[i], lw=0.8, color='darkblue')
            axes[i, 1].axhline(0, color='gray', lw=0.5, ls='--')
            axes[i, 1].set_ylim(-0.5, 3.0)
            if i == 0:
                axes[i, 1].set_title(
                    f"dF/F  (rolling {WIN_S:.0f}s window, {PRCTILE}th pct baseline)",
                    fontsize=10)

        for col in range(2):
            axes[-1, col].set_xlabel("Time (s)")

        plt.suptitle(f"{run_folder_name}  |  Plane {p}  |  {n_cells} cells",
                     fontsize=10)
        plt.tight_layout()
        out = os.path.join(pp_dir, f"traces_plane{p}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved traces_plane{p}.png")

    # ── Save combined outputs ─────────────────────────────────────────────────
    if not all_dff:
        print("\nNo cells found across any plane. Check iscell.npy.")
        return

    dff_combined = np.vstack(all_dff)           # (total_cells, n_frames)
    plane_arr    = np.array(all_plane, dtype=np.int32)
    idx_arr      = np.array(all_idx,   dtype=np.int32)

    np.save(os.path.join(pp_dir, "dff_all_planes.npy"), dff_combined)
    np.save(os.path.join(pp_dir, "cell_plane.npy"),     plane_arr)
    np.save(os.path.join(pp_dir, "cell_roi_idx.npy"),   idx_arr)

    print(f"\nTotal cells : {len(dff_combined)}")
    print(f"dF/F shape  : {dff_combined.shape}")
    print(f"dF/F range  : {dff_combined.min():.3f}  to  {dff_combined.max():.3f}")

    # ── Population average plot ───────────────────────────────────────────────
    # If bleach correction worked, the population average should be approximately
    # flat (no slow downward drift). A wave or event around t=20s is real biology.
    t       = np.arange(dff_combined.shape[1]) / FS
    pop_avg = dff_combined.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(20, 8))

    n_trace = min(30, len(dff_combined))
    for i in range(n_trace):
        axes[0].plot(t, dff_combined[i] + i * 0.3, lw=0.5, alpha=0.7)
    axes[0].set_title(f"Individual dF/F traces (offset for visibility) — "
                      f"{len(dff_combined)} total cells")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("dF/F")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, pop_avg, lw=1.5, color='darkblue')
    axes[1].axhline(0, color='gray', lw=0.8, ls='--')
    axes[1].fill_between(t, pop_avg, 0, alpha=0.2, color='darkblue')
    axes[1].set_title(
        "Population average dF/F\n"
        "✓ Flat baseline = bleach correction worked | "
        "Any peak = real neural event (do not remove)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mean dF/F")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle(f"{run_folder_name}  |  {PP_TAG}", fontsize=11)
    plt.tight_layout()
    out = os.path.join(pp_dir, "population_dff.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved population_dff.png")
    print(f"\nAll outputs in: {pp_dir}/")

# ─────────────────────────────────────────────────────────────────────────────
# Run postprocessing for each selected run folder
# ─────────────────────────────────────────────────────────────────────────────
for folder in run_folders:
    process_run(folder)

print("\nDone.")
