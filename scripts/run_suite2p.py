
"""
run_suite2p.py — Automated suite2p full run for zebrafish GCaMP8M data
Session: Fish A1, any session from 20iv26 dataset

Usage:
    python run_suite2p.py <session_id>

Examples:
    python run_suite2p.py 164628
    python run_suite2p.py 165925
    python run_suite2p.py 144321

Output folder: results/<session_id>_run<N>_<RUN_TAG>/
Run number is auto-detected — just run the script, it won't overwrite anything.

Changes from previous version:
  - Stripe removal: column-median only (1D, shape Lx) instead of full frame mean (2D, Ly x Lx)
  - tau: 0.7s for GCaMP8M (was 1.5, which is GCaMP6s)
  - soma_crop removed (not in suite2p v1.1.0 API)
  - col_medians.npy saved to output folder for use in postprocessing
"""

# ── CHANGE THIS to describe what is different about this run ─────────────────
# No spaces. Examples: "colmedian_tau07", "colmedian_tau15", "fullmean_tau07"
RUN_TAG = "tuned_optuna_v2"
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import re
import h5py
import numpy as np
import math
import time
import shutil
import argparse
import suite2p
import suite2p.io

# ── Parse session ID from command line ────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run suite2p on a GCaMP8M session.")
parser.add_argument("session_id", help="Session ID suffix, e.g. 164628")
args = parser.parse_args()
SESSION_ID = args.session_id

# ── Fixed paths — only change these if the lab machine changes ───────────────
DATE_PREFIX  = "2026-04-20"   # all 20iv26 sessions are on this date
DATA_ROOT    = "/home/abl-workstation2/fishdynamics_data/20iv26"
RESULTS_ROOT = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results"

H5_PATH = os.path.join(
    DATA_ROOT,
    f"{DATE_PREFIX}_{SESSION_ID}",
    "raw",
    "stack_1-A1-GCaMP8M_channel_1_obj_bottom",
    "Cam_long_00000.lux.h5"
)

if not os.path.exists(H5_PATH):
    print(f"ERROR: H5 file not found:\n  {H5_PATH}", file=sys.stderr)
    sys.exit(1)

# ── Auto-detect run number so we never overwrite previous results ────────────
os.makedirs(RESULTS_ROOT, exist_ok=True)
existing_runs = [
    d for d in os.listdir(RESULTS_ROOT)
    if re.match(rf"^{re.escape(SESSION_ID)}_run(\d+)_", d)
    and os.path.isdir(os.path.join(RESULTS_ROOT, d))
]
if existing_runs:
    run_nums = [int(re.search(r"run(\d+)", d).group(1)) for d in existing_runs]
    run_num  = max(run_nums) + 1
else:
    run_num = 1

RUN_NAME    = f"{SESSION_ID}_run{run_num}_{RUN_TAG}"
OUTPUT_DIR  = os.path.join(RESULTS_ROOT, RUN_NAME)
SCRATCH_DIR = os.path.join(OUTPUT_DIR, "scratch")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

# ── Acquisition parameters ────────────────────────────────────────────────────
N_PLANES   = 7
FS_VOLUME  = 5.0    # Hz per plane (700 frames / 140 seconds)
TAU        = 0.7    # GCaMP8M medium kinetics ~0.7s. GCaMP6s would be 1.5.
CHUNK_VOLS = 50     # volumes to load at a time during column median computation

print("=" * 60)
print(f"SUITE2P FULL RUN")
print(f"  Session  : {SESSION_ID}")
print(f"  Run name : {RUN_NAME}")
print(f"  H5 file  : {H5_PATH}")
print(f"  Output   : {OUTPUT_DIR}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Compute per-plane column medians for stripe removal
#
# What this does:
#   For each of the 7 planes, we compute one median value per column of pixels
#   (shape: Lx = 2048 values). This represents the background brightness pattern
#   of that column due to the light sheet — the actual stripe artifact.
#
# What this does NOT do:
#   It does NOT subtract the cell signal. Cell bodies are round, not column-shaped,
#   so their brightness averages out across the Y axis and barely affects the median.
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 1: Computing per-plane column medians (stripe removal reference)...")
t0 = time.time()

with h5py.File(H5_PATH, "r") as f:
    n_frames_total = f["Data"].shape[0]
    Ly             = f["Data"].shape[1]
    Lx             = f["Data"].shape[2]
    n_volumes      = n_frames_total // N_PLANES
    n_chunks_total = math.ceil(n_volumes / CHUNK_VOLS)

    print(f"  {n_frames_total} frames = {n_volumes} volumes × {N_PLANES} planes")
    print(f"  Frame size: {Ly} × {Lx}")

    # col_medians[p] has shape (Lx,) — one median per column, for each plane
    col_medians   = np.zeros((N_PLANES, Lx), dtype=np.float64)
    n_chunks_done = 0

    for vol_start in range(0, n_volumes, CHUNK_VOLS):
        vol_end     = min(vol_start + CHUNK_VOLS, n_volumes)
        frame_start = vol_start * N_PLANES
        frame_end   = vol_end   * N_PLANES
        chunk       = f["Data"][frame_start:frame_end].astype(np.float32)

        for p in range(N_PLANES):
            plane_frames = chunk[p::N_PLANES]   # shape: (n_vols_in_chunk, Ly, Lx)
            # Median over both the time axis (0) and the Y axis (1) → shape (Lx,)
            # This gives the median column brightness, robust to bright cell transients
            col_medians[p] += np.median(plane_frames, axis=(0, 1))

        n_chunks_done += 1
        if n_chunks_done % 3 == 0 or n_chunks_done == n_chunks_total:
            print(f"  Chunk {n_chunks_done}/{n_chunks_total}...")

col_medians /= n_chunks_total    # average of per-chunk medians
col_medians  = col_medians.astype(np.float32)

# Save for reference and for postprocessing
np.save(os.path.join(OUTPUT_DIR, "col_medians.npy"), col_medians)

print(f"  Done. ({time.time()-t0:.1f}s)")
print(f"  col_medians shape: {col_medians.shape}  ← should be ({N_PLANES}, {Lx})")
print(f"  Saved col_medians.npy")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Monkey-patch suite2p's H5 reader to apply stripe removal on the fly
#
# suite2p reads the H5 file in batches and converts to its own binary format.
# We intercept that step to subtract col_medians before the data is written.
# This way, the registered binary and all downstream outputs are stripe-free.
#
# Key fix vs previous version:
#   OLD: im[plane_idx] -= col_means[p][np.newaxis, :, :]
#        col_means[p] had shape (Ly, Lx) — subtracted the entire mean image!
#        This removed cell signal and set the baseline to ~zero.
#
#   NEW: im[plane_idx] -= col_medians[p][np.newaxis, np.newaxis, :]
#        col_medians[p] has shape (Lx,) — broadcasts as (1, 1, Lx)
#        This subtracts only column-wise background, leaving cell brightness intact.
# ─────────────────────────────────────────────────────────────────────────────
def h5py_to_binary_with_stripe_removal(dbs, settings, reg_file, reg_file_chan2):
    nplanes   = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    h5list    = dbs[0]["file_list"]
    keys      = dbs[0]["h5py_key"]
    if isinstance(keys, str):
        keys = [keys]

    iall = 0
    for j in range(nplanes):
        dbs[j]["nframes_per_folder"] = np.zeros(len(h5list), np.int32)

    for ih5, h5 in enumerate(h5list):
        with h5py.File(h5, "r") as f:
            for key in keys:
                nframes_all = f[key].shape[0]
                nbatch      = nplanes * max(1, dbs[0]["batch_size"] // nplanes)
                ik          = 0

                while True:
                    irange = np.arange(ik, min(ik + nbatch, nframes_all), 1)
                    if irange.size == 0:
                        break

                    im            = f[key][irange, :, :].astype(np.float32)
                    nframes_batch = im.shape[0]

                    # ── Stripe removal: subtract per-column median ──────────
                    # col_medians[p] shape: (Lx,)
                    # Expanded to (1, 1, Lx) → broadcasts over (n_frames, Ly, Lx)
                    for p in range(nplanes):
                        plane_idx = np.arange(p, nframes_batch, nplanes)
                        if len(plane_idx) > 0:
                            im[plane_idx] -= col_medians[p][np.newaxis, np.newaxis, :]

                    im = np.clip(im, 0, None)
                    # Divide by 2 to keep values within int16 range (-32768 to 32767)
                    # This scale factor cancels out when computing dF/F (a ratio)
                    im = (im / 2).astype(np.int16)

                    for j in range(nplanes):
                        if iall == 0:
                            dbs[j]["meanImg"] = np.zeros(
                                (im.shape[1], im.shape[2]), np.float32)
                            if nchannels > 1:
                                dbs[j]["meanImg_chan2"] = np.zeros(
                                    (im.shape[1], im.shape[2]), np.float32)
                            dbs[j]["nframes"] = 0

                        plane_idx = np.arange(j, nframes_batch, nplanes)
                        if len(plane_idx) > 0:
                            im2write = im[plane_idx].astype(np.int16)
                            reg_file[j].write(bytearray(im2write))
                            dbs[j]["meanImg"]  += im2write.astype(np.float32).sum(axis=0)
                            dbs[j]["nframes"]  += im2write.shape[0]
                            dbs[j]["nframes_per_folder"][ih5] += im2write.shape[0]

                    ik   += nframes_batch
                    iall += nframes_batch

    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["Ly"] = im2write.shape[1]
        db["Lx"] = im2write.shape[2]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs

suite2p.io.h5py_to_binary = h5py_to_binary_with_stripe_removal
print("\nStep 2: Stripe removal patch applied (column-median only, 1D).")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Suite2p settings
# ─────────────────────────────────────────────────────────────────────────────
settings = suite2p.default_settings()

settings['fs']           = FS_VOLUME   # 5.0 Hz per plane
settings['diameter']     = [20, 20]
settings['tau']          = TAU         # 0.7s for GCaMP8M
settings['torch_device'] = 'cuda:0'

settings['run']['do_registration']  = 1
settings['run']['do_detection']     = True
settings['run']['do_deconvolution'] = True

settings['registration']['nimg_init']   = 400
settings['registration']['batch_size']  = 100
settings['registration']['maxregshift'] = 0.1
settings['registration']['block_size']  = [128, 128]

settings['detection']['algorithm']         = 'sparsery'
settings['detection']['threshold_scaling'] = 0.303
settings['detection']['max_overlap']       = 0.810
settings['detection']['highpass_time']     = 91
# soma_crop removed — not in v1.1.0 sparsery API

settings['classification']['use_builtin_classifier'] = True

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: db — per-run file and path configuration
# ─────────────────────────────────────────────────────────────────────────────
db = {
    'data_path':           [os.path.dirname(H5_PATH)],
    'file_list':           [H5_PATH],
    'look_one_level_down': False,
    'input_format':        'h5',
    'h5py_key':            'Data',
    'nplanes':             N_PLANES,
    'nchannels':           1,
    'save_path0':          OUTPUT_DIR,
    'fast_disk':           SCRATCH_DIR,
    'nframes':             n_frames_total // N_PLANES,
}

print("\n" + "=" * 60)
print("Step 3: Starting suite2p pipeline...")
print(f"  fs={settings['fs']} Hz  |  tau={settings['tau']}s  |  diameter={settings['diameter']} px")
print(f"  nplanes={N_PLANES}  |  nframes/plane={n_frames_total // N_PLANES}")
print("=" * 60)
t1 = time.time()

output_ops = suite2p.run_s2p(db=db, settings=settings)

total_time = time.time() - t1
print("=" * 60)
print(f"Suite2p complete. Time: {total_time/60:.1f} minutes")
print(f"Results saved to: {OUTPUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Summary across all planes
# ─────────────────────────────────────────────────────────────────────────────
print("\nResults per plane:")
total_rois = total_cells = 0
for p in range(N_PLANES):
    plane_dir   = os.path.join(OUTPUT_DIR, "suite2p", f"plane{p}")
    stat_path   = os.path.join(plane_dir, "stat.npy")
    iscell_path = os.path.join(plane_dir, "iscell.npy")
    if os.path.exists(stat_path) and os.path.exists(iscell_path):
        stat    = np.load(stat_path, allow_pickle=True)
        iscell  = np.load(iscell_path)
        n_rois  = len(stat)
        n_cells = int(iscell[:, 0].sum())
        total_rois  += n_rois
        total_cells += n_cells
        print(f"  Plane {p}: {n_rois} ROIs, {n_cells} cells")
    else:
        print(f"  Plane {p}: results not found")
print(f"\nTotal: {total_rois} ROIs, {total_cells} cells")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Visualization — mean image with cell locations for each plane
# ─────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for p in range(N_PLANES):
        plane_dir   = os.path.join(OUTPUT_DIR, "suite2p", f"plane{p}")
        ops_path    = os.path.join(plane_dir, "ops.npy")
        stat_path   = os.path.join(plane_dir, "stat.npy")
        iscell_path = os.path.join(plane_dir, "iscell.npy")

        if os.path.exists(ops_path):
            ops    = np.load(ops_path, allow_pickle=True).item()
            stat   = np.load(stat_path, allow_pickle=True)
            iscell = np.load(iscell_path)
            n_c    = int(iscell[:, 0].sum())

            axes[p].imshow(ops['meanImg'], cmap='gray',
                           vmin=np.percentile(ops['meanImg'], 1),
                           vmax=np.percentile(ops['meanImg'], 99))
            for i, s in enumerate(stat):
                if iscell[i, 0] == 1:
                    axes[p].plot(s['med'][1], s['med'][0], 'r.', markersize=1.5)
            axes[p].set_title(f"Plane {p} — {n_c} cells")
            axes[p].axis('off')
        else:
            axes[p].set_title(f"Plane {p} — no results")
            axes[p].axis('off')

    axes[7].axis('off')
    plt.suptitle(f"Suite2p — session {SESSION_ID} — {RUN_NAME}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cell_locations_all_planes.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nVisualization saved: {out_path}")
except Exception as e:
    print(f"\nVisualization failed (non-critical): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Cleanup scratch files
# ─────────────────────────────────────────────────────────────────────────────
print("\nCleaning up scratch files...")
if os.path.exists(SCRATCH_DIR):
    shutil.rmtree(SCRATCH_DIR)
    print(f"Deleted: {SCRATCH_DIR}")

print(f"\nDone. Run folder: {RUN_NAME}")
print(f"Next step: python postprocess.py {SESSION_ID}")
