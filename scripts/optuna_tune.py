#!/usr/bin/env python3
"""
optuna_tune.py — Hyperparameter tuning for suite2p using Optuna

Usage:
    python optuna_tune.py <session_id> <gpu_device> [n_trials]

Examples:
    python optuna_tune.py 165925 cuda:0 8
    python optuna_tune.py 164302 cuda:1 8

Run these TWO COMMANDS IN TWO SEPARATE SCREEN SESSIONS to use both GPUs
in parallel:

    screen -S optuna_165925
    conda activate suite2p
    cd /home/abl-workstation2/Prakriti_FishBrain/suite2p/scripts
    python optuna_tune.py 165925 cuda:0 8
    (Ctrl+A D to detach)

    screen -S optuna_164302
    conda activate suite2p
    cd /home/abl-workstation2/Prakriti_FishBrain/suite2p/scripts
    python optuna_tune.py 164302 cuda:1 8
    (Ctrl+A D to detach)

What this does:
  - Reuses col_medians.npy already computed from the run1_colmedian_tau07 run
    (skips recomputing stripe removal reference — saves ~2 minutes per trial)
  - For each trial: suggests threshold_scaling, max_overlap, highpass_time
  - Runs suite2p at FULL RESOLUTION (2048x2048) with those settings
  - Score = sum of classifier confidence (iscell[:,1]) across all 7 planes
  - Deletes large binary files after each trial to save disk space
  - Keeps stat.npy, iscell.npy, ops.npy for every trial so you can inspect later
  - Saves study to a SQLite file — safe to resume if interrupted

Search space (focused, 3 parameters):
    threshold_scaling : 0.3 - 2.0   (extended lower bound from 0.5; prior best was 0.516)
    max_overlap       : 0.5 - 0.9   (default was 0.75)
    highpass_time     : 10  - 100   (default was 100)

Fixes vs previous version:
  - allow_pickle=True added to np.save(trial_summary.npy) — fixes Optuna
    receiving None for all trials and TPE never getting feedback
  - detect_outputs.npy, reg_outputs.npy, spks.npy added to cleanup list —
    these were the actual disk hogs (~130MB/plane), not data.bin
  - threshold_scaling lower bound extended to 0.3 — prior best (0.516) was
    at the edge of the old search range, true optimum may be below 0.5

Estimated time per trial: ~35-40 minutes (same as a normal full run).
With 8 trials that's ~5-5.5 hours.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import shutil
import time
import h5py
import numpy as np
import suite2p
import suite2p.io
import optuna

# ── Parse args ────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: python optuna_tune.py <session_id> <gpu_device> [n_trials]")
    sys.exit(1)

SESSION_ID = sys.argv[1]
GPU_DEVICE = sys.argv[2]          # 'cuda:0' or 'cuda:1'
N_TRIALS   = int(sys.argv[3]) if len(sys.argv) > 3 else 8

# ── Fixed paths ──────────────────────────────────────────────────────────────
DATE_PREFIX  = "2026-04-20"
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

# ── Reuse col_medians from the existing run1 ──────────────────────────────────
COL_MEDIANS_PATH = os.path.join(
    RESULTS_ROOT, f"{SESSION_ID}_run1_colmedian_tau07", "col_medians.npy"
)
if not os.path.exists(COL_MEDIANS_PATH):
    print(f"ERROR: col_medians.npy not found at:\n  {COL_MEDIANS_PATH}")
    print("Run run_suite2p.py for this session first, or this script needs")
    print("to compute col_medians from scratch (not implemented here).")
    sys.exit(1)

col_medians = np.load(COL_MEDIANS_PATH)
print(f"Loaded col_medians: shape {col_medians.shape}")

# ── Acquisition parameters ────────────────────────────────────────────────────
N_PLANES  = 7
FS_VOLUME = 5.0
TAU       = 0.7

# ── Optuna study setup ─────────────────────────────────────────────────────────
OPTUNA_DIR = os.path.join(RESULTS_ROOT, f"{SESSION_ID}_optuna")
os.makedirs(OPTUNA_DIR, exist_ok=True)
STORAGE    = "sqlite:////home/abl-workstation2/Prakriti_FishBrain/suite2p/results/165925_optuna/study_v2.db"
STUDY_NAME = f"suite2p_{SESSION_ID}_v2"

print("=" * 60)
print(f"OPTUNA TUNING — session {SESSION_ID}")
print(f"  GPU device : {GPU_DEVICE}")
print(f"  Trials     : {N_TRIALS}")
print(f"  Study DB   : {STORAGE}")
print(f"  Output dir : {OPTUNA_DIR}")
print("=" * 60)

# Read frame count once
with h5py.File(H5_PATH, "r") as f:
    n_frames_total = f["Data"].shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch factory — needs col_medians and nplanes in scope
# ─────────────────────────────────────────────────────────────────────────────
def make_h5py_to_binary_patch(col_medians, nplanes):
    def h5py_to_binary_with_stripe_removal(dbs, settings, reg_file, reg_file_chan2):
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
                    nbatch = nplanes * max(1, dbs[0]["batch_size"] // nplanes)
                    ik = 0

                    while True:
                        irange = np.arange(ik, min(ik + nbatch, nframes_all), 1)
                        if irange.size == 0:
                            break

                        im = f[key][irange, :, :].astype(np.float32)
                        nframes_batch = im.shape[0]

                        for p in range(nplanes):
                            plane_idx = np.arange(p, nframes_batch, nplanes)
                            if len(plane_idx) > 0:
                                im[plane_idx] -= col_medians[p][np.newaxis, np.newaxis, :]

                        im = np.clip(im, 0, None)
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

    return h5py_to_binary_with_stripe_removal


suite2p.io.h5py_to_binary = make_h5py_to_binary_patch(col_medians, N_PLANES)


# ─────────────────────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial):
    # FIX: extended lower bound to 0.3 (prior best 0.516 was at edge of old range)
    threshold_scaling = trial.suggest_float("threshold_scaling", 0.3, 2.0)
    max_overlap       = trial.suggest_float("max_overlap", 0.5, 0.9)
    highpass_time     = trial.suggest_int("highpass_time", 10, 100)

    trial_dir   = os.path.join(OPTUNA_DIR, f"trial_{trial.number:03d}")
    scratch_dir = os.path.join(trial_dir, "scratch")
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Trial {trial.number}: threshold_scaling={threshold_scaling:.3f}, "
          f"max_overlap={max_overlap:.3f}, highpass_time={highpass_time}")
    print("=" * 60)

    settings = suite2p.default_settings()
    settings['fs']           = FS_VOLUME
    settings['diameter']     = [20, 20]
    settings['tau']          = TAU
    settings['torch_device'] = GPU_DEVICE

    settings['run']['do_registration']  = 1
    settings['run']['do_detection']     = True
    settings['run']['do_deconvolution'] = False   # skip — not needed for score, saves time

    settings['registration']['nimg_init']   = 400
    settings['registration']['batch_size']  = 100
    settings['registration']['maxregshift'] = 0.1
    settings['registration']['block_size']  = [128, 128]

    settings['detection']['algorithm']         = 'sparsery'
    settings['detection']['threshold_scaling'] = threshold_scaling
    settings['detection']['max_overlap']       = max_overlap
    settings['detection']['highpass_time']     = highpass_time

    settings['classification']['use_builtin_classifier'] = True

    db = {
        'data_path':           [os.path.dirname(H5_PATH)],
        'file_list':           [H5_PATH],
        'look_one_level_down': False,
        'input_format':        'h5',
        'h5py_key':            'Data',
        'nplanes':             N_PLANES,
        'nchannels':           1,
        'save_path0':          trial_dir,
        'fast_disk':           scratch_dir,
        'nframes':             n_frames_total // N_PLANES,
    }

    t0 = time.time()
    try:
        suite2p.run_s2p(db=db, settings=settings)
    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        shutil.rmtree(trial_dir, ignore_errors=True)
        return 0.0

    elapsed = time.time() - t0
    print(f"Trial {trial.number} suite2p run took {elapsed/60:.1f} min")

    # ── Score: sum of classifier confidence across all planes ────────────────
    total_score = 0.0
    total_cells = 0
    for p in range(N_PLANES):
        iscell_path = os.path.join(trial_dir, "suite2p", f"plane{p}", "iscell.npy")
        if os.path.exists(iscell_path):
            iscell = np.load(iscell_path)
            total_score += float(iscell[:, 1].sum())
            total_cells += int(iscell[:, 0].sum())

    print(f"Trial {trial.number} score: {total_score:.2f} "
          f"({total_cells} cells classified as 'cell')")

    # FIX: allow_pickle=True so Optuna receives the score and TPE gets feedback
    np.save(os.path.join(trial_dir, "trial_summary.npy"), {
        "threshold_scaling": threshold_scaling,
        "max_overlap": max_overlap,
        "highpass_time": highpass_time,
        "score": total_score,
        "n_cells": total_cells,
        "elapsed_min": elapsed / 60,
    }, allow_pickle=True)

    # ── Clean up large files to save disk space ───────────────────────────────
    # FIX: added detect_outputs.npy, reg_outputs.npy, spks.npy — these were
    # the actual disk hogs (~130MB/plane); data.bin was already being deleted
    for p in range(N_PLANES):
        plane_dir = os.path.join(trial_dir, "suite2p", f"plane{p}")
        if not os.path.exists(plane_dir):
            continue

        for fname in ["data.bin", "data_raw.bin", "data_chan2.bin",
                      "detect_outputs.npy", "reg_outputs.npy", "spks.npy"]:
            fpath = os.path.join(plane_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        # Strip large arrays from ops.npy, keep only small metadata fields
        ops_path = os.path.join(plane_dir, "ops.npy")
        if os.path.exists(ops_path):
            ops = np.load(ops_path, allow_pickle=True).item()
            keys_to_keep = {
                "Ly", "Lx", "fs", "tau", "nframes", "nplanes",
                "threshold_scaling", "max_overlap", "highpass_time",
                "yrange", "xrange"
            }
            ops_small = {k: v for k, v in ops.items() if k in keys_to_keep}
            np.save(ops_path, ops_small)

    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return total_score


# ─────────────────────────────────────────────────────────────────────────────
# Run the study
# ─────────────────────────────────────────────────────────────────────────────
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE,
    direction="maximize",
    load_if_exists=True,
)

print(f"\nStarting Optuna study with {N_TRIALS} trials...")
print(f"(Existing trials in study so far: {len(study.trials)})")

study.optimize(objective, n_trials=N_TRIALS)

print("\n" + "=" * 60)
print("STUDY COMPLETE")
print(f"Best score      : {study.best_value:.2f}")
print(f"Best parameters : {study.best_params}")
print("=" * 60)
print(f"\nFull study saved to: {STORAGE}")
print(f"All trial outputs in: {OPTUNA_DIR}/trial_XXX/")
print("\nTo inspect results later:")
print(f"  import optuna")
print(f"  study = optuna.load_study(study_name='{STUDY_NAME}', storage='{STORAGE}')")
print(f"  print(study.trials_dataframe())")