#!/usr/bin/env python3
"""
seed_optuna_study.py — Create a fresh Optuna study seeded with real scores
from the previous 33 trials (whose scores were lost due to the allow_pickle bug).

This means TPE will start guided search immediately on trial 1 tonight,
rather than wasting 10 trials on random exploration.

Run this ONCE before launching optuna_tune.py tonight.

Old study.db is preserved as study_old.db — do not delete it until you've
confirmed the new study looks correct.
"""

import os
import shutil
import optuna
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

OPTUNA_DIR = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results/165925_optuna"
OLD_DB     = os.path.join(OPTUNA_DIR, "study.db")
NEW_DB     = os.path.join(OPTUNA_DIR, "study_v2.db")
STUDY_NAME = "suite2p_165925_v2"

# ── Back up old study ─────────────────────────────────────────────────────────
if os.path.exists(OLD_DB):
    backup = os.path.join(OPTUNA_DIR, "study_old.db")
    shutil.copy2(OLD_DB, backup)
    print(f"Old study backed up to: study_old.db")

# ── Real scores recovered from trial_summary.npy files ───────────────────────
# trial number -> (threshold_scaling, max_overlap, highpass_time, score)
real_trials = {
    0:  (0.716762, 0.550049, 94,  5455.82),
    1:  (0.502449, 0.567923, 70,  7119.91),
    2:  (1.326182, 0.563954, 86,  1663.50),
    3:  (1.008005, 0.854445, 98,  3634.76),
    5:  (1.437658, 0.622419, 32,  1034.12),
    6:  (1.137180, 0.802587, 50,  1970.17),
    7:  (1.044549, 0.890436, 11,  1658.78),
    8:  (0.903583, 0.510604, 99,  3986.56),
    9:  (1.593831, 0.591633, 95,  1061.78),
    10: (0.800530, 0.645141, 24,  2818.28),
    11: (1.504208, 0.624289, 61,  1055.30),
    12: (1.687920, 0.854875, 28,   687.91),
    13: (0.515840, 0.686435, 97,  8170.24),
    14: (1.197848, 0.786005, 52,  1750.01),
    15: (0.813514, 0.577742, 66,  3954.99),
    16: (1.436858, 0.891508, 22,   984.06),
    17: (1.072658, 0.864128, 80,  2867.34),
    18: (0.636639, 0.864727, 70,  6337.45),
    19: (0.587911, 0.772903, 10,  3406.60),
    30: (0.944890, 0.597943, 65,  3117.94),
}

# ── Create fresh study ────────────────────────────────────────────────────────
storage = f"sqlite:///{NEW_DB}"
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=storage,
    direction="maximize",
    load_if_exists=False,
)

print(f"Created fresh study: {STUDY_NAME}")
print(f"Database: {NEW_DB}")
print(f"\nSeeding {len(real_trials)} trials with real scores...")
print("-" * 60)

for orig_num, (ts, mo, ht, score) in sorted(real_trials.items()):
    # Create a trial with the exact parameters and score
    trial = optuna.trial.create_trial(
        params={
            "threshold_scaling": ts,
            "max_overlap":       mo,
            "highpass_time":     ht,
        },
        distributions={
            "threshold_scaling": optuna.distributions.FloatDistribution(0.3, 2.0),
            "max_overlap":       optuna.distributions.FloatDistribution(0.5, 0.9),
            "highpass_time":     optuna.distributions.IntDistribution(10, 100),
        },
        value=score,
    )
    study.add_trial(trial)
    print(f"  orig trial {orig_num:02d} -> seeded: "
          f"threshold_scaling={ts:.3f}, score={score:.2f}")

print("-" * 60)
print(f"\nStudy seeded with {len(study.trials)} trials.")
print(f"Best score so far : {study.best_value:.2f}")
print(f"Best params       : {study.best_params}")
print(f"\nTPE will now start guided search from trial 1 tonight.")
print(f"\nIMPORTANT: Update optuna_tune.py to use the new study:")
print(f"  STORAGE    = 'sqlite:///{NEW_DB}'")
print(f"  STUDY_NAME = '{STUDY_NAME}'")