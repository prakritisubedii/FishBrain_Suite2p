# FishBrain Suite2p Pipeline

This repo contains the suite2p side of a zebrafish calcium imaging analysis project. The goal is to detect neurons and extract their activity traces from light-sheet microscopy data, then compare the results against CaImAn (a different detection algorithm) to benchmark which works better.

**Reference paper:** biorxiv 2026.02.04.703741  
**CaImAn side:** Soluchi (GitHub: soluchi07)  
**Reference CaImAn pipeline:** Kiitan (GitHub: Mikito-Coder), repo: Fish_Brain_Dynamics

---

## What This Project Does (Big Picture)

1. We have zebrafish brain imaging data — videos of neurons lighting up as the fish is active
2. Suite2p finds the neurons in those videos and extracts how bright each one is over time
3. We convert that brightness into a signal called dF/F (how much each neuron changed relative to its baseline)
4. We compare those results against CaImAn to see which tool finds more real neurons

---

## Data

- **Animal:** Fish A1, expressing GCaMP8M (a fluorescent calcium indicator)
- **Sessions:** 5 recording sessions, each 140 seconds long at 5 volumes/second
- **Raw data location on GPU machine:**
```
/home/abl-workstation2/fishdynamics_data/20iv26/
```
- **File format:** Each session is one HDF5 file at this path:
```
/home/abl-workstation2/fishdynamics_data/20iv26/2026-04-20_<SESSION_ID>/raw/stack_1-A1-GCaMP8M_channel_1_obj_bottom/Cam_long_00000.lux.h5
```
- **Data shape:** `(4900, 2048, 2048)` — 4900 frames, each 2048×2048 pixels, stored as uint16
- **Volume structure:** 700 volumes × 7 planes stacked consecutively (every 7 frames = one full brain volume)
- **Primary benchmark session:** `165925` — this session has the clearest sustained rhythmic neural activity throughout the full 140 seconds, making it the best one to compare suite2p vs CaImAn on

---

## Machine & Environment

- **GPU machine:** `abl-workstation2`, 2× NVIDIA RTX A6000, CUDA 12.8
- **Conda environment:** `suite2p` (Python 3.10, suite2p 1.1.0)

> ⚠️ **Important:** NumPy must stay at version 1.26.4. Do NOT upgrade it. NumPy 2.x breaks suite2p's cell detection code (sparsedetect.py) silently.

To activate the environment:
```bash
conda activate suite2p
```

**Scripts are here:**
```
/home/abl-workstation2/Prakriti_FishBrain/suite2p/scripts/
```

**Results are here:**
```
/home/abl-workstation2/Prakriti_FishBrain/suite2p/results/
```

---

## Important Preprocessing Fix (Read This)

The light-sheet microscope creates horizontal stripe artifacts in the images — bright/dark bands that run across the whole frame and are not real neural signal.

**The original code tried to remove these stripes the wrong way.** It subtracted the average brightness of every pixel across all timepoints (a 2048×2048 image). This accidentally erased the cells' baseline brightness too, making dF/F values explode to ±100–800 (should be roughly 0 to 2).

**The fix:** subtract only the median brightness of each *column* of pixels (2048 values total, one per column). This removes the stripe artifact (which runs column-wise) without touching the cells. This is the same approach Kiitan uses in the CaImAn pipeline.

This fix is built into `run_suite2p.py` and runs automatically — you don't need to do anything extra.

---

## Scripts

### Run these — core pipeline

These are the only two scripts you need to run the pipeline from scratch:

| Script | What it does |
|--------|-------------|
| `run_suite2p.py` | Detects neurons in a session and extracts their raw fluorescence traces |
| `postprocess.py` | Converts raw fluorescence into dF/F (the normalized activity signal) |

### Already done — only run if re-tuning is needed

Hyperparameter tuning has already been completed. The best parameters have been found and are baked into `run_suite2p.py`. You only need these scripts if you want to search for even better parameters:

| Script | What it does |
|--------|-------------|
| `optuna_tune.py` | Finds the best detection parameters by running many trials automatically |
| `seed_optuna_study.py` | Seeds the tuning with known results so it searches smarter from the start |

### Diagnostic — optional inspection tools

These were used during development to investigate specific questions. Not needed for normal use:

| Script | What it does |
|--------|-------------|
| `extract_contours.py` | Draws the outlines of detected cells on the mean image |
| `crop_contours.py` | Zooms into a region to check if cells are overlapping too much |
| `analyze_false_positives.py` | Flags cells whose brightness just keeps increasing (likely not real neurons) |
| `analyze_false_positives_per_plane.py` | Same as above but broken down per imaging plane |

---

## Step-by-Step Workflow

### Step 1 — Run suite2p to detect neurons

```bash
cd /home/abl-workstation2/Prakriti_FishBrain/suite2p/scripts
conda activate suite2p
python run_suite2p.py <session_id>

# Example for the primary benchmark session:
python run_suite2p.py 165925
```

Before running, open `run_suite2p.py` and update:
- `RUN_TAG` at the top — a short label describing what's different about this run (no spaces, e.g. `tuned_optuna_v2`)
- The three detection parameters in the settings block (use the tuned values from Step 2 below)

The script auto-numbers runs so it never overwrites previous results. Output goes to:
```
results/<session_id>_run<N>_<RUN_TAG>/
```

Key files produced per imaging plane:
- `stat.npy` — shape and location of each detected cell
- `F.npy` / `Fneu.npy` — raw fluorescence trace for each cell and its surrounding neuropil
- `iscell.npy` — classifier decision: column 0 = is it a cell (1/0), column 1 = confidence score
- `col_medians.npy` — the stripe removal reference values (needed if you run Optuna later)


---

### Step 2 — Use these tuned parameters (already found)

Suite2p has settings that control how aggressively it detects cells. These were found by running 50+ automated trials using Optuna. **Just use these values in `run_suite2p.py` — no need to re-tune:**

| Parameter | Best value | Default (before tuning) | What it controls |
|-----------|-----------|--------------------------|-----------------|
| `threshold_scaling` | **0.303** | 1.0 | How strict the cell detection threshold is — lower finds more cells |
| `max_overlap` | **0.810** | 0.75 | How much two cells can overlap before one is removed |
| `highpass_time` | **91** | 100 | High-pass filter cutoff in frames |

These come from Trial 32 of the v2 Optuna study and improved detection from 3,602 cells to ~10,900 cells on session 165925 — roughly a 3× improvement.

The key insight: **`threshold_scaling` is by far the most important parameter.** Scores drop sharply as it increases above ~0.75. The sweet spot is 0.30–0.35.

To inspect all tuning trial results:
```python
import optuna
study = optuna.load_study(
    study_name='suite2p_165925_v2',
    storage='sqlite:////home/abl-workstation2/Prakriti_FishBrain/suite2p/results/165925_optuna/study_v2.db'
)
print(study.trials_dataframe().sort_values('value', ascending=False))
```

**If you want to run more tuning trials:**

First seed a fresh Optuna study with known scores so it searches intelligently from trial 1:
```bash
python seed_optuna_study.py
```

Then launch on both GPUs in separate screen sessions:
```bash
# GPU 0
screen -S optuna_a
conda activate suite2p
python optuna_tune.py 165925 cuda:0 10

# GPU 1 (open a new terminal first)
screen -S optuna_b
conda activate suite2p
python optuna_tune.py 165925 cuda:1 10
```

> ⚠️ **Disk warning:** Each trial uses ~2GB of disk. Monitor with:
> ```bash
> df -h /home/abl-workstation2/
> ```

---

### Step 3 — Postprocessing (compute dF/F)

dF/F tells you how much each neuron's brightness changed relative to its own baseline. A value of 0.5 means the neuron got 50% brighter than its baseline — a likely real response.

```bash
# Run on the most recent result for a session:
python postprocess.py 165925

# Run on a specific result folder:
python postprocess.py 165925 --run 165925_run3_tuned_optuna_v2

# Run on all result folders for a session:
python postprocess.py 165925 --all
```

**Method:** For each cell, we compute a rolling 8th-percentile baseline (using a 40-second sliding window). This baseline represents the cell's "resting" brightness. Then:
```
dF/F = (fluorescence - baseline) / baseline
```

Outputs go to: `<run_folder>/postprocessing/rolling_prctile_win40_pct8/`
- `dff_all_planes.npy` — dF/F traces for all cells, shape (n_cells, n_frames)
- `cell_plane.npy` — which imaging plane each cell belongs to
- `cell_roi_idx.npy` — index of each cell within its plane
- Per-plane trace plots and a population average plot (PNGs)


---

## Results Summary

The `results_summary/` folder contains PNG plots from completed runs so you can see what the output looks like without running anything.

### Before vs after tuning (session 165925)

| Run | Parameters | Cells detected |
|-----|-----------|---------------|
| `run1` — default | threshold_scaling=1.0 | 3,602 |
| `run3` — tuned | threshold_scaling=0.303 | ~10,900 |

The cell location plots show clearly how many more neurons are detected after tuning.

### Folder structure in results_summary

```
results_summary/results/
  145544_run1_colmedian_tau07/     ← session 145544, default params
  164302_run1_colmedian_tau07/     ← session 164302, default params
  164628_run1_colmedian_tau07/     ← session 164628, default params
  165925_run1_colmedian_tau07/     ← session 165925, default params (before tuning)
  165925_run2_tuned_optuna_t13/    ← session 165925, intermediate tuned params
  165925_run3_tuned_optuna_v2/     ← session 165925, BEST tuned params (use this)
```

Each folder contains:
- `cell_locations_all_planes.png` — mean image of each plane with detected cells marked in red
- `postprocessing/rolling_prctile_win40_pct8/population_dff.png` — average dF/F across all cells
- `postprocessing/rolling_prctile_win40_pct8/traces_plane*.png` — individual cell traces per plane

---

## Completed Runs on the Machine

### Default parameters (run1) — all 4 valid sessions
| Session | Cells detected | Notes |
|---------|---------------|-------|
| 145544 | 4,491 | Quiet after initial onset |
| 164302 | 5,208 | Rich activity throughout |
| 164628 | 4,512 | Quiet after initial onset |
| 165925 | 3,602 | Primary benchmark session |

### Tuned parameters (run3) — session 165925
| Session | Cells detected | Optuna score |
|---------|---------------|-------------|
| 165925 | ~10,900 | 10,932 |

---

## Things Still To Do

- [ ] **Cross-session validation** — run tuned parameters on sessions 145544, 164302, 164628 to confirm they generalize
- [ ] **Time-split validation** — run on first 70s vs last 70s of session 165925 to check consistency
