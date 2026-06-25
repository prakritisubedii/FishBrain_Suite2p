#!/usr/bin/env python3
"""
extract_contours.py — Extract ROI contours from completed suite2p sessions

Runs on CPU only — safe to run while Optuna jobs use both GPUs.

What this does:
  For each of the 4 valid sessions, for each plane:
    - Loads stat.npy (has ypix/xpix pixel coordinates per ROI) and iscell.npy
    - For each cell classified as 'cell':
        - Builds a small binary mask from its ypix/xpix (cropped to a
          bounding box for speed — not a full 2048x2048 array per cell)
        - Traces the outline of that mask using skimage.measure.find_contours
        - Stores the contour as an array of (y, x) coordinates in FULL
          image coordinates, plus the cell's centroid (from stat['med'])
    - Saves all contours for the plane as a .npy file (list of dicts)
    - Generates a visualization: mean image with cell outlines drawn
      (instead of just center dots)

Output (per session, inside the run folder):
    contours/
        contours_plane0.npy ... contours_plane6.npy
            each is a list of dicts:
                {'roi_idx': int, 'contour': (N,2) array of (y,x), 'centroid': (y,x)}
        cell_contours_all_planes.png
            7-panel figure, mean image + cell outlines per plane

Usage:
    python extract_contours.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage.measure import find_contours

RESULTS_ROOT = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results"
SESSIONS = {
    "145544": "145544_run1_colmedian_tau07",
    "164302": "164302_run1_colmedian_tau07",
    "164628": "164628_run1_colmedian_tau07",
    "165925": "165925_run1_colmedian_tau07",
}
N_PLANES = 7

# Each ROI's xpix/ypix includes a low-weight "tail" of pixels (per-pixel
# weight given by stat['lam']) that contribute little to F.npy but make
# contours look spiky and span multiple cell bodies. LAM_FRAC keeps only
# pixels with lam >= LAM_FRAC * lam.max() for that cell, giving a tighter
# contour around the high-weight "core" — closer to the actual cell body.
#
# Set to 0.0 to disable thresholding (use all pixels, original behavior).
# If contours still look too large/spiky, try increasing toward 0.4-0.5.
# If contours look too small/fragmented, decrease toward 0.1.
LAM_FRAC = 0.4


def get_contour_for_cell(ypix, xpix, lam, lam_frac=LAM_FRAC, pad=2):
    """
    Build a small cropped mask from ROI pixel coordinates and trace its
    outline. Returns contour coordinates in FULL image (y, x) space.

    lam_frac: keep only pixels where lam >= lam_frac * lam.max().
              Set to 0.0 to keep all pixels (no thresholding).
    pad: extra border pixels around the bounding box, needed so
         find_contours can trace a closed boundary even for ROIs that
         touch the edge of their bounding box.
    """
    if lam_frac > 0:
        keep = lam >= (lam_frac * lam.max())
        ypix = ypix[keep]
        xpix = xpix[keep]

    if len(ypix) < 3:
        return None  # too few pixels left to form a contour

    y0, y1 = int(ypix.min()), int(ypix.max())
    x0, x1 = int(xpix.min()), int(xpix.max())

    h = (y1 - y0 + 1) + 2 * pad
    w = (x1 - x0 + 1) + 2 * pad

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ypix - y0 + pad, xpix - x0 + pad] = 1

    contours = find_contours(mask, level=0.5)
    if not contours:
        return None

    # Take the largest contour (most pixels) — some ROIs may have small
    # disconnected fragments from sparsery's mask
    contour = max(contours, key=len)

    # Shift back to full-image coordinates
    contour = contour.copy()
    contour[:, 0] += y0 - pad   # row / y
    contour[:, 1] += x0 - pad   # col / x

    return contour


for session_id, run_name in SESSIONS.items():
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    s2p_dir = os.path.join(run_dir, "suite2p")
    out_dir = os.path.join(run_dir, "contours")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSession {session_id}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for p in range(N_PLANES):
        plane_dir   = os.path.join(s2p_dir, f"plane{p}")
        stat_path   = os.path.join(plane_dir, "stat.npy")
        iscell_path = os.path.join(plane_dir, "iscell.npy")
        ops_path    = os.path.join(plane_dir, "ops.npy")

        if not (os.path.exists(stat_path) and os.path.exists(ops_path)):
            print(f"  Plane {p}: missing files, skipping")
            axes[p].axis('off')
            continue

        stat   = np.load(stat_path, allow_pickle=True)
        iscell = np.load(iscell_path)
        ops    = np.load(ops_path, allow_pickle=True).item()

        cell_indices = np.where(iscell[:, 0] == 1)[0]

        plane_contours = []
        n_skipped = 0
        for roi_idx in cell_indices:
            roi_idx = int(roi_idx)
            s = stat[roi_idx]
            contour = get_contour_for_cell(s['ypix'], s['xpix'], s['lam'])
            if contour is None:
                n_skipped += 1
                continue
            plane_contours.append({
                'roi_idx': roi_idx,
                'contour': contour.astype(np.float32),   # (N, 2) -> (y, x)
                'centroid': np.array(s['med'], dtype=np.float32),  # (y, x)
            })

        out_path = os.path.join(out_dir, f"contours_plane{p}.npy")
        np.save(out_path, plane_contours, allow_pickle=True)

        print(f"  Plane {p}: {len(plane_contours)}/{len(cell_indices)} "
              f"cells got contours ({n_skipped} skipped, too few pixels "
              f"after lam>={LAM_FRAC} threshold) -> {os.path.basename(out_path)}")

        # ── Visualization: mean image + outlines ──────────────────────────
        axes[p].imshow(ops['meanImg'], cmap='gray',
                       vmin=np.percentile(ops['meanImg'], 1),
                       vmax=np.percentile(ops['meanImg'], 99))

        # Use LineCollection for speed with thousands of contours
        segments = [c['contour'][:, ::-1] for c in plane_contours]  # (x, y) for plotting
        if segments:
            lc = LineCollection(segments, colors='red', linewidths=0.5, alpha=0.8)
            axes[p].add_collection(lc)

        axes[p].set_title(f"Plane {p} — {len(plane_contours)} cells")
        axes[p].axis('off')

    if N_PLANES < 8:
        axes[N_PLANES].axis('off')

    plt.suptitle(f"Cell contours — session {session_id} — {run_name}", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "cell_contours_all_planes.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

print("\nDone.")
print("\nTo load contours for a specific cell later:")
print("  contours = np.load('.../contours/contours_plane0.npy', allow_pickle=True)")
print("  contours[0]['contour']   # (N, 2) array of (y, x) boundary points")
print("  contours[0]['centroid']  # (y, x) center")
print("  contours[0]['roi_idx']   # index into stat.npy / F.npy for this plane")