#!/usr/bin/env python3
"""
crop_contours.py — Zoom into a region to inspect contour density/overlap

Runs on CPU only.

What this does:
  Loads the mean image and contours for one session/plane, crops a small
  region, and plots it at full resolution with contours drawn — so you can
  see individual cell outlines clearly instead of a dense red blob.

  Also computes a pairwise overlap statistic for cells in the cropped
  region: for each pair of nearby cells, what fraction of the smaller
  ROI's area overlaps with the other. This gives a number, not just a
  picture, to judge whether over-segmentation is happening.

Usage:
    python crop_contours.py <session_id> <plane> <center_y> <center_x> [crop_size]

Example (crop a 200x200 region centered at y=1000, x=1000 in Plane 0):
    python crop_contours.py 164628 0 1000 1000 200

If you don't know where to look, run with no center coordinates and it
will pick the densest 200x200 region automatically based on cell centroid
count:
    python crop_contours.py 164628 0
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

RESULTS_ROOT = "/home/abl-workstation2/Prakriti_FishBrain/suite2p/results"
SESSIONS = {
    "145544": "145544_run1_colmedian_tau07",
    "164302": "164302_run1_colmedian_tau07",
    "164628": "164628_run1_colmedian_tau07",
    "165925": "165925_run1_colmedian_tau07",
}

if len(sys.argv) < 3:
    print("Usage: python crop_contours.py <session_id> <plane> [center_y center_x] [crop_size]")
    sys.exit(1)

SESSION_ID = sys.argv[1]
PLANE      = int(sys.argv[2])

if SESSION_ID not in SESSIONS:
    print(f"Unknown session {SESSION_ID}. Valid: {list(SESSIONS.keys())}")
    sys.exit(1)

run_name = SESSIONS[SESSION_ID]
run_dir  = os.path.join(RESULTS_ROOT, run_name)

contours_path = os.path.join(run_dir, "contours", f"contours_plane{PLANE}.npy")
ops_path      = os.path.join(run_dir, "suite2p", f"plane{PLANE}", "ops.npy")

if not os.path.exists(contours_path):
    print(f"Contours not found: {contours_path}")
    print("Run extract_contours.py first.")
    sys.exit(1)

contours = np.load(contours_path, allow_pickle=True)
ops      = np.load(ops_path, allow_pickle=True).item()
meanImg  = ops['meanImg']
Ly, Lx   = meanImg.shape

print(f"Session {SESSION_ID}, Plane {PLANE}: {len(contours)} cells total")

# ── Determine crop center ─────────────────────────────────────────────────
if len(sys.argv) >= 5:
    cy = int(sys.argv[3])
    cx = int(sys.argv[4])
    crop_size = int(sys.argv[5]) if len(sys.argv) >= 6 else 200
else:
    # Auto-find densest region: grid the image into 200x200 blocks,
    # count centroids per block, pick the densest
    crop_size = 200
    grid_y = Ly // crop_size
    grid_x = Lx // crop_size
    counts = np.zeros((grid_y, grid_x), dtype=int)

    for c in contours:
        y, x = c['centroid']
        gy = min(int(y // crop_size), grid_y - 1)
        gx = min(int(x // crop_size), grid_x - 1)
        counts[gy, gx] += 1

    best = np.unravel_index(np.argmax(counts), counts.shape)
    cy = best[0] * crop_size + crop_size // 2
    cx = best[1] * crop_size + crop_size // 2
    print(f"Auto-selected densest {crop_size}x{crop_size} region: "
          f"center=({cy},{cx}), {counts[best]} cells")

half = crop_size // 2
y0, y1 = max(0, cy - half), min(Ly, cy + half)
x0, x1 = max(0, cx - half), min(Lx, cx + half)

print(f"Crop region: y=[{y0}:{y1}], x=[{x0}:{x1}]")

# ── Find cells whose centroid falls in this crop ──────────────────────────
cells_in_crop = []
for c in contours:
    y, x = c['centroid']
    if y0 <= y < y1 and x0 <= x < x1:
        cells_in_crop.append(c)

print(f"Cells with centroid in crop: {len(cells_in_crop)}")

# ── Pairwise overlap check ──────────────────────────────────────────────────
# Build a full-image mask per cell (only for cells in/near the crop) and
# compute, for each pair, overlap_area / smaller_area
def contour_to_mask(contour, shape):
    """Fill the contour polygon into a boolean mask of given shape."""
    from matplotlib.path import Path
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.vstack((yy.ravel(), xx.ravel())).T
    path = Path(contour)  # contour is (y, x) pairs
    mask = path.contains_points(points).reshape(shape)
    return mask

# Only build masks for cells in crop, using LOCAL crop-sized arrays for speed
crop_h, crop_w = y1 - y0, x1 - x0
masks = []
for c in cells_in_crop:
    local_contour = c['contour'].copy()
    local_contour[:, 0] -= y0
    local_contour[:, 1] -= x0
    # Clip to crop bounds
    local_contour[:, 0] = np.clip(local_contour[:, 0], 0, crop_h - 1)
    local_contour[:, 1] = np.clip(local_contour[:, 1], 0, crop_w - 1)
    mask = contour_to_mask(local_contour, (crop_h, crop_w))
    masks.append(mask)

print("\nPairwise overlap (overlap_area / smaller_cell_area):")
high_overlap_pairs = 0
total_pairs = 0
for i in range(len(masks)):
    for j in range(i + 1, len(masks)):
        area_i = masks[i].sum()
        area_j = masks[j].sum()
        if area_i == 0 or area_j == 0:
            continue
        overlap = (masks[i] & masks[j]).sum()
        if overlap == 0:
            continue
        smaller_area = min(area_i, area_j)
        frac = overlap / smaller_area
        total_pairs += 1
        if frac > 0.3:
            roi_i = cells_in_crop[i]['roi_idx']
            roi_j = cells_in_crop[j]['roi_idx']
            print(f"  ROI {roi_i} <-> ROI {roi_j}: "
                  f"overlap={overlap}px, smaller_area={smaller_area}px, "
                  f"frac={frac:.2f}")
            if frac > 0.5:
                high_overlap_pairs += 1

print(f"\nPairs with >30% overlap (of smaller ROI): see above")
print(f"Pairs with >50% overlap (of smaller ROI): {high_overlap_pairs} "
      f"out of {total_pairs} overlapping pairs, "
      f"{len(cells_in_crop)} cells total in crop")

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
crop_img = meanImg[y0:y1, x0:x1]
ax.imshow(crop_img, cmap='gray',
          vmin=np.percentile(crop_img, 1),
          vmax=np.percentile(crop_img, 99))

segments = []
for c in cells_in_crop:
    local_contour = c['contour'].copy()
    local_contour[:, 0] -= y0
    local_contour[:, 1] -= x0
    segments.append(local_contour[:, ::-1])  # (x, y) for plotting

lc = LineCollection(segments, colors='red', linewidths=1.0, alpha=0.9)
ax.add_collection(lc)

# Mark centroids too
for c in cells_in_crop:
    y, x = c['centroid']
    ax.plot(x - x0, y - y0, 'b+', markersize=4)

ax.set_title(f"Session {SESSION_ID}, Plane {PLANE} — "
              f"crop [{y0}:{y1}, {x0}:{x1}] — {len(cells_in_crop)} cells")
ax.axis('off')

out_dir = os.path.join(run_dir, "contours")
out_path = os.path.join(out_dir, f"crop_plane{PLANE}_y{cy}_x{cx}.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved: {out_path}")