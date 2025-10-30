# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:46:23 2025

Scan YOLO-OBB labels (class x1 y1 x2 y2 x3 y3 x4 y4, normalized),
estimate object sizes in pixels per image, and recommend tile sizes.

- Computes per-object: width_px, height_px (from quad sides), area_px, "diameter" = sqrt(area)
- Aggregates stats (min/median/mean/p90/p95) per split and overall
- For candidate tile sizes, estimates how big objects become on the model input (e.g., 640)
  and reports % of objects below {8, 12, 16} px for various size metrics.
- Recommends tile sizes by simple rules (see RECOMMENDATION RULES below).

@author: amoslemi
"""

import os
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import cv2
import numpy as np
import pandas as pd

# ------------------- CONFIG -------------------
IMAGE_DIRS = [
    "datasets/GeoMap/images/train",
    "datasets/GeoMap/images/val",
]
LABEL_DIRS = [
    "datasets/GeoMap/labels/train",
    "datasets/GeoMap/labels/val",
]
IMG_EXTS = (".jpg", ".jpeg", ".png")

CANDIDATE_TILE_SIZES = [128, 416]
MODEL_INPUT_PER_TILE = {128: 128, 416: 416}

# Thresholds for "small on input" warning
PIX_THRESHOLDS = [8, 12, 16]

OUT_CSV = "symbol_size_stats.csv"
# ---------------------------------------------


@dataclass
class ObjSize:
    stem: str
    split: str
    cls: int
    w_px: float
    h_px: float
    area_px: float
    diam_px: float  # sqrt(area)

def _read_labels_safe(lbl_path: str) -> pd.DataFrame | None:
    """
    Safe reader for YOLO-OBB labels (class x1 y1 x2 y2 x3 y3 x4 y4), normalized.
    Returns a DataFrame with 9 columns or None if the file is missing/empty/unreadable.
    - Skips blank/invalid lines.
    - Clips values to [0, 1] and drops rows with NaNs.
    """
    if not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        return None  # treat as 'no labels' for this image

    try:
        df = pd.read_csv(
            lbl_path,
            sep=r"\s+",
            header=None,
            engine="python",
            comment="#",
            on_bad_lines="skip"
        )
    except Exception:
        return None

    if df.shape[1] < 9:
        return None

    df = df.iloc[:, :9].copy()
    # coerce to numeric, drop bad rows
    for c in range(9):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None

    # Clip normalized coords into [0,1] to guard minor noise
    for c in range(1, 9):
        df[c] = df[c].clip(0.0, 1.0)

    return df


def _load_image_size(path: str) -> Tuple[int, int]:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Cannot read image: {path}")
    h, w = im.shape[:2]
    return w, h


def _quad_side_lengths(px: np.ndarray) -> Tuple[float, float, float, float]:
    """
    px: (4,2) array of points in pixel coords ordered as (x1,y1,x2,y2,x3,y3,x4,y4)
    Returns lengths of sides: (p1-p2, p2-p3, p3-p4, p4-p1)
    """
    d01 = np.linalg.norm(px[0] - px[1])
    d12 = np.linalg.norm(px[1] - px[2])
    d23 = np.linalg.norm(px[2] - px[3])
    d30 = np.linalg.norm(px[3] - px[0])
    return d01, d12, d23, d30


def _quad_width_height(px: np.ndarray) -> Tuple[float, float]:
    """
    Estimate width/height of oriented bbox from quad by averaging opposite sides.
    """
    d01, d12, d23, d30 = _quad_side_lengths(px)
    w = 0.5 * (d01 + d23)
    h = 0.5 * (d12 + d30)
    # enforce w >= h for consistency (optional)
    # w, h = max(w, h), min(w, h)
    return float(w), float(h)


def _poly_area(px: np.ndarray) -> float:
    """Shoelace area for quad."""
    x = px[:, 0]; y = px[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def collect_sizes() -> List[ObjSize]:
    out: List[ObjSize] = []
    for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
        split = "train" if "train" in img_dir else "val" if "val" in img_dir else os.path.basename(img_dir)
        img_paths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir) if fn.lower().endswith(IMG_EXTS)]
        stems = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}

        for stem, img_path in stems.items():
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            # read image shape safely
            try:
                w_img, h_img = _load_image_size(img_path)
            except RuntimeError:
                continue

            # read labels safely (skip if missing/empty/invalid)
            df = _read_labels_safe(lbl_path)
            if df is None:
                # no valid labels for this image → skip silently
                continue

            # expect: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0..1)
            for _, row in df.iterrows():
                cls = int(row.iloc[0])
                xs = row.iloc[1::2].to_numpy(dtype=np.float32) * w_img
                ys = row.iloc[2::2].to_numpy(dtype=np.float32) * h_img
                quad = np.stack([xs, ys], axis=1)  # (4,2)
                w_px, h_px = _quad_width_height(quad)
                area = _poly_area(quad)
                diam = math.sqrt(max(area, 0.0))
                out.append(ObjSize(stem, split, cls, w_px, h_px, area, diam))

    if not out:
        raise RuntimeError(
            "No objects found. Either labels are missing/empty or LABEL_DIRS are wrong. "
            "Ensure format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."
        )
    return out


def summarize(sizes: List[ObjSize]) -> Dict[str, Dict[str, float]]:
    arr_w = np.array([o.w_px for o in sizes])
    arr_h = np.array([o.h_px for o in sizes])
    arr_d = np.array([o.diam_px for o in sizes])

    def stats(x):
        return {
            "count": int(x.size),
            "min": float(np.min(x)),
            "p10": float(np.percentile(x, 10)),
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)),
        }

    return {"width_px": stats(arr_w), "height_px": stats(arr_h), "diam_px": stats(arr_d)}


def eval_tile_sizes(sizes: List[ObjSize],
                    candidates: List[int],
                    model_input,  # can be int or dict[int,int]
                    pix_thresholds: List[int]) -> pd.DataFrame:
    """
    For each tile size T:
      - If model_input is a dict, use s = model_input[T] / T
      - Else (scalar), use s = model_input / T
    Then report % of objects whose {min_side, diameter} after scaling fall below thresholds.
    """
    rows = []
    min_side = np.array([min(o.w_px, o.h_px) for o in sizes], dtype=float)
    diam     = np.array([o.diam_px for o in sizes], dtype=float)

    def _scale_for(T: int) -> float:
        if isinstance(model_input, dict):
            if T not in model_input:
                raise KeyError(f"model_input missing entry for tile={T}")
            return float(model_input[T]) / float(T)
        else:
            return float(model_input) / float(T)

    for T in candidates:
        s = _scale_for(T)               # in your setup: 128->1.0, 416->1.0
        ms = min_side * s               # size on the model input
        ds = diam * s

        row = {
            "tile": T,
            "scale_to_input": round(s, 3),
            "p10_min_side_on_input": float(np.percentile(ms, 10)),
            "median_min_side_on_input": float(np.median(ms)),
            "p10_diam_on_input": float(np.percentile(ds, 10)),
            "median_diam_on_input": float(np.median(ds)),
        }
        for thr in pix_thresholds:
            row[f"%min_side<{thr}px"] = float(100.0 * np.mean(ms < thr))
            row[f"%diam<{thr}px"]     = float(100.0 * np.mean(ds < thr))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("tile").reset_index(drop=True)
    return df

def recommend(df: pd.DataFrame) -> List[int]:
    """
    RECOMMENDATION RULES (conservative, pick smallest T that passes):
      - p10_min_side_on_input >= 12 px
      - p10_diam_on_input     >= 12 px
      - median_min_side_on_input >= 16 px
    """
    good = df[
        (df["p10_min_side_on_input"] >= 12.0) &
        (df["p10_diam_on_input"] >= 12.0) &
        (df["median_min_side_on_input"] >= 16.0)
    ]["tile"].tolist()
    return good


def main():
    sizes = collect_sizes()
    print(f"[INFO] Collected {len(sizes)} objects from {len(IMAGE_DIRS)} splits.")

    # overall stats
    overall = summarize(sizes)
    print("\n=== Overall object size stats (pixels) ===")
    for k, v in overall.items():
        print(f"{k}: {v}")

    # split-wise stats
    by_split: Dict[str, List[ObjSize]] = {}
    for o in sizes:
        by_split.setdefault(o.split, []).append(o)
    for split, arr in by_split.items():
        stats = summarize(arr)
        print(f"\n=== {split} stats (pixels) ===")
        for k, v in stats.items():
            print(f"{k}: {v}")

    # evaluate tile sizes
    df = eval_tile_sizes(sizes, CANDIDATE_TILE_SIZES, MODEL_INPUT_PER_TILE, PIX_THRESHOLDS)
    print("\n=== Tile-size quality @ native per-tile inputs (scale≈1) ===")
    print(df.to_string(index=False))

    # simple recommendations
    good_tiles = recommend(df)
    if good_tiles:
        print("\n[RECOMMENDATIONS]")
        print(f"- Suitable tiles (smallest first): {sorted(good_tiles)}")
        print("  Rationale: p10(min_side) ≥ 12 px, p10(diam) ≥ 12 px, median(min_side) ≥ 16 px on the model input.")
        print("  If multiple pass, prefer the smallest to save VRAM/time.")
    else:
        print("\n[RECOMMENDATIONS]")
        print("- None of the candidates meet conservative thresholds.")
        print("  Consider increasing tile size or training with larger input size.")

    # CSV dump for later analysis
    with open(OUT_CSV, "w", newline="") as f:
        df.to_csv(f, index=False)
    print(f"\n[Saved] {OUT_CSV}")

    # Extra tip for overlap (based on size stats)
    # Use 0.5 * p95(max_side) as a starting overlap to avoid border cut-offs.
    max_side = np.array([max(o.w_px, o.h_px) for o in sizes])
    p95_max = float(np.percentile(max_side, 95))
    print("\n=== Overlap heuristic ===")
    print(f"p95(max_side) ≈ {p95_max:.1f} px on the map.")
    print("Suggested overlap per tile: ~0.5 × p95(max_side). Round to nearest multiple of 16 or 32.")


if __name__ == "__main__":
    main()
