#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:52:07 2025

@author: abolfazl
"""

import os
import cv2
import pandas as pd
import random
import torch
from ultralytics import YOLO
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import json

CHANNELS = 4               # set to 3, 4, or 6
APPLY_FILTERED_RGB = True

# Configuration
need_cropping = False 
need_augmentation = False
Dual_GPU = True
TILE_SIZE = 416
overlap = 100
EPOCHS = 150
BATCH_SIZE = 16
WORKERS = 2
CACHE = False       
RECT = False  
object_boundary_threshold = 0.1  # Minimum fraction of the bounding box that must remain in the crop
class_balance_threshold = 800  # Minimum number of samples per class for balance
augmentation_repeats = 2  # Number of times to augment underrepresented classes

R_TARGET = 4 

# === Filtering ===
USM_RADIUS = 7.0           
USM_WEIGHT = 0.6            
NLM_H = 3                    
NLM_T = 7                   
NLM_S = 21  

if Dual_GPU:     
    DEVICE = "0,1" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "0" if torch.cuda.is_available() else "cpu"                 

def enumerate_and_save_nonempty_tiles(image_dir, label_dir, output_image_dir, output_label_dir,
                                      out_list_txt, tile_size=128, overlap=50,
                                      rng_seed=42, split_name="train",
                                      empty_meta_path="datasets/GeoMap/_empty_meta_train.json"):
    """
    PASS-1 for TRAIN:
      - Enumerate ALL tiles, but SAVE ONLY non-empty (after boundary filter).
      - Collect metadata for empty tiles (to save later according to R_TARGET).
      - Write out_list_txt with saved positive tiles (only).
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    stride = tile_size - overlap
    assert stride > 0, "overlap must be < tile_size"

    def _cov_frac(row, x, y, ts):
        xs = [row["x1"], row["x2"], row["x3"], row["x4"]]
        ys = [row["y1"], row["y2"], row["y3"], row["y4"]]
        bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
        tbx1, tby1, tbx2, tby2 = x, y, x+ts, y+ts
        ax = max(0, min(bx2, tbx2) - max(bx1, tbx1))
        ay = max(0, min(by2, tby2) - max(by1, tby1))
        inter = ax * ay
        area = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / area

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    new_paths, empty_meta = [], []
    P_total, E_total = 0, 0

    for image_file in image_files:
        ip = os.path.join(image_dir, image_file)
        img = cv2.imread(ip)
        if img is None:
            print(f"[WARN] cannot read: {image_file}")
            continue
        H, W = img.shape[:2]

        lbl_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        labels = read_labels_or_empty(lbl_path, img_w=W, img_h=H)  # به پیکسل

        pos_saved_img = 0
        empties_enum_img = 0
        tile_id = 0

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                if y + tile_size > H or x + tile_size > W:
                    continue

                cand = labels[
                    ((labels["x1"] + labels["x4"]) / 2 >= x) & ((labels["x1"] + labels["x4"]) / 2 < x + tile_size) &
                    ((labels["y1"] + labels["y4"]) / 2 >= y) & ((labels["y1"] + labels["y4"]) / 2 < y + tile_size)
                ].copy()

                if len(cand) > 0:
                    cov = cand.apply(lambda r: _cov_frac(r, x, y, tile_size), axis=1)
                    cand = cand[cov >= object_boundary_threshold].copy()

                if len(cand) > 0:
                    cand[["x1","x2","x3","x4"]] -= x
                    cand[["y1","y2","y3","y4"]] -= y
                    cand[["x1","x2","x3","x4"]] = cand[["x1","x2","x3","x4"]].clip(0, tile_size)
                    cand[["y1","y2","y3","y4"]] = cand[["y1","y2","y3","y4"]].clip(0, tile_size)
                    cand[["x1","x2","x3","x4"]] /= tile_size
                    cand[["y1","y2","y3","y4"]] /= tile_size

                    crop = img[y:y+tile_size, x:x+tile_size]
                    tile_img = f"{os.path.splitext(image_file)[0]}_tile_{tile_id}.jpg"
                    tile_lbl = f"{os.path.splitext(image_file)[0]}_tile_{tile_id}.txt"
                    op_img = os.path.join(output_image_dir, tile_img)
                    op_lbl = os.path.join(output_label_dir, tile_lbl)
                    cv2.imwrite(op_img, crop)
                    cand.to_csv(op_lbl, sep=" ", header=False, index=False)
                    new_paths.append(op_img)
                    P_total += 1
                    pos_saved_img += 1
                else:
                    empty_meta.append({
                        "image_file": image_file,
                        "tile_id": int(tile_id),
                        "x": int(x), "y": int(y),
                        "tile_size": int(tile_size)
                    })
                    E_total += 1
                    empties_enum_img += 1

                tile_id += 1

        print(f"[TILED] {image_file} -> tiles: {pos_saved_img + empties_enum_img} "
              f"(positives saved: {pos_saved_img}, empties enumerated: {empties_enum_img})")

    update_txt_file(out_list_txt, new_paths)

    with open(empty_meta_path, "w") as f:
        json.dump({
            "image_dir": image_dir,
            "output_image_dir": output_image_dir,
            "output_label_dir": output_label_dir,
            "empty": empty_meta
        }, f)

    print(f"[{split_name}] PASS-1 done. Positives saved: {P_total:,} | Empty enumerated: {E_total:,}")
    return {"P_total": P_total, "E_total": E_total, "empty_meta_path": empty_meta_path}


def count_positives_from_label_dir(label_dir: str) -> int:
    """
    Count how many cropped tiles are positive based on label files in label_dir.
    A non-empty .txt (with any non-blank line) counts as positive.
    """
    cnt = 0
    for fn in os.listdir(label_dir):
        if not fn.endswith(".txt"):
            continue
        p = os.path.join(label_dir, fn)
        try:
            if os.path.getsize(p) > 0:
                with open(p, "r") as f:
                    if any(line.strip() for line in f):
                        cnt += 1
        except Exception:
            pass
    return cnt

def save_selected_empty_tiles(empty_meta_path: str,
                              keep_fraction: float,
                              out_list_txt: str,
                              rng_seed: int = 42):
    """
    Keep a fraction of previously enumerated empty tiles.
    Save images + empty label files, and append their paths to out_list_txt.
    """
    assert 0.0 <= keep_fraction <= 1.0
    with open(empty_meta_path, "r") as f:
        meta = json.load(f)

    image_dir = meta["image_dir"]
    out_img_dir = meta["output_image_dir"]
    out_lbl_dir = meta["output_label_dir"]
    empties = meta["empty"]

    if len(empties) == 0:
        print("[INFO] No empty tiles to save.")
        return {"E_kept": 0}

    k = int(round(keep_fraction * len(empties)))
    rng = np.random.RandomState(rng_seed)
    idx = np.arange(len(empties))
    rng.shuffle(idx)
    idx = idx[:k]
    chosen = [empties[i] for i in idx]

    # cache big images
    cache = {}

    kept_paths = []
    for rec in chosen:
        base = rec["image_file"]
        if base not in cache:
            ip = os.path.join(image_dir, base)
            cache[base] = cv2.imread(ip)
            if cache[base] is None:
                print(f"[WARN] cannot read: {base}")
                continue
        img = cache[base]
        x, y, ts = rec["x"], rec["y"], rec["tile_size"]
        crop = img[y:y+ts, x:x+ts]

        tile_img_name = f"{os.path.splitext(base)[0]}_tile_{rec['tile_id']}.jpg"
        tile_lbl_name = f"{os.path.splitext(base)[0]}_tile_{rec['tile_id']}.txt"
        op_img = os.path.join(out_img_dir, tile_img_name)
        op_lbl = os.path.join(out_lbl_dir, tile_lbl_name)

        cv2.imwrite(op_img, crop)
        open(op_lbl, "w").close()  # empty label

        kept_paths.append(op_img)

    # append to list
    with open(out_list_txt, "a") as f:
        for p in kept_paths:
            f.write(p + "\n")

    print(f"[TRAIN] Empty kept: {len(kept_paths):,} of {len(empties):,} (fraction={keep_fraction:.3f})")
    return {"E_kept": len(kept_paths), "E_total": len(empties)}

def read_labels_or_empty(label_path: str, img_w: int, img_h: int) -> pd.DataFrame:
    """
    Safe label loader: missing/empty/corrupt -> empty DataFrame with expected columns.
    Denormalizes 0..1 coords to pixel coords using (img_w, img_h).
    """
    cols = ["class","x1","y1","x2","y2","x3","y3","x4","y4"]

    # Missing or zero-byte file -> empty DF
    if (not os.path.exists(label_path)) or (os.path.getsize(label_path) == 0):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(
            label_path,
            sep=r"\s+",
            header=None,
            engine="python",
            comment="#",
            on_bad_lines="skip"
        )
    except Exception:
        return pd.DataFrame(columns=cols)

    if df.shape[1] < 9:
        return pd.DataFrame(columns=cols)

    df = df.iloc[:, :9].copy()
    df.columns = cols
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    # denormalize to pixels
    df[["x1","x2","x3","x4"]] *= float(img_w)
    df[["y1","y2","y3","y4"]] *= float(img_h)
    return df

def update_txt_file(txt_file, new_paths):
    """
    Update the .txt file with new paths of cropped or augmented images.
    """
    with open(txt_file, "w") as f:
        for path in new_paths:
            f.write(f"{path}\n")

def save_tiff_multipage_from_chw(chw: np.ndarray, out_path: str):
    """
    Save (C, H, W) uint8 as a multi-page .tiff using OpenCV's imwritemulti.
    Each channel is written as a separate page.
    """
    if not hasattr(cv2, "imwritemulti"):
        raise RuntimeError("Your OpenCV build lacks 'imwritemulti'. Install opencv-python with TIFF support.")
    assert chw.ndim == 3 and chw.shape[0] in (4, 6), f"Expected (4,H,W) or (6,H,W), got {chw.shape}"
    pages = [np.ascontiguousarray(chw[c].astype(np.uint8, copy=False)) for c in range(chw.shape[0])]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imwritemulti(str(out_path), pages)
    if not ok:
        raise RuntimeError(f"cv2.imwritemulti failed for: {out_path}")

        
def convert_to_grayscale(image):
    """
    Convert an image to grayscale and ensure it has 3 channels.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  
    return gray_image

def crop_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir,
                           txt_file, cropped_txt_file, tile_size=512, overlap=0,
                           keep_empty_fraction=None, rng_seed=42, split_name="train",
                           boundary_threshold=None):
    """
      Enumerate ALL tiles in memory.
      Apply 'boundary_threshold' to drop near-border tiny boxes.
      Keep all non-empty + a fraction of empty tiles (global).
      If keep_empty_fraction is None or -1, auto-compute via R_TARGET.
    """
    if boundary_threshold is None:
        boundary_threshold = object_boundary_threshold

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    all_tiles = []
    stride = tile_size - overlap
    assert stride > 0, "overlap must be < tile_size"

    def _cov_frac(row, x, y, ts):
        xs = [row["x1"], row["x2"], row["x3"], row["x4"]]
        ys = [row["y1"], row["y2"], row["y3"], row["y4"]]
        bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
        tbx1, tby1, tbx2, tby2 = x, y, x+ts, y+ts
        ax = max(0, min(bx2, tbx2) - max(bx1, tbx1))
        ay = max(0, min(by2, tby2) - max(by1, tby1))
        inter = ax * ay
        area = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / area

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] cannot read image: {image_file}")
            continue
        h, w = image.shape[:2]

        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        labels = read_labels_or_empty(label_path, img_w=w, img_h=h)  # به پیکسل

        tile_id = 0
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                if y + tile_size > h or x + tile_size > w:
                    continue

                cand = labels[
                    ((labels["x1"] + labels["x4"]) / 2 >= x) & ((labels["x1"] + labels["x4"]) / 2 < x + tile_size) &
                    ((labels["y1"] + labels["y4"]) / 2 >= y) & ((labels["y1"] + labels["y4"]) / 2 < y + tile_size)
                ].copy()

                if len(cand) > 0:
                    cov = cand.apply(lambda r: _cov_frac(r, x, y, tile_size), axis=1)
                    cand = cand[cov >= boundary_threshold].copy()

                # shift+normalize ONLY if still non-empty
                if len(cand) > 0:
                    cand[["x1","x2","x3","x4"]] -= x
                    cand[["y1","y2","y3","y4"]] -= y
                    cand[["x1","x2","x3","x4"]] = cand[["x1","x2","x3","x4"]].clip(0, tile_size)
                    cand[["y1","y2","y3","y4"]] = cand[["y1","y2","y3","y4"]].clip(0, tile_size)
                    cand[["x1","x2","x3","x4"]] /= tile_size
                    cand[["y1","y2","y3","y4"]] /= tile_size

                all_tiles.append({
                    "image_file": image_file,
                    "tile_id": tile_id,
                    "x": x, "y": y,
                    "is_empty": len(cand) == 0,
                    "tile_labels": cand
                })
                tile_id += 1

        print(f"[ENUM] {split_name}:{image_file} -> tiles: {tile_id}")

    total_tiles = len(all_tiles)
    total_empty = sum(1 for t in all_tiles if t["is_empty"])
    total_nonempty = total_tiles - total_empty

    if keep_empty_fraction is None or keep_empty_fraction == -1:
        if total_empty > 0:
            keep_empty_fraction = min(1.0, (R_TARGET * total_nonempty) / total_empty)
        else:
            keep_empty_fraction = 0.0

    print(f"\n[{split_name.upper()}] SUMMARY BEFORE EMPTY REMOVAL:")
    print(f"  Total tiles:        {total_tiles:,}")
    print(f"  Non-empty tiles:    {total_nonempty:,}")
    print(f"  Empty tiles:        {total_empty:,}")
    print(f"  -> keep_empty_fraction = {keep_empty_fraction:.3f} (auto={keep_empty_fraction if keep_empty_fraction is not None else 'n/a'})\n")

    empty_idxs = [i for i,t in enumerate(all_tiles) if t["is_empty"]]
    nonempty_idxs = [i for i,t in enumerate(all_tiles) if not t["is_empty"]]

    rng = np.random.RandomState(rng_seed)
    k = int(round(keep_empty_fraction * len(empty_idxs))) if len(empty_idxs) > 0 else 0
    if 0 <= k < len(empty_idxs):
        rng.shuffle(empty_idxs)
        empty_idxs = empty_idxs[:k]

    keep_set = set(nonempty_idxs + empty_idxs)

    new_paths, _cache_img = [], {}
    for i, t in enumerate(all_tiles):
        if i not in keep_set:
            continue

        base = t["image_file"]
        if base not in _cache_img:
            ip = os.path.join(image_dir, base)
            _cache_img[base] = cv2.imread(ip)
            if _cache_img[base] is None:
                print(f"[WARN] cannot read (late): {base}")
                continue

        img_big = _cache_img[base]
        crop = img_big[t["y"]:t["y"] + tile_size, t["x"]:t["x"] + tile_size]

        tile_img_name = f"{os.path.splitext(base)[0]}_tile_{t['tile_id']}.jpg"
        tile_lbl_name = f"{os.path.splitext(base)[0]}_tile_{t['tile_id']}.txt"
        out_img_path = os.path.join(output_image_dir, tile_img_name)
        out_lbl_path = os.path.join(output_label_dir, tile_lbl_name)

        cv2.imwrite(out_img_path, crop)
        if t["is_empty"]:
            open(out_lbl_path, "w").close()
        else:
            t["tile_labels"].to_csv(out_lbl_path, sep=" ", header=False, index=False)

        new_paths.append(out_img_path)

    update_txt_file(cropped_txt_file, new_paths)

    print(f"[{split_name}] saved tiles: {len(new_paths)} | "
          f"non-empty kept: {len(nonempty_idxs)} | empty kept: {len(empty_idxs)} "
          f"(keep_empty_fraction={keep_empty_fraction:.3f})")


def elastic_transform(image, alpha=None, sigma=None):
    """
    Apply elastic transformation while ensuring it doesn't break OpenCV's remap function.
    """
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    if alpha is None:
        alpha = min(shape) * 0.03
    if sigma is None:
        sigma = alpha * 0.1
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
    indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
    return cv2.remap(image, indices_x, indices_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_single_class_augmentation(image, labels, target_class):
    """
    Apply augmentations to an image and labels, ensuring labels remain valid for YOLO format.
    """
    height, width = image.shape[:2]
    aug_results = []

    x_cols = [1, 3, 5, 7]
    y_cols = [2, 4, 6, 8]

    def copy_labels():
        return labels.copy()

    def remove_duplicate_labels(df):
        # Round all numeric columns except class to avoid floating-point noise
        df_rounded = df.copy()
        df_rounded.iloc[:, 1:] = df_rounded.iloc[:, 1:].round(4)
        return df_rounded.drop_duplicates()

    # 1. Scaling
    scaled_img = cv2.resize(image, (int(width * 1.2), int(height * 1.2)))
    scaled_labels = copy_labels()
    scaled_labels[x_cols] *= width
    scaled_labels[y_cols] *= height
    scaled_labels[x_cols] *= 1.2
    scaled_labels[y_cols] *= 1.2
    new_width, new_height = scaled_img.shape[1], scaled_img.shape[0]
    scaled_labels[x_cols] /= new_width
    scaled_labels[y_cols] /= new_height
    scaled_labels.iloc[:, 1:] = scaled_labels.iloc[:, 1:].clip(0, 1)
    scaled_labels = remove_duplicate_labels(scaled_labels)
    aug_results.append(('scale', scaled_img, scaled_labels))

    # 2. Shifting
    shift_x = random.randint(-30, 30)
    shift_y = random.randint(-30, 30)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(image, M, (width, height))
    shifted_labels = copy_labels()
    shifted_labels[x_cols] *= width
    shifted_labels[y_cols] *= height
    shifted_labels[x_cols] += shift_x
    shifted_labels[y_cols] += shift_y
    shifted_labels[x_cols] /= width
    shifted_labels[y_cols] /= height
    shifted_labels.iloc[:, 1:] = shifted_labels.iloc[:, 1:].clip(0, 1)
    shifted_labels = remove_duplicate_labels(shifted_labels)
    aug_results.append(('shift', shifted_img, shifted_labels))

    # 3. HSV
    hsv_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv_img[:, :, 1] *= random.uniform(0.6, 1.4)
    hsv_img[:, :, 2] *= random.uniform(0.6, 1.4)
    hsv_img = np.clip(hsv_img, 0, 255).astype(np.uint8)
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    hsv_labels = copy_labels()
    hsv_labels.iloc[:, 1:] = hsv_labels.iloc[:, 1:].clip(0, 1)
    hsv_labels = remove_duplicate_labels(hsv_labels)
    aug_results.append(('hsv', hsv_img, hsv_labels))

    # 4. Elastic
    # elastic_img = elastic_transform(image.copy())
    # elastic_labels = copy_labels()
    # elastic_labels.iloc[:, 1::2] /= width
    # elastic_labels.iloc[:, 2::2] /= height
    # elastic_labels.iloc[:, 1:] = elastic_labels.iloc[:, 1:].clip(0, 1)
    # aug_results.append(('elastic', elastic_img, elastic_labels))

    return aug_results


def update_balanced_txt_file(txt_file, new_paths):
    """
    Append new paths of augmented images to the .txt file.
    """
    with open(txt_file, "a") as f:
        for path in new_paths:
            f.write(f"{path}\n")

def balance_classes(image_dir, label_dir, txt_file, class_balance_threshold=100, augmentation_repeats=5):
    """
    Balance classes by oversampling underrepresented classes with augmentations,
    and update the txt file with new image paths.
    """

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    class_counts = {}

    for label_file in label_files:
        labels = pd.read_csv(os.path.join(label_dir, label_file), sep=" ", header=None)
        for class_id in labels[0]:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print(f"Initial class distribution: {class_counts}")

    new_image_paths = []

    counter = 0
    for class_id, count in class_counts.items():
        if count >= class_balance_threshold:
            continue

        print(f"Balancing class {class_id} (current count: {count})")
        images_with_class = [lf for lf in label_files if class_id in pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)[0].values]
        
        for _ in range(augmentation_repeats):
            for label_file in images_with_class:
                image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
                image = cv2.imread(image_path)
                if image is None:
                    continue
                labels = pd.read_csv(os.path.join(label_dir, label_file), sep=" ", header=None)
                augmented = apply_single_class_augmentation(image, labels, class_id)
                for aug_type, aug_img, aug_lbls in augmented:
                    unique_id = counter
                    aug_img_filename = f"{os.path.splitext(label_file)[0]}_aug_{aug_type}_{unique_id}.jpg"
                    aug_img_path = os.path.join(image_dir, aug_img_filename)
                    cv2.imwrite(aug_img_path, aug_img)

                    aug_lbl_filename = f"{os.path.splitext(label_file)[0]}_aug_{aug_type}_{unique_id}.txt"
                    aug_lbl_path = os.path.join(label_dir, aug_lbl_filename)
                    aug_lbls.to_csv(aug_lbl_path, sep=" ", header=False, index=False)

                    new_image_paths.append(aug_img_path)
                    counter += 1

    update_balanced_txt_file(txt_file, new_image_paths)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    class_counts = {}

    for label_file in label_files:
        labels = pd.read_csv(os.path.join(label_dir, label_file), sep=" ", header=None)
        for class_id in labels[0]:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    print(f"Balanced class distribution: {class_counts}")

def unsharp_L_on_BGR(bgr: np.ndarray, radius: float = USM_RADIUS, weight: float = USM_WEIGHT) -> np.ndarray:
    """ImageJ-like Unsharp on L channel (LAB), returns BGR uint8."""
    lab32 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab32)
    Lb = cv2.GaussianBlur(L, (0, 0), sigmaX=radius, sigmaY=radius, borderType=cv2.BORDER_REFLECT_101)
    Ls = np.clip(L + weight * (L - Lb), 0, 255)
    lab_s = cv2.merge([Ls, A, B]).astype(np.uint8)
    return cv2.cvtColor(lab_s, cv2.COLOR_LAB2BGR)

def nlm_rgb_to_rgb(bgr: np.ndarray, h: int = NLM_H, t: int = NLM_T, s: int = NLM_S) -> np.ndarray:
    """NLM per-channel in RGB space, returns RGB uint8."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = cv2.fastNlMeansDenoising(rgb[..., c], None, h=h, templateWindowSize=t, searchWindowSize=s)
    return out

def filter_rgb_pipeline(bgr: np.ndarray) -> np.ndarray:
    """
    Unsharp on L (LAB) -> NLM on RGB, then return a standard 3ch BGR for saving.
    """
    bgr_usm  = unsharp_L_on_BGR(bgr, radius=USM_RADIUS, weight=USM_WEIGHT)
    rgb_proc = nlm_rgb_to_rgb(bgr_usm, h=NLM_H, t=NLM_T, s=NLM_S)    # RGB uint8
    return cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2BGR)                 # BGR for cv2.imwrite

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def mirror_labels_by_stem(src_lbl_dir: str, dst_lbl_dir: str, stems: list):
    ensure_dir(dst_lbl_dir)
    copied, missing = 0, 0
    for s in stems:
        src = os.path.join(src_lbl_dir, f"{s}.txt")
        dst = os.path.join(dst_lbl_dir, f"{s}.txt")
        if os.path.exists(src):
            cv2.imwrite  # no-op to avoid lints
            import shutil
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    print(f"[INFO] labels copied={copied}, missing={missing} -> {dst_lbl_dir}")

def list_stems_in_dir(dirpath: str) -> list:
    exts = (".jpg", ".jpeg", ".png")
    stems = []
    for fn in os.listdir(dirpath):
        if fn.lower().endswith(exts):
            stems.append(os.path.splitext(fn)[0])
    return stems

def filter_folder_rgb(src_img_dir: str, src_lbl_dir: str,
                      dst_img_dir: str, dst_lbl_dir: str) -> list:
    """
    Read every image in src_img_dir, apply Unsharp->NLM, save to dst_img_dir.
    Copy corresponding labels from src_lbl_dir to dst_lbl_dir by stem.
    Returns absolute paths of filtered images.
    """
    ensure_dir(dst_img_dir); ensure_dir(dst_lbl_dir)
    out_paths = []
    exts = (".jpg", ".jpeg", ".png")
    for fn in os.listdir(src_img_dir):
        if not fn.lower().endswith(exts):
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {ip}")
            continue
        bgr_f = filter_rgb_pipeline(bgr)
        op = os.path.join(dst_img_dir, fn)  # keep same name
        cv2.imwrite(op, bgr_f)
        out_paths.append(os.path.abspath(op))

    # mirror labels
    stems = [os.path.splitext(os.path.basename(p))[0] for p in out_paths]
    mirror_labels_by_stem(src_lbl_dir, dst_lbl_dir, stems)
    return out_paths

def build_4ch_CHW_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Returns (4, H, W) uint8 = [R, G, B, L_proc]
    where L_proc = NLM( Unsharp(L in LAB) ).
    """
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                       # (H,W,3)
    lab32   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab32)
    Lb      = cv2.GaussianBlur(L, (0, 0), sigmaX=USM_RADIUS, sigmaY=USM_RADIUS, borderType=cv2.BORDER_REFLECT_101)
    Ls      = np.clip(L + USM_WEIGHT * (L - Lb), 0, 255).astype(np.uint8)  # sharpened L
    L_proc  = cv2.fastNlMeansDenoising(Ls, None, h=NLM_H, templateWindowSize=NLM_T, searchWindowSize=NLM_S)
    four_hwc = np.dstack([rgb_raw, L_proc]).astype(np.uint8)             # (H,W,4)
    four_chw = four_hwc.transpose(2, 0, 1)                               # (4,H,W)
    return np.ascontiguousarray(four_chw)

def build_6ch_CHW_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Returns (6, H, W) uint8 = [RGB_raw, RGB_proc]
    where RGB_proc = NLM( Unsharp(L in LAB) applied to BGR then converted to RGB ).
    """
    rgb_raw  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)                      # (H,W,3)
    bgr_usm  = unsharp_L_on_BGR(bgr, radius=USM_RADIUS, weight=USM_WEIGHT)
    rgb_proc = nlm_rgb_to_rgb(bgr_usm, h=NLM_H, t=NLM_T, s=NLM_S)        # (H,W,3)
    six_hwc  = np.dstack([rgb_raw, rgb_proc]).astype(np.uint8)           # (H,W,6)
    six_chw  = six_hwc.transpose(2, 0, 1)                                # (6,H,W)
    return np.ascontiguousarray(six_chw)

IMG_EXTS = (".jpg", ".jpeg", ".png")

def convert_folder_to_4ch_tiff(src_img_dir: str, dst_img_dir: str) -> list:
    os.makedirs(dst_img_dir, exist_ok=True)
    out_paths = []
    for fn in sorted(os.listdir(src_img_dir)):
        if not fn.lower().endswith(IMG_EXTS): 
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {ip}"); continue
        four_chw = build_4ch_CHW_from_bgr(bgr)  # (4,H,W)
        op = os.path.join(dst_img_dir, os.path.splitext(fn)[0] + ".tiff")
        save_tiff_multipage_from_chw(four_chw, op)
        out_paths.append(os.path.abspath(op))
    return out_paths

def convert_folder_to_6ch_tiff(src_img_dir: str, dst_img_dir: str) -> list:
    os.makedirs(dst_img_dir, exist_ok=True)
    out_paths = []
    for fn in sorted(os.listdir(src_img_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {ip}"); continue
        six_chw = build_6ch_CHW_from_bgr(bgr)   # (6,H,W)
        op = os.path.join(dst_img_dir, os.path.splitext(fn)[0] + ".tiff")
        save_tiff_multipage_from_chw(six_chw, op)
        out_paths.append(os.path.abspath(op))
    return out_paths

def convert_folder_to_4ch_tiff_palette(src_img_dir: str, dst_img_dir: str,
                                       centers_rgb: np.ndarray) -> list:
    os.makedirs(dst_img_dir, exist_ok=True)
    out_paths = []
    for fn in sorted(os.listdir(src_img_dir)):
        if not fn.lower().endswith(IMG_EXTS): 
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {ip}"); 
            continue
        four_chw = build_4ch_CHW_from_bgr_palette(bgr, centers_rgb)  # (4,H,W)
        op = os.path.join(dst_img_dir, os.path.splitext(fn)[0] + ".tiff")
        save_tiff_multipage_from_chw(four_chw, op)
        out_paths.append(os.path.abspath(op))
    return out_paths

def convert_folder_to_4ch_tiff_msedge(src_img_dir: str, dst_img_dir: str,
                                      sigmas=(0,1.0,2.0,4.0)) -> list:
    os.makedirs(dst_img_dir, exist_ok=True)
    out_paths = []
    for fn in sorted(os.listdir(src_img_dir)):
        if not fn.lower().endswith(IMG_EXTS): 
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {ip}"); continue
        four_chw = build_4ch_CHW_from_bgr_msedge(bgr, sigmas=sigmas)
        op = os.path.join(dst_img_dir, os.path.splitext(fn)[0] + ".tiff")
        save_tiff_multipage_from_chw(four_chw, op)
        out_paths.append(os.path.abspath(op))
    return out_paths

def convert_folder_to_4ch_tiff_dtedge(src_img_dir: str, dst_img_dir: str,
                                      sigmas=(0,0.8,1.6,3.2), **kwargs) -> list:
    os.makedirs(dst_img_dir, exist_ok=True)
    out_paths = []
    for fn in sorted(os.listdir(src_img_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {ip}"); continue
        four_chw = build_4ch_CHW_from_bgr_dtedge(bgr, sigmas=sigmas, **kwargs)
        op = os.path.join(dst_img_dir, os.path.splitext(fn)[0] + ".tiff")
        save_tiff_multipage_from_chw(four_chw, op)
        out_paths.append(os.path.abspath(op))
    return out_paths

def fit_kmeans_on_folder(src_img_dir: str, n_colors: int = 15, sample_per_image: int = 5000, seed: int = 0):
    rng = np.random.RandomState(seed)
    samples = []
    exts = (".jpg", ".jpeg", ".png")
    for fn in sorted(os.listdir(src_img_dir)):
        if not fn.lower().endswith(exts):
            continue
        ip = os.path.join(src_img_dir, fn)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None: 
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        if rgb.shape[0] > sample_per_image:
            idx = rng.choice(rgb.shape[0], sample_per_image, replace=False)
            rgb = rgb[idx]
        samples.append(rgb)
    if not samples:
        raise RuntimeError(f"No images found in {src_img_dir} for KMeans.")
    X = np.concatenate(samples, axis=0).astype(np.float32)
    km = KMeans(n_clusters=n_colors, random_state=seed, n_init="auto")
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)  

def quantize_rgb_with_centers(bgr: np.ndarray, centers_rgb: np.ndarray) -> np.ndarray:

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W, _ = rgb.shape
    flat = rgb.reshape(-1, 3)  # (N,3)
    dists = ((flat[:, None, :] - centers_rgb[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint8)
    return labels.reshape(H, W)

def build_4ch_CHW_from_bgr_palette(bgr: np.ndarray, centers_rgb: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    idx = quantize_rgb_with_centers(bgr, centers_rgb)  # (H,W) uint8
    four = np.dstack([rgb, idx]).astype(np.uint8)      # (H,W,4)
    return four.transpose(2, 0, 1).copy()              # (4,H,W)


# --- Multi-Scale Edge (msEdge) helpers ---
def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def _structure_tensor_coherence(gray: np.ndarray, win: int = 5, sigma: float = 1.0) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Ixx, Iyy, Ixy = gx*gx, gy*gy, gx*gy
    if sigma > 0:
        Ixx = cv2.GaussianBlur(Ixx, (0,0), sigma)
        Iyy = cv2.GaussianBlur(Iyy, (0,0), sigma)
        Ixy = cv2.GaussianBlur(Ixy, (0,0), sigma)
    else:
        Ixx = cv2.boxFilter(Ixx, -1, (win,win), normalize=True)
        Iyy = cv2.boxFilter(Iyy, -1, (win,win), normalize=True)
        Ixy = cv2.boxFilter(Ixy, -1, (win,win), normalize=True)
    tr  = Ixx + Iyy
    det = Ixx*Iyy - Ixy*Ixy
    tmp = np.sqrt((Ixx - Iyy)**2 + 4.0*(Ixy**2))
    lam1 = (tr + tmp) * 0.5
    lam2 = (tr - tmp) * 0.5
    eps = 1e-6
    coh = (lam1 - lam2) / (lam1 + lam2 + eps)
    return np.clip(coh, 0, 1)

def cme_edge_channel_from_bgr(bgr: np.ndarray,
                              sigmas=(0, 1.0, 2.0, 4.0),
                              coh_win=5, coh_sigma=1.0,
                              hysteresis=(0.1, 0.3)) -> np.uint8:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    acc = None
    for s in sigmas:
        blur = cv2.GaussianBlur(gray, (0,0), s, s, borderType=cv2.BORDER_REFLECT_101) if s>0 else gray
        mag  = _grad_mag(blur)
        acc  = mag if acc is None else np.maximum(acc, mag)
    # 2) coherence
    coh = _structure_tensor_coherence(gray, win=coh_win, sigma=coh_sigma)  # 0..1
    cme = acc * (0.5 + 0.5*coh)   

    lo, hi = np.percentile(cme, [1, 99])
    cme = np.clip((cme - lo) / max(1e-6, (hi - lo)), 0, 1)
    # 4) hysteresis optional 
    tl, th = hysteresis
    strong = (cme >= th).astype(np.uint8)
    weak   = ((cme >= tl) & (cme < th)).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    grown  = cv2.dilate(strong, kernel, iterations=1)
    keep   = np.where((weak==1) & (grown==1), 1, strong)
    cme = np.where(keep==1, cme, 0.0)
    return (cme * 255).astype(np.uint8)

def build_4ch_CHW_from_bgr_msedge(bgr: np.ndarray, sigmas=(0,1.0,2.0,4.0)) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    edge = cme_edge_channel_from_bgr(bgr, sigmas=sigmas)                  # (H,W) uint8
    four = np.dstack([rgb, edge]).astype(np.uint8)                       # (H,W,4)
    return four.transpose(2, 0, 1).copy()                                # (4,H,W)

# --- DT-Edge (Distance Transform of Multi-Scale Edges) ---
def _scharr_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def dt_edge_channel_from_bgr(
    bgr: np.ndarray,
    sigmas=(0, 0.8, 1.6, 3.2),
    bin_method: str = "otsu",  # "percentile" | "otsu"
    p_hi: int = 85, p_lo: int = 65,
    morph_open: int = 2
) -> np.uint8:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # multi-scale gradient
    acc = None
    for s in sigmas:
        blur = cv2.GaussianBlur(gray, (0,0), s, s, borderType=cv2.BORDER_REFLECT_101) if s > 0 else gray
        mag  = _scharr_mag(blur)
        acc  = mag if acc is None else np.maximum(acc, mag)

    # threshold to edge map
    if bin_method == "otsu":
        acc8 = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(acc8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        lo, hi = np.percentile(acc, [p_lo, p_hi])
        edges = (acc >= hi).astype(np.uint8) * 255

    # small morphology clean
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k, iterations=morph_open)

    # distance transform on non-edges
    non_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)[1]
    dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3).astype(np.float32)

    # robust normalize (1..99 percentiles)
    lo, hi = np.percentile(dist, [1, 99])
    dist = np.clip((dist - lo) / max(1e-6, (hi - lo)), 0, 1)
    tau = 3.0
    soft = np.exp(-dist / tau)   
    acc8 = cv2.normalize(acc, None, 0, 1, cv2.NORM_MINMAX)
    soft = 0.7*soft + 0.3*acc8
    soft = np.clip(soft, 0, 1)
    return (soft * 255).astype(np.uint8)

def build_4ch_CHW_from_bgr_dtedge(
    bgr: np.ndarray,
    sigmas=(0, 0.8, 1.6, 3.2),
    **kwargs
) -> np.ndarray:
    """(4,H,W) = [R,G,B, DT-Edge]"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    dt  = dt_edge_channel_from_bgr(bgr, sigmas=sigmas, **kwargs)
    four = np.dstack([rgb, dt]).astype(np.uint8)
    return four.transpose(2, 0, 1).copy()


if __name__ == "__main__":
    
    image_dir = "datasets/GeoMap/images/train"
    label_dir = "datasets/GeoMap/labels/train"
    output_image_dir = f"datasets/GeoMap/cropped{TILE_SIZE}/images/train"
    output_label_dir = f"datasets/GeoMap/cropped{TILE_SIZE}/labels/train"
    txt_file = "datasets/GeoMap/train.txt"
    cropped_txt_file = f"datasets/GeoMap/train_cropped{TILE_SIZE}.txt"

    val_image_dir = "datasets/GeoMap/images/val"
    val_label_dir = "datasets/GeoMap/labels/val"
    val_output_image_dir = f"datasets/GeoMap/cropped{TILE_SIZE}/images/val"
    val_output_label_dir = f"datasets/GeoMap/cropped{TILE_SIZE}/labels/val"
    val_txt_file = "datasets/GeoMap/val.txt"
    val_cropped_txt_file = f"datasets/GeoMap/val_cropped{TILE_SIZE}.txt"
    
    # 4ch and 6ch output roots (parallel to cropped/)
    out_img4_train = "datasets/GeoMap/cropped4/images/train"
    out_lbl4_train = "datasets/GeoMap/cropped4/labels/train"
    out_img4_val   = "datasets/GeoMap/cropped4/images/val"
    out_lbl4_val   = "datasets/GeoMap/cropped4/labels/val"
    train_list_4ch = "datasets/GeoMap/train_cropped_4ch.txt"
    val_list_4ch   = "datasets/GeoMap/val_cropped_4ch.txt"
    
    out_img6_train = "datasets/GeoMap/cropped6/images/train"
    out_lbl6_train = "datasets/GeoMap/cropped6/labels/train"
    out_img6_val   = "datasets/GeoMap/cropped6/images/val"
    out_lbl6_val   = "datasets/GeoMap/cropped6/labels/val"
    train_list_6ch = "datasets/GeoMap/train_cropped_6ch.txt"
    val_list_6ch   = "datasets/GeoMap/val_cropped_6ch.txt"

    f_output_image_dir = "datasets/GeoMap/cropped_filt/images/train"
    f_output_label_dir = "datasets/GeoMap/cropped_filt/labels/train"
    f_val_output_image_dir = "datasets/GeoMap/cropped_filt/images/val"
    f_val_output_label_dir = "datasets/GeoMap/cropped_filt/labels/val"
    
    train_filtered_list = "datasets/GeoMap/train_cropped_filt.txt"
    val_filtered_list   = "datasets/GeoMap/val_cropped_filt.txt"
    
    if need_cropping:
        # -------- TRAIN: PASS-1 (save only positives, enumerate empties) --------
        empty_meta_path = "datasets/GeoMap/_empty_meta_train.json"
        stats1 = enumerate_and_save_nonempty_tiles(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            out_list_txt=cropped_txt_file,
            tile_size=TILE_SIZE,
            overlap=overlap,
            rng_seed=42,
            split_name="train",
            empty_meta_path=empty_meta_path
        )
    
        # Augmentation only on TRAIN
        if need_augmentation:
            balance_classes(
                image_dir=output_image_dir,
                label_dir=output_label_dir,
                txt_file=cropped_txt_file,
                class_balance_threshold=class_balance_threshold,
                augmentation_repeats=augmentation_repeats
            )  # this appends new augmented positives to train list  :contentReference[oaicite:0]{index=0}
    
        # -------- TRAIN: decide keep_fraction automatically from R_TARGET --------
        P_post = count_positives_from_label_dir(output_label_dir)  
        E_total = stats1["E_total"]
        if E_total > 0:
            keep_fraction_auto = min(1.0, (R_TARGET * P_post) / E_total)
        else:
            keep_fraction_auto = 0.0
    
        print(f"[TRAIN] AUTO keep_fraction computed: {keep_fraction_auto:.4f} "
              f"(R_TARGET={R_TARGET}, P_post={P_post:,}, E_total={E_total:,})")
    
        # -------- TRAIN: PASS-2 (save empties according to auto fraction) --------
        save_selected_empty_tiles(
            empty_meta_path=empty_meta_path,
            keep_fraction=keep_fraction_auto,
            out_list_txt=cropped_txt_file,
            rng_seed=42
        )
        
        # Final report for TRAIN after saving empties
        P_final = count_positives_from_label_dir(output_label_dir)
        E_final = sum(1 for fn in os.listdir(output_label_dir) if fn.endswith(".txt")) - P_final
        print(f"[TRAIN] FINAL COUNTS after augmentation + empties:")
        print(f"  Positives: {P_final:,}")
        print(f"  Empties:   {E_final:,}")
    
        # -------- VAL --------
        crop_images_and_labels(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            output_image_dir=val_output_image_dir,
            output_label_dir=val_output_label_dir,
            txt_file=val_txt_file,
            cropped_txt_file=val_cropped_txt_file,
            tile_size=TILE_SIZE,
            overlap=overlap,                   
            keep_empty_fraction=None,         
            rng_seed=42,
            split_name="val",
            boundary_threshold=object_boundary_threshold
        )

    # ===== Build training/val inputs based on CHANNELS =====
    if CHANNELS == 3:
        if APPLY_FILTERED_RGB:
            print("[INFO] Creating filtered RGB copies for 3-channel training...")
            train_filtered_paths = filter_folder_rgb(
                src_img_dir=output_image_dir,
                src_lbl_dir=output_label_dir,
                dst_img_dir=f_output_image_dir,
                dst_lbl_dir=f_output_label_dir
            )
            val_filtered_paths = filter_folder_rgb(
                src_img_dir=val_output_image_dir,
                src_lbl_dir=val_output_label_dir,
                dst_img_dir=f_val_output_image_dir,
                dst_lbl_dir=f_val_output_label_dir
            )
            with open(train_filtered_list, "w") as f:
                for p in train_filtered_paths: f.write(p + "\n")
            with open(val_filtered_list, "w") as f:
                for p in val_filtered_paths: f.write(p + "\n")
            DATA_YAML = "datasets/GeoMap/data_filt.yaml"
        else:
            DATA_YAML = f"datasets/GeoMap/data{TILE_SIZE}.yaml"
    
    elif CHANNELS == 4:
        if APPLY_FILTERED_RGB:
            MS_SIGMAS = (0, 0.6, 1.2, 2.4)
            print("[INFO] Converting TRAIN to 4ch [RGB, DT-Edge] TIFFs...")
            tr_paths = convert_folder_to_4ch_tiff_dtedge(
                output_image_dir, out_img4_train,
                sigmas=MS_SIGMAS, bin_method="percentile", p_hi=90, p_lo=65, morph_open=1
            )
            print("[INFO] Converting VAL to 4ch [RGB, DT-Edge] TIFFs...")
            va_paths = convert_folder_to_4ch_tiff_dtedge(
                val_output_image_dir, out_img4_val,
                sigmas=MS_SIGMAS, bin_method="percentile", p_hi=90, p_lo=65, morph_open=1
            )
    
            tr_stems = [os.path.splitext(os.path.basename(p))[0] for p in tr_paths]
            va_stems = [os.path.splitext(os.path.basename(p))[0] for p in va_paths]
            mirror_labels_by_stem(output_label_dir, out_lbl4_train, tr_stems)
            mirror_labels_by_stem(val_output_label_dir, out_lbl4_val, va_stems)
    
            with open(train_list_4ch, "w") as f:
                for p in tr_paths: f.write(p + "\n")
            with open(val_list_4ch, "w") as f:
                for p in va_paths: f.write(p + "\n")
        
        DATA_YAML = "datasets/GeoMap/data4ch.yaml"  # create yaml that points to cropped4/images/{train,val}
    
    elif CHANNELS == 6:
        if APPLY_FILTERED_RGB:
            print("[INFO] Converting cropped(+aug) to 6-channel multi-page TIFFs...")
            tr_paths = convert_folder_to_6ch_tiff(output_image_dir, out_img6_train)
            va_paths = convert_folder_to_6ch_tiff(val_output_image_dir, out_img6_val)
            tr_stems = [os.path.splitext(os.path.basename(p))[0] for p in tr_paths]
            va_stems = [os.path.splitext(os.path.basename(p))[0] for p in va_paths]
            mirror_labels_by_stem(output_label_dir, out_lbl6_train, tr_stems)
            mirror_labels_by_stem(val_output_label_dir, out_lbl6_val, va_stems)
            with open(train_list_6ch, "w") as f:
                for p in tr_paths: f.write(p + "\n")
            with open(val_list_6ch, "w") as f:
                for p in va_paths: f.write(p + "\n")
        DATA_YAML = "datasets/GeoMap/data6ch.yaml"  # create yaml that points to cropped6/images/{train,val}
    
    else:
        raise ValueError("CHANNELS must be 3, 4, or 6")
    
    model = YOLO("yolo11x-obb.pt")
    
    # # Size 128
    # model.train(
    #     data=DATA_YAML,
    #     epochs=EPOCHS,
    #     imgsz=TILE_SIZE,
    #     batch=BATCH_SIZE,
    #     workers=WORKERS,
    #     cache=CACHE,
    #     rect=RECT,
    #     device=DEVICE,
    #     multi_scale=False,
    #     lr0=0.003,
    #     lrf=0.05,
    #     weight_decay=0.001,
    #     dropout=0.0,
    #     patience=50,
    #     plots=True,              
    #     overlap_mask=False,
    #     # task='obb',
    #     # mosaic=0.0, mixup=0.0, copy_paste=0.0,
    #     # hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    #     # amp=False
    # )
    
    # Size 416
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=TILE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=CACHE,
        rect=RECT,
        device=DEVICE,
        multi_scale=False,
        lr0 = 0.003,  
        lrf = 0.05,      
        weight_decay = 0.001, 
        dropout = 0.0,
        # warmup_epochs = 5.0,
        # warmup_momentum = 0.85,
        # warmup_bias_lr = 0.08,
        patience=50,
        plots = True,
        overlap_mask = False,
    )