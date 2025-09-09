# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 18:29:09 2025

6-Channel YOLO11-OBB training:

@author: amoslemi
"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import shutil
from pathlib import Path
from tqdm import tqdm

# ==============================
# Config
# ==============================
# Tiling / dataset
TILE_SIZE = 128
OVERLAP = 50
channels = 6
NEED_CROPPING = True
NEED_AUGMENTATION = True
NEED_6CHANNEL = True
Dual_GPU = True

# Training
EPOCHS = 150
BATCH_SIZE = 32
WORKERS = 2
CACHE = False       
RECT = False  
if Dual_GPU:     
    DEVICE = "0,1" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "0" if torch.cuda.is_available() else "cpu"

# Balancing
CLASS_BALANCE_THRESHOLD = 400
AUGMENTATION_REPEATS = 2

# Paths (match your structure)
TRAIN_IMG_DIR = "datasets/GeoMap/images/train"
TRAIN_LBL_DIR = "datasets/GeoMap/labels/train"
TRAIN_OUT_IMG_DIR = "datasets/GeoMap/cropped/images/train"
TRAIN_OUT_LBL_DIR = "datasets/GeoMap/cropped/labels/train"
TRAIN_LIST = "datasets/GeoMap/train.txt"
TRAIN_LIST_CROPPED = "datasets/GeoMap/train_cropped.txt"

VAL_IMG_DIR = "datasets/GeoMap/images/val"
VAL_LBL_DIR = "datasets/GeoMap/labels/val"
VAL_OUT_IMG_DIR = "datasets/GeoMap/cropped/images/val"
VAL_OUT_LBL_DIR = "datasets/GeoMap/cropped/labels/val"
VAL_LIST = "datasets/GeoMap/val.txt"
VAL_LIST_CROPPED = "datasets/GeoMap/val_cropped.txt"

TRAIN_OUT_IMG6_DIR = "datasets/GeoMap/cropped6/images/train"
VAL_OUT_IMG6_DIR   = "datasets/GeoMap/cropped6/images/val"

TRAIN_OUT_LBL6_DIR = "datasets/GeoMap/cropped6/labels/train"
VAL_OUT_LBL6_DIR   = "datasets/GeoMap/cropped6/labels/val"

TRAIN_LIST_6CH = "datasets/GeoMap/train_cropped_6ch.txt"
VAL_LIST_6CH   = "datasets/GeoMap/val_cropped_6ch.txt"

DATA_YAML = "datasets/GeoMap/data6ch.yaml"   
PRETRAINED = "yolo11x-obb.pt"              
    
USM_RADIUS = 7.0     
USM_WEIGHT = 1.40 
NLM_H = 7
NLM_T = 7
NLM_S = 21

# ==============================
# Helpers
# ==============================
def write_list(txt_file, paths):
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    with open(txt_file, "w") as f:
        for p in paths:
            f.write(p + "\n")

def append_list(txt_file, paths):
    with open(txt_file, "a") as f:
        for p in paths:
            f.write(p + "\n")

def build_six_from_bgr(bgr: np.ndarray,
                       usm_radius: float = USM_RADIUS,
                       usm_weight: float = USM_WEIGHT,
                       nlm_h: int = NLM_H, nlm_t: int = NLM_T, nlm_s: int = NLM_S) -> np.ndarray:
    """Return HxWx6 uint8 = [RGB_raw, RGB_proc] where RGB_proc = Unsharp(L) -> NLM(RGB)."""
    # RGB raw
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Unsharp (ImageJ-like) on L in LAB
    lab32 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab32)
    Lb = cv2.GaussianBlur(L, (0, 0), sigmaX=usm_radius, sigmaY=usm_radius, borderType=cv2.BORDER_REFLECT_101)
    Ls = np.clip(L + usm_weight * (L - Lb), 0, 255)
    bgr_usm = cv2.cvtColor(np.dstack([Ls, A, B]).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # NLM per RGB channel
    rgb = cv2.cvtColor(bgr_usm, cv2.COLOR_BGR2RGB)
    rgb_proc = np.empty_like(rgb)
    for c in range(3):
        rgb_proc[..., c] = cv2.fastNlMeansDenoising(rgb[..., c], None,
                                                    h=nlm_h, templateWindowSize=nlm_t, searchWindowSize=nlm_s)

    six = np.dstack([rgb_raw, rgb_proc]).astype(np.uint8)  # (H,W,6)
    return np.ascontiguousarray(six)
    
def mirror_labels_for_6ch(image6_paths: list, src_lbl_dir: str, dst_lbl_dir: str):
    """
    Copy label TXT files from cropped/labels/ to cropped6/labels/
    """
    os.makedirs(dst_lbl_dir, exist_ok=True)
    copied, missing = 0, 0
    for ip in image6_paths:
        stem = Path(ip).stem              # e.g., "xxx_tile_123"
        src = os.path.join(src_lbl_dir, f"{stem}.txt")
        dst = os.path.join(dst_lbl_dir, f"{stem}.txt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            print(f"[WARN] Label not found for {stem}: {src}")
    print(f"[INFO] Mirrored labels: {copied}, missing: {missing} → {dst_lbl_dir}")
# ==============================
# Tiling + OBB label adjust 
# ==============================
def crop_images_and_labels(image_dir, label_dir, out_img_dir, out_lbl_dir, list_out,
                           tile_size=512, overlap=0):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    new_paths = []

    step = tile_size - overlap

    for image_file in image_files:
        ipath = os.path.join(image_dir, image_file)
        bgr = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Cannot read image: {image_file}")
            continue

        h, w = bgr.shape[:2]
        label_file = os.path.splitext(image_file)[0] + ".txt"
        lpath = os.path.join(label_dir, label_file)
        if not os.path.exists(lpath):
            print(f"[WARN] Label not found: {label_file}")
            continue

        df = pd.read_csv(lpath, sep=" ", header=None)
        df.columns = ["class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        # denorm to pixels
        for col in ["x1","x2","x3","x4"]:
            df[col] *= w
        for col in ["y1","y2","y3","y4"]:
            df[col] *= h

        tid = 0
        # centroid-based selection (x1,y1 to x4,y4 ordering assumed consistent)
        cx = (df["x1"] + df["x4"]) / 2.0
        cy = (df["y1"] + df["y4"]) / 2.0

        for y in range(0, h, step):
            for x in range(0, w, step):
                crop = bgr[y:y+tile_size, x:x+tile_size]
                if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                    continue

                mask = (cx >= x) & (cx < x + tile_size) & \
                       (cy >= y) & (cy < y + tile_size)
                tl = df[mask].copy()
                if tl.empty:
                    continue

                # shift to tile coords
                for col in ["x1","x2","x3","x4"]:
                    tl[col] -= x
                for col in ["y1","y2","y3","y4"]:
                    tl[col] -= y

                # clip to tile
                for col in ["x1","x2","x3","x4"]:
                    tl[col] = tl[col].clip(0, tile_size)
                for col in ["y1","y2","y3","y4"]:
                    tl[col] = tl[col].clip(0, tile_size)

                # normalize to [0,1]
                for col in ["x1","x2","x3","x4"]:
                    tl[col] /= tile_size
                for col in ["y1","y2","y3","y4"]:
                    tl[col] /= tile_size

                # save
                base = f"{os.path.splitext(image_file)[0]}_tile_{tid}"
                img_out = os.path.join(out_img_dir, base + ".jpg")
                lbl_out = os.path.join(out_lbl_dir, base + ".txt")
                cv2.imwrite(img_out, crop)
                tl.to_csv(lbl_out, sep=" ", header=False, index=False)

                new_paths.append(img_out)
                tid += 1

        print(f"[INFO] Tiled: {image_file}")

    write_list(list_out, new_paths)


# ==============================
# Simple per-class balancing 
# ==============================
def apply_single_class_augmentation(image, labels):
    height, width = image.shape[:2]
    aug_results = []
    x_cols = [1,3,5,7]
    y_cols = [2,4,6,8]

    def dup(df): return df.copy()

    # Scale 1.2x
    scaled = cv2.resize(image, (int(width*1.2), int(height*1.2)))
    sl = dup(labels)
    sl[x_cols] *= width; sl[y_cols] *= height
    sl[x_cols] *= 1.2;   sl[y_cols] *= 1.2
    nw, nh = scaled.shape[1], scaled.shape[0]
    sl[x_cols] /= nw; sl[y_cols] /= nh
    sl.iloc[:,1:] = sl.iloc[:,1:].clip(0,1)
    aug_results.append(("scale", scaled, sl))

    # Shift
    sx = random.randint(-40, 40); sy = random.randint(-40, 40)
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(image, M, (width, height))
    sh = dup(labels)
    sh[x_cols] *= width; sh[y_cols] *= height
    sh[x_cols] += sx;    sh[y_cols] += sy
    sh[x_cols] /= width; sh[y_cols] /= height
    sh.iloc[:,1:] = sh.iloc[:,1:].clip(0,1)
    aug_results.append(("shift", shifted, sh))

    # HSV jitter
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:,:,1] *= random.uniform(0.7, 1.3)
    hsv[:,:,2] *= random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hs = dup(labels)
    hs.iloc[:,1:] = hs.iloc[:,1:].clip(0,1)
    aug_results.append(("hsv", hsv, hs))

    return aug_results

def balance_classes(image_dir, label_dir, list_file, class_balance_threshold=100, augmentation_repeats=5):
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    counts = {}
    for lf in label_files:
        df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
        for cid in df[0]:
            counts[cid] = counts.get(cid, 0) + 1
    print(f"[INFO] Initial class distribution: {counts}")

    new_paths = []
    uid = 0
    for cid, cnt in counts.items():
        if cnt >= class_balance_threshold:
            continue
        print(f"[INFO] Balancing class {cid} (count={cnt})")
        files_with = []
        for lf in label_files:
            df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
            if cid in df[0].values:
                files_with.append(lf)

        for _ in range(augmentation_repeats):
            for lf in files_with:
                ip = os.path.join(image_dir, lf.replace(".txt",".jpg"))
                img = cv2.imread(ip)
                if img is None: continue
                df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
                aug_list = apply_single_class_augmentation(img, df)
                for a_type, a_img, a_lbl in aug_list:
                    base = os.path.splitext(lf)[0]
                    out_img = os.path.join(image_dir, f"{base}_aug_{a_type}_{uid}.jpg")
                    out_lbl = os.path.join(label_dir, f"{base}_aug_{a_type}_{uid}.txt")
                    cv2.imwrite(out_img, a_img)
                    a_lbl.to_csv(out_lbl, sep=" ", header=False, index=False)
                    new_paths.append(out_img)
                    uid += 1

    append_list(list_file, new_paths)

    # report final counts
    counts = {}
    for lf in os.listdir(label_dir):
        if not lf.endswith(".txt"): continue
        df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
        for cid in df[0]:
            counts[cid] = counts.get(cid, 0) + 1
    print(f"[INFO] Balanced class distribution: {counts}")

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

def build_6ch_CHW_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Return (6,H,W) uint8 = [RGB_raw, RGB_proc] where RGB_proc = Unsharp(L) -> NLM(RGB)."""
    rgb_raw  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bgr_usm  = unsharp_L_on_BGR(bgr)
    rgb_proc = nlm_rgb_to_rgb(bgr_usm)
    six_hwc  = np.dstack([rgb_raw, rgb_proc]).astype(np.uint8)  # (H,W,6)
    six_chw  = six_hwc.transpose(2, 0, 1)                       # (6,H,W)  
    return six_chw

# ---------- IO helpers ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def convert_folder_to_6ch_tiff_like_ultra(src_img_dir: str, dst_img_dir: str) -> list:
    ensure_dirs(dst_img_dir)
    out_paths = []
    for p in tqdm(sorted(Path(src_img_dir).iterdir()), desc=f"6ch:{Path(src_img_dir).name}"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {p}")
            continue

        six_chw = build_6ch_CHW_from_bgr(bgr)  # (6,H,W) uint8

        op = Path(dst_img_dir) / (p.stem + ".tiff")
        save_tiff_multipage_from_chw(six_chw, str(op))
        out_paths.append(str(op.resolve()))
    return out_paths

def mirror_labels_by_stem(src_lbl_dir: str, dst_lbl_dir: str, stems: list):
    """Copy .txt labels with the same stem names into dst_lbl_dir."""
    ensure_dirs(dst_lbl_dir)
    copied, missing = 0, 0
    for s in stems:
        src = Path(src_lbl_dir) / f"{s}.txt"
        dst = Path(dst_lbl_dir) / f"{s}.txt"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
    print(f"[INFO] labels copied={copied}, missing={missing} → {dst_lbl_dir}")

def save_tiff_multipage_from_chw(chw: np.ndarray, out_path: str):
    """
    Save a (C, H, W) uint8 array as a multi-page TIFF file using OpenCV's imwritemulti.
    Each channel becomes a separate page in the TIFF file.

    Args:
        chw (np.ndarray): Image array of shape (C, H, W), dtype=uint8.
        out_path (str): Output file path (.tiff).
    """
    if not hasattr(cv2, "imwritemulti"):
        raise RuntimeError("Your OpenCV build does not have 'imwritemulti'. "
                           "Please install opencv-python>=4.8 with TIFF support.")

    if chw.ndim != 3:
        raise ValueError(f"Expected array with shape (C, H, W), got {chw.shape}")

    # Convert channel-first (C,H,W) to list of (H,W) pages
    pages = [np.ascontiguousarray(chw[c].astype(np.uint8, copy=False)) for c in range(chw.shape[0])]

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwritemulti(str(out_path), pages)
    if not ok:
        raise RuntimeError(f"cv2.imwritemulti failed for: {out_path}")

      
# ==============================
# Main
# ==============================
if __name__ == "__main__":
            
    # 1) Tiling 
    if NEED_CROPPING:
        crop_images_and_labels(
            image_dir=TRAIN_IMG_DIR,
            label_dir=TRAIN_LBL_DIR,
            out_img_dir=TRAIN_OUT_IMG_DIR,
            out_lbl_dir=TRAIN_OUT_LBL_DIR,
            list_out=TRAIN_LIST_CROPPED,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )
        crop_images_and_labels(
            image_dir=VAL_IMG_DIR,
            label_dir=VAL_LBL_DIR,
            out_img_dir=VAL_OUT_IMG_DIR,
            out_lbl_dir=VAL_OUT_LBL_DIR,
            list_out=VAL_LIST_CROPPED,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )

    # 2) Class balancing on CROPPED tiles
    if NEED_AUGMENTATION:
        balance_classes(
            image_dir=TRAIN_OUT_IMG_DIR,
            label_dir=TRAIN_OUT_LBL_DIR,
            list_file=TRAIN_LIST_CROPPED,
            class_balance_threshold=CLASS_BALANCE_THRESHOLD,
            augmentation_repeats=AUGMENTATION_REPEATS
        )
        balance_classes(
            image_dir=VAL_OUT_IMG_DIR,
            label_dir=VAL_OUT_LBL_DIR,
            list_file=VAL_LIST_CROPPED,
            class_balance_threshold=CLASS_BALANCE_THRESHOLD,
            augmentation_repeats=AUGMENTATION_REPEATS
        )   
        
    if NEED_6CHANNEL: 
        train_out_paths = convert_folder_to_6ch_tiff_like_ultra(TRAIN_OUT_IMG_DIR, TRAIN_OUT_IMG6_DIR)
        val_out_paths   = convert_folder_to_6ch_tiff_like_ultra(VAL_OUT_IMG_DIR,   VAL_OUT_IMG6_DIR)
    
        train_stems = [Path(p).stem for p in train_out_paths]
        val_stems   = [Path(p).stem for p in val_out_paths]
        mirror_labels_by_stem(TRAIN_OUT_LBL_DIR, TRAIN_OUT_LBL6_DIR, train_stems)
        mirror_labels_by_stem(VAL_OUT_LBL_DIR,   VAL_OUT_LBL6_DIR,   val_stems)
    
        write_list(TRAIN_LIST_6CH, train_out_paths)
        write_list(VAL_LIST_6CH,   val_out_paths)
        print(f"[OK] wrote {len(train_out_paths)} train and {len(val_out_paths)} val 6-ch .tiff files")
        print(f"[OK] lists:\n  {TRAIN_LIST_6CH}\n  {VAL_LIST_6CH}")

    model = YOLO("yolo11x-obb.pt")

    # Size 128
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
        lr0=0.005,
        lrf=0.05,
        weight_decay=0.001,
        dropout=0.2,
        patience=10000,
        plots=False,              
        overlap_mask=False,
        # task='obb',
        # mosaic=0.0, mixup=0.0, copy_paste=0.0,
        # hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        # amp=False
    )
    
    # # Size 416
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
    #     lr0 = 0.002,  
    #     lrf = 0.05,      
    #     weight_decay = 0.001, 
    #     dropout = 0.2,
    #     # warmup_epochs = 5.0,
    #     # warmup_momentum = 0.85,
    #     # warmup_bias_lr = 0.08,
    #     patience=10000,
    #     plots = False,
    #     overlap_mask = False,
    # )
