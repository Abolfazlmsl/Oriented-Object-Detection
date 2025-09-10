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

CHANNELS = 3               # set to 3, 4, or 6
APPLY_FILTERED_RGB = False

# Configuration
need_cropping = False 
need_augmentation = False
Dual_GPU = True
TILE_SIZE = 128
overlap = 50
EPOCHS = 150
BATCH_SIZE = 32
WORKERS = 2
CACHE = False       
RECT = False  
object_boundary_threshold = 0.1  # Minimum fraction of the bounding box that must remain in the crop
class_balance_threshold = 400  # Minimum number of samples per class for balance
augmentation_repeats = 2  # Number of times to augment underrepresented classes

# === Filtering ===
apply_filtered_rgb = False
USM_RADIUS = 7.0           
USM_WEIGHT = 1.40            
NLM_H = 7                    
NLM_T = 7                   
NLM_S = 21  

if Dual_GPU:     
    DEVICE = "0,1" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "0" if torch.cuda.is_available() else "cpu"                 

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
                           txt_file, cropped_txt_file, tile_size=512, overlap=0):
    """
    Crop images and adjust labels for YOLO format. Save results and update the .txt file with new image paths.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    new_paths = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_file}")
            continue

        # Convert image to grayscale
        # image = convert_to_grayscale(image)
        
        h, w, _ = image.shape
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file not found: {label_file}")
            continue

        labels = pd.read_csv(label_path, sep=" ", header=None)
        labels.columns = ["class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        
        labels[["x1", "x2", "x3", "x4"]] *= w
        labels[["y1", "y2", "y3", "y4"]] *= h
        
        step = tile_size - overlap
        tile_id = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                crop = image[y:y + tile_size, x:x + tile_size]
                if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                    continue

                tile_labels = labels[
                    ((labels["x1"] + labels["x4"])/2 >= x) & ((labels["x1"] + labels["x4"])/2 < x + tile_size) &
                    ((labels["y1"] + labels["y4"])/2 >= y) & ((labels["y1"] + labels["y4"])/2 < y + tile_size)
                ].copy()

                tile_labels[["x1", "x2", "x3", "x4"]] -= x
                tile_labels[["y1", "y2", "y3", "y4"]] -= y
                
                tile_labels[["x1", "x2", "x3", "x4"]] = tile_labels[["x1", "x2", "x3", "x4"]].clip(0, tile_size)
                tile_labels[["y1", "y2", "y3", "y4"]] = tile_labels[["y1", "y2", "y3", "y4"]].clip(0, tile_size)

                tile_labels[["x1", "x2", "x3", "x4"]] /= tile_size
                tile_labels[["y1", "y2", "y3", "y4"]] /= tile_size

                if tile_labels.empty:
                    continue

                # Save cropped image
                tile_image_filename = f"{os.path.splitext(image_file)[0]}_tile_{tile_id}.jpg"
                tile_image_path = os.path.join(output_image_dir, tile_image_filename)
                cv2.imwrite(tile_image_path, crop)

                # Save adjusted labels
                tile_label_filename = f"{os.path.splitext(image_file)[0]}_tile_{tile_id}.txt"
                tile_label_path = os.path.join(output_label_dir, tile_label_filename)
                tile_labels.to_csv(tile_label_path, sep=" ", header=False, index=False)


                if not os.path.exists(tile_image_path) or tile_labels.empty:
                    if os.path.exists(tile_image_path):
                        os.remove(tile_image_path)
                    if os.path.exists(tile_label_path):
                        os.remove(tile_label_path)
                    continue
                
                # Store new image path for updating the txt file
                new_paths.append(tile_image_path)

                tile_id += 1

        print(f"Processed image: {image_file}")

    update_txt_file(cropped_txt_file, new_paths)

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
    shift_x = random.randint(-40, 40)
    shift_y = random.randint(-40, 40)
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

if __name__ == "__main__":
    
    image_dir = "datasets/GeoMap/images/train"
    label_dir = "datasets/GeoMap/labels/train"
    output_image_dir = "datasets/GeoMap/cropped/images/train"
    output_label_dir = "datasets/GeoMap/cropped/labels/train"
    txt_file = "datasets/GeoMap/train.txt"
    cropped_txt_file = "datasets/GeoMap/train_cropped.txt"

    val_image_dir = "datasets/GeoMap/images/val"
    val_label_dir = "datasets/GeoMap/labels/val"
    val_output_image_dir = "datasets/GeoMap/cropped/images/val"
    val_output_label_dir = "datasets/GeoMap/cropped/labels/val"
    val_txt_file = "datasets/GeoMap/val.txt"
    val_cropped_txt_file = "datasets/GeoMap/val_cropped.txt"
    
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
        crop_images_and_labels(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            txt_file=txt_file,
            cropped_txt_file=cropped_txt_file,
            tile_size=TILE_SIZE,
            overlap=overlap,
        )

        crop_images_and_labels(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            output_image_dir=val_output_image_dir,
            output_label_dir=val_output_label_dir,
            txt_file=val_txt_file,
            cropped_txt_file=val_cropped_txt_file,
            tile_size=TILE_SIZE,
            overlap=overlap,
        )

    if need_augmentation:
        balance_classes(
            image_dir=output_image_dir,
            label_dir=output_label_dir,
            txt_file=cropped_txt_file,
            class_balance_threshold=class_balance_threshold,
            augmentation_repeats=augmentation_repeats            
        )

        balance_classes(
            image_dir=val_output_image_dir,
            label_dir=val_output_label_dir,
            txt_file=val_cropped_txt_file,
            class_balance_threshold=class_balance_threshold,
            augmentation_repeats=augmentation_repeats 
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
            DATA_YAML = "datasets/GeoMap/data.yaml"
    
    elif CHANNELS == 4:
        print("[INFO] Converting cropped(+aug) to 4-channel multi-page TIFFs...")
        tr_paths = convert_folder_to_4ch_tiff(output_image_dir, out_img4_train)
        va_paths = convert_folder_to_4ch_tiff(val_output_image_dir, out_img4_val)
        # mirror labels by stem
        tr_stems = [os.path.splitext(os.path.basename(p))[0] for p in tr_paths]
        va_stems = [os.path.splitext(os.path.basename(p))[0] for p in va_paths]
        mirror_labels_by_stem(output_label_dir, out_lbl4_train, tr_stems)
        mirror_labels_by_stem(val_output_label_dir, out_lbl4_val, va_stems)
        # write train/val lists
        with open(train_list_4ch, "w") as f:
            for p in tr_paths: f.write(p + "\n")
        with open(val_list_4ch, "w") as f:
            for p in va_paths: f.write(p + "\n")
        DATA_YAML = "datasets/GeoMap/data4ch.yaml"  # create yaml that points to cropped4/images/{train,val}
    
    elif CHANNELS == 6:
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