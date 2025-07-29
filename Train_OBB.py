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

# Configuration
need_cropping = True 
need_augmentation = True
tile_size = 416
overlap = 150
epochs = 150
batch_size = 16
object_boundary_threshold = 0.1  # Minimum fraction of the bounding box that must remain in the crop
class_balance_threshold = 400  # Minimum number of samples per class for balance
augmentation_repeats = 2  # Number of times to augment underrepresented classes

def update_txt_file(txt_file, new_paths):
    """
    Update the .txt file with new paths of cropped or augmented images.
    """
    with open(txt_file, "w") as f:
        for path in new_paths:
            f.write(f"{path}\n")
        
def convert_to_grayscale(image):
    """
    Convert an image to grayscale and ensure it has 3 channels.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  
    return gray_image

def crop_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, txt_file, cropped_txt_file, tile_size=512, overlap=0):
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

    if need_cropping:
        crop_images_and_labels(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            txt_file=txt_file,
            cropped_txt_file=cropped_txt_file,
            tile_size=tile_size,
            overlap=overlap,
        )

        crop_images_and_labels(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            output_image_dir=val_output_image_dir,
            output_label_dir=val_output_label_dir,
            txt_file=val_txt_file,
            cropped_txt_file=val_cropped_txt_file,
            tile_size=tile_size,
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

    model = YOLO("yolo11x-obb.pt")
    
    # Size 416
    model.train(
        data="datasets/GeoMap/data.yaml",
        epochs=epochs,
        imgsz=tile_size, 
        batch=batch_size,
        multi_scale=False,
        lr0 = 0.002,  
        lrf = 0.05,      
        weight_decay = 0.001, 
        dropout = 0.2,
        # warmup_epochs = 5.0,
        # warmup_momentum = 0.85,
        # warmup_bias_lr = 0.08,
        patience=0,
        plots = True,
        overlap_mask = False,
        device=[0, 1] if torch.cuda.is_available() else "CPU",
    )
    
    # # Size 128
    # model.train(
    #     data="datasets/GeoMap/data.yaml",
    #     epochs=epochs,
    #     imgsz=tile_size,  
    #     batch=batch_size,
    #     multi_scale=True,
    #     lr0 = 0.005,  
    #     lrf = 0.05,      
    #     weight_decay = 0.001, 
    #     dropout = 0.3,
    #     plots = True,
    #     patience=10000,
    #     overlap_mask = False,
    #     device=[0, 1] if torch.cuda.is_available() else "CPU",
    # )