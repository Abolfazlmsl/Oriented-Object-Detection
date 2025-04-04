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
need_cropping = False 
need_augmentation = False
tile_size = 128
overlap = 50
epochs = 300
batch_size = 16
object_boundary_threshold = 0.1  # Minimum fraction of the bounding box that must remain in the crop
class_balance_threshold = 600  # Minimum number of samples per class for balance
augmentation_repeats = 10  # Number of times to augment underrepresented classes

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
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Ensure 3-channel format for consistency
    return image

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
        
        # Convert normalized coordinates to absolute values
        labels[["x1", "x2", "x3", "x4"]] *= w
        labels[["y1", "y2", "y3", "y4"]] *= h
        
        step = tile_size - overlap
        tile_id = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                crop = image[y:y + tile_size, x:x + tile_size]
                if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                    continue

                # Find labels within the crop region
                tile_labels = labels[
                    ((labels["x1"] + labels["x4"])/2 >= x) & ((labels["x1"] + labels["x4"])/2 < x + tile_size) &
                    ((labels["y1"] + labels["y4"])/2 >= y) & ((labels["y1"] + labels["y4"])/2 < y + tile_size)
                ].copy()

                # Adjust coordinates of labels for the crop
                tile_labels[["x1", "x2", "x3", "x4"]] -= x
                tile_labels[["y1", "y2", "y3", "y4"]] -= y
                
                tile_labels[["x1", "x2", "x3", "x4"]] = tile_labels[["x1", "x2", "x3", "x4"]].clip(0, tile_size)
                tile_labels[["y1", "y2", "y3", "y4"]] = tile_labels[["y1", "y2", "y3", "y4"]].clip(0, tile_size)

                # Normalize coordinates with respect to tile size
                tile_labels[["x1", "x2", "x3", "x4"]] /= tile_size
                tile_labels[["y1", "y2", "y3", "y4"]] /= tile_size

                # Skip saving this crop if no valid labels remain
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

    # Set alpha and sigma based on image size
    if alpha is None:
        alpha = min(shape) * 0.03  
    if sigma is None:
        sigma = alpha * 0.1  

    # Generate displacement fields
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))

    # Clip indices to stay in range
    indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)

    # Ensure indices have the correct shape
    assert indices_x.shape == shape, f"indices_x shape mismatch: {indices_x.shape} vs {shape}"
    assert indices_y.shape == shape, f"indices_y shape mismatch: {indices_y.shape} vs {shape}"

    return cv2.remap(image, indices_x, indices_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_single_class_augmentation(image, labels, target_class):
    """
    Apply augmentations to an image and labels, ensuring labels remain valid for YOLO format.
    """
    aug_image = image.copy()
    aug_labels = labels.copy()

    # Select only target class labels
    target_labels = aug_labels[aug_labels[0] == target_class].copy()
    
    # Original dimensions
    height, width = aug_image.shape[:2]

    # Apply random scaling
    scale_factor = random.uniform(0.6, 1.4)  # Random scale between 60% to 140%
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    aug_image = cv2.resize(aug_image, (new_width, new_height))

    # Adjust label coordinates based on scaling
    for i in range(1, 9):  # Updating all 8 coordinate values
        target_labels[i] *= scale_factor

    # Apply random shifting
    shift_x = random.randint(-20, 20)
    shift_y = random.randint(-20, 20)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aug_image = cv2.warpAffine(aug_image, M, (new_width, new_height))

    # Adjust labels after shifting
    target_labels.iloc[:, 1::2] += shift_x  
    target_labels.iloc[:, 2::2] += shift_y  

    # Convert image to HSV and modify brightness/saturation
    hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] *= random.uniform(0.8, 1.2)  
    hsv[:, :, 2] *= random.uniform(0.8, 1.2)  
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    aug_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Apply elastic transformation
    aug_image = elastic_transform(aug_image)

    # **Normalize label coordinates (Fixing YOLO issue)**
    target_labels.iloc[:, 1::2] /= new_width  # Normalize x-coordinates
    target_labels.iloc[:, 2::2] /= new_height  # Normalize y-coordinates

    # **Ensure all label values remain in the [0,1] range**
    target_labels.iloc[:, 1:] = target_labels.iloc[:, 1:].clip(0, 1)

    aug_labels.update(target_labels)
    return aug_image, aug_labels


def update_balanced_txt_file(txt_file, new_paths):
    """
    Append new paths of augmented images to the .txt file.
    """
    with open(txt_file, "a") as f:  # Append mode
        for path in new_paths:
            f.write(f"{path}\n")

def balance_classes(image_dir, label_dir, txt_file, class_balance_threshold=100, augmentation_repeats=5):
    """
    Balance classes by oversampling underrepresented classes with augmentations,
    and update the txt file with new image paths.
    """
    # Calculate class distribution
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    class_counts = {}
    
    for label_file in label_files:
        labels = pd.read_csv(os.path.join(label_dir, label_file), sep=" ", header=None)
        for class_id in labels[0]:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print(f"Initial class distribution: {class_counts}")

    new_image_paths = []
    
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
                aug_image, aug_labels = apply_single_class_augmentation(image, labels, class_id)
                
                # Save augmented image
                aug_image_filename = f"{os.path.splitext(label_file)[0]}_aug_{random.randint(0, 10000)}.jpg"
                aug_image_path = os.path.join(image_dir, aug_image_filename)
                cv2.imwrite(aug_image_path, aug_image)
                
                # Save augmented labels
                aug_label_filename = f"{os.path.splitext(label_file)[0]}_aug_{random.randint(0, 10000)}.txt"
                aug_label_path = os.path.join(label_dir, aug_label_filename)
                aug_labels.to_csv(aug_label_path, sep=" ", header=False, index=False)
                
                # Add new image path to list for txt file
                new_image_paths.append(aug_image_path)
    
    # Update the txt file with new paths
    update_balanced_txt_file(txt_file, new_image_paths)

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
        )

        balance_classes(
            image_dir=val_output_image_dir,
            label_dir=val_output_label_dir,
            txt_file=val_cropped_txt_file,
        )

    model = YOLO("yolo11x-obb.pt")
    
    # # Size 416
    # model.train(
    #     data="datasets/GeoMap/data.yaml",
    #     epochs=epochs,
    #     imgsz=tile_size,  # Image size (same as crop size)
    #     batch=batch_size,
    #     multi_scale=False,
    #     lr0 = 0.002,  
    #     lrf = 0.03,      
    #     weight_decay = 0.003, 
    #     dropout = 0.4,
    #     # warmup_epochs = 5.0,
    #     # warmup_momentum = 0.85,
    #     # warmup_bias_lr = 0.08,
    #     patience=0,
    #     plots = True,
    #     overlap_mask = False,
    #     device=[0, 1] if torch.cuda.is_available() else "CPU",
    # )
    
    # Size 128
    model.train(
        data="datasets/GeoMap/data.yaml",
        epochs=epochs,
        imgsz=tile_size,  # Image size (same as crop size)
        batch=batch_size,
        multi_scale=True,
        lr0 = 0.005,  
        lrf = 0.05,      
        weight_decay = 0.001, 
        dropout = 0.3,
        plots = True,
        patience=10000,
        overlap_mask = False,
        device=[0, 1] if torch.cuda.is_available() else "CPU",
    )