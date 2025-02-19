#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:52:07 2025

@author: abolfazl
"""

import os
import cv2
import pandas as pd
import numpy as np
import random
import torch
from ultralytics import YOLO

# Configuration
need_cropping = False 
need_augmentation = False
tile_size = 150
overlap = 50
epochs = 150
batch_size = 16
object_boundary_threshold = 0.1  # Minimum fraction of the bounding box that must remain in the crop
class_balance_threshold = 500  # Minimum number of samples per class for balance
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
                    (labels["x1"] >= x) & (labels["x1"] < x + tile_size) &
                    (labels["y1"] >= y) & (labels["y1"] < y + tile_size)
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

def apply_single_class_augmentation(image, labels, target_class):
    """
    Apply augmentations to an image and labels, targeting a specific class.
    """
    aug_image = image.copy()
    aug_labels = labels.copy()

    # Select only target class labels
    target_labels = aug_labels[aug_labels[0] == target_class].copy()
    
    # Apply random scaling
    scale_factor = random.uniform(0.8, 1.2)  # Random scale between 80% to 120%
    height, width = aug_image.shape[:2]
    aug_image = cv2.resize(aug_image, (int(width * scale_factor), int(height * scale_factor)))
    
    # Adjust label coordinates based on scaling
    for i in range(1, 9):  # Updating all 8 coordinate values
        target_labels[i] *= scale_factor
    
    # Change brightness and contrast
    aug_image = cv2.convertScaleAbs(aug_image, alpha=1.2, beta=50)

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
    
    # # Size 400
    # model.train(
    #     data="datasets/GeoMap/data.yaml",
    #     epochs=epochs,
    #     imgsz=tile_size,  # Image size (same as crop size)
    #     batch=batch_size,
    #     multi_scale=True,
    #     lr0 = 0.003,  
    #     lrf = 0.05,      
    #     weight_decay = 0.001, 
    #     dropout = 0.4,
    #     plots = True,
    #     overlap_mask = False,
    #     device=[0, 1] if torch.cuda.is_available() else "CPU",
    # )
    
    # Size 150
    model.train(
        data="datasets/GeoMap/data.yaml",
        epochs=epochs,
        imgsz=tile_size,  # Image size (same as crop size)
        batch=batch_size,
        multi_scale=True,
        lr0 = 0.005,  
        lrf = 0.05,      
        weight_decay = 0.001, 
        dropout = 0.2,
        plots = True,
        overlap_mask = False,
        device=[0, 1] if torch.cuda.is_available() else "CPU",
    )