#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:55:54 2025

@author: amoslemi
"""

import cv2
import os
from ultralytics import YOLO
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

tile_sizes = [150]
overlaps = [50]
iou_threshold = 0.01
models = [YOLO("best150.pt")]

# Define colors for different classes
CLASS_COLORS = {
    0: (255, 0, 0),  # Landslide
    1: (0, 255, 0),  # Strike
    2: (0, 0, 255),  # Spring
    3: (255, 255, 0),  # Minepit
    4: (255, 0, 255),  # Hillside
    5: (0, 255, 255),  # Feuchte
    6: (0, 0, 0),  # Torf
    7: (127, 127, 127),  # Bergsturz
    8: (50, 20, 60), # Landslide 2  
    9: (60, 50, 20), # Spring 2  
    10: (200, 150, 80), # Spring 3  
    11: (100, 200, 150), # Minepit 2 
    12: (12, 52, 83), # Spring B2 
}

# Define class names
CLASS_NAMES = {
    0: "Landslide 1",
    1: "Strike",
    2: "Spring 1",
    3: "Minepit 1",
    4: "Hillside",
    5: "Feuchte",
    6: "Torf",
    7: "Bergsturz",
    8: "Landslide 2",
    9: "Spring 2",
    10: "Spring 3",
    11: "Minepit 2",
    12: "Spring B2",
}

# Add threshold for each class
CLASS_THRESHOLDS = {
    0: 0.6,  # Landslide 1
    1: 0.7,  # Strike
    2: 0.6,  # Spring 1
    3: 0.6,  # Minepit 1
    4: 0.7,  # Hillside
    5: 0.05,  # Feuchte
    6: 0.05,  # Torf
    7: 0.05,  # Bergsturz
    8: 0.05,  # Landslide 2
    9: 0.05,  # Spring 2
    10: 0.05,  # Spring 3
    11: 0.4,  # Minepit 2
    12: 0.05,  # Spring B2
}

# Classes to exclude completely (will not be shown on the image)
EXCLUDED_CLASSES = {12}  

def compute_angle_from_bbox(points):
    """
    Compute the orientation angle of a bounding box given its four corner points.
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    angle = np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)
    return angle


def convert_to_grayscale(image):
    """
    Convert an image to grayscale and return a 3-channel grayscale image.

    Parameters:
    image (np.ndarray): Input color image.

    Returns:
    np.ndarray: 3-channel grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def compute_polygon_iou(box1, box2):
    """
    Compute IoU between two rotated bounding boxes.
    
    Parameters:
    box1, box2: Lists containing 8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4).

    Returns:
    float: IoU score.
    """
    poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
    poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0  # If polygons are invalid, return IoU as 0

    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union > 0 else 0.0

def detect_symbols(image, model, tile_size, overlap):
    """
    Detect objects in an image using a given model, tile size, and overlap.
    """
    h, w, _ = image.shape
    step = tile_size - overlap
    detections = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            crop = image[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                continue
            crop_gray = convert_to_grayscale(crop)
            results = model(crop_gray)
            for box in results[0].obb:
                points = [int(x) for x in box.xyxyxyxy[0].flatten().tolist()]
                x1, y1, x2, y2, x3, y3, x4, y4 = points
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CLASS_THRESHOLDS.get(cls, 0.05):
                    continue
                angle = compute_angle_from_bbox(points)
                detections.append((x1 + x, y1 + y, x2 + x, y2 + y, x3 + x, y3 + y,\
                                   x4 + x, y4 + y, cls, conf, angle))
    return detections

def merge_detections(detections, iou_threshold=0.5):
    """
    Merge overlapping detections while considering confidence and class types.
    
    Parameters:
    detections (list): List of detected bounding boxes (x1, y1, ..., x4, y4, class, confidence).
    iou_threshold (float): IoU threshold for merging overlapping detections.

    Returns:
    list: Filtered list of detections.
    """
    if not detections:
        return []
    
    detections.sort(key=lambda x: x[-1], reverse=True)  # Sort by confidence (higher first)
    merged = []

    for i, det1 in enumerate(detections):
        box1 = det1[:8]  # Extract the 4 points
        cls1, conf1 = det1[8], det1[9]
        poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
        area1 = poly1.area
        keep = True

        for j, det2 in enumerate(detections):
            if i == j:
                continue
            
            box2 = det2[:8]
            cls2, conf2 = det2[8], det2[9]
            poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])
            area2 = poly2.area
            iou = compute_polygon_iou(box1, box2)

            if iou >= iou_threshold and cls1 == cls2:
                intersection_area = poly1.intersection(poly2).area
                non_overlap_area = area1 - intersection_area

                if conf1 < conf2:
                    if (non_overlap_area / area1) >= 0.4:
                        keep = True
                    else:
                        keep = False
                        break

        if keep:
            merged.append(det1)

    return merged


def process_image(image_path, output_dir):
    """
    Process an image by applying object detection at multiple tile sizes and merging the results.
    """
    image = cv2.imread(image_path)
    all_detections = []
    for tile_size, overlap, model in zip(tile_sizes, overlaps, models):
        all_detections.extend(detect_symbols(image, model, tile_size, overlap))
    
    merged_detections = merge_detections(all_detections, iou_threshold)
    result_image = image.copy()
    image_name = os.path.basename(image_path)
    excel_path = os.path.join(output_dir, image_name.replace(".jpg", ".xlsx").replace(".png", ".xlsx"))
    
    data = []
    for x1, y1, x2, y2, x3, y3, x4, y4, cls, conf, angle in all_detections:
        color = CLASS_COLORS.get(cls, (0, 255, 255))
        label = CLASS_NAMES.get(cls, f"Class{cls}")
        cv2.polylines(result_image, [np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)], isClosed=True, color=color, thickness=2)
        cv2.putText(result_image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        data.append([label, x1, y1, x2, y2, x3, y3, x4, y4, conf, angle])

    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_detected.jpg").replace(".png", "_detected.png"))
    cv2.imwrite(output_path, result_image)
    
    df = pd.DataFrame(data, columns=["Class", "X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "Confidence", "Angle"])
    df.to_excel(excel_path, index=False)

# Define input and output directories
input_dir = "Input"
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for image_file in os.listdir(input_dir):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        print(f"Processing {image_file}...")
        process_image(os.path.join(input_dir, image_file), output_dir)
        print(f"Results saved for {image_file}")
