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
import time

start_time = time.time()

tile_sizes = [128, 416]
overlaps = [70, 260]
iou_threshold = 0.2
models = [YOLO("best128.pt"), YOLO("best416.pt")]

# Define colors for different classes
CLASS_COLORS = {
    0: (255, 0, 0),  # Landslide
    1: (0, 255, 0),  # Strike
    2: (0, 0, 255),  # Spring
    3: (255, 255, 0),  # Minepit
    4: (255, 0, 255),  # Hillside
    5: (0, 255, 255),  # Feuchte
    6: (0, 0, 0),  # Torf
    7: (240, 34, 0),  # Bergsturz
    8: (50, 20, 60), # Landslide 2  
    9: (60, 50, 20), # Spring 2  
    10: (200, 150, 80), # Spring 3  
    11: (100, 200, 150), # Minepit 2 
    12: (12, 52, 83), # Spring B2
    13: (123, 232, 23), # Spring B2 
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
    13: "Hillside B2",
}

# Add threshold for each class
CLASS_THRESHOLDS = {
    0: 0.7,  # Landslide 1
    1: 0.8,  # Strike
    2: 0.8,  # Spring 1
    3: 0.8,  # Minepit 1
    4: 0.8,  # Hillside
    5: 0.7,  # Feuchte
    6: 0.7,  # Torf
    7: 0.45,  # Bergsturz
    8: 0.7,  # Landslide 2
    9: 0.7,  # Spring 2
    10: 0.7,  # Spring 3
    11: 0.6,  # Minepit 2
    12: 0.05,  # Spring B2
    13: 0.05,  # Hillside B2
}

# Classes to exclude completely (will not be shown on the image)
EXCLUDED_CLASSES = {12, 13}  

def compute_angle_from_bbox(points):
    """
    Compute the orientation angle of a bounding box given its four corner points.
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    angle = np.arctan2(x4 - x1, y4 - y1) * (180.0 / np.pi)
    
    if angle > 0:
        angle = 180 - angle
    else:
        angle = np.abs(angle)
        
    return angle


def convert_bgr_to_rgb(image):
    """
    Convert an image to grayscale and return a 3-channel grayscale image.

    Parameters:
    image (np.ndarray): Input color image.

    Returns:
    np.ndarray: 3-channel grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return gray

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
            # crop_gray = convert_bgr_to_rgb(crop)
            results = model(crop)
            for box in results[0].obb:
                points = [int(x) for x in box.xyxyxyxy[0].flatten().tolist()]
                x1, y1, x2, y2, x3, y3, x4, y4 = points
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CLASS_THRESHOLDS.get(cls, 0.05):
                    continue
                
                if CLASS_NAMES.get(cls, f"Class{cls}") == "Strike": 
                    angle = compute_angle_from_bbox(points)
                else:
                    angle = 0
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
    
    detections.sort(key=lambda x: x[9], reverse=True)  # Sort by confidence (higher first)
    merged = []
    excluded_boxes = [det[:10] for det in detections if det[8] in EXCLUDED_CLASSES]
    
    for det1 in detections:
        box1, cls1, conf1 = det1[:8], det1[8], det1[9]
        if cls1 in EXCLUDED_CLASSES:
            continue 
        
        # poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
        keep = True

        for det_excl in excluded_boxes:
           excl_box, excl_cls, excl_conf = det_excl[:8], det_excl[8], det_excl[9]
           # poly_excl = Polygon([(excl_box[i], excl_box[i+1]) for i in range(0, 8, 2)])
           iou = compute_polygon_iou(box1, excl_box)
           
           if iou > 0.3:
               if conf1 > 0.85 or excl_conf < 0.5:
                   continue  
               else:
                   keep = False  
                   break
        
        for det2 in merged:
            box2, cls2 = det2[:8], det2[8]
            # poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])
            
            if cls1 == cls2 and compute_polygon_iou(box1, box2) >= iou_threshold:
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
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    merged_detections = merge_detections(all_detections, iou_threshold)
    result_image = image.copy()
    image_name = os.path.basename(image_path)
    excel_path = os.path.join(output_dir, image_name.replace(".jpg", ".xlsx").replace(".png", ".xlsx"))
    
    data = []
    for x1, y1, x2, y2, x3, y3, x4, y4, cls, conf, angle in merged_detections:
        if cls in EXCLUDED_CLASSES:
            continue  # Ignore excluded classes
        
        color = CLASS_COLORS.get(cls, (0, 255, 255))
        label = CLASS_NAMES.get(cls, f"Class{cls}")
        
        # Draw polygon
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        cv2.polylines(result_image, [points], isClosed=True, color=color, thickness=2)
        
        # Place label text above the topmost y-coordinate of the box
        text_x = min(x1, x2, x3, x4)
        text_y = min(y1, y2, y3, y4) - 10  # Shift text above the box
        cv2.putText(result_image, f"{label} {conf:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

print("--- %s seconds ---" % (time.time() - start_time))