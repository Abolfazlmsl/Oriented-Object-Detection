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
from shapely.geometry import Polygon, Point
import time
import math

start_time = time.time()

calculate_metrics = True
MAP_MIN_SCORE = 0.001                   
IOU_LIST_FOR_MAP = [0.5] + [round(0.5 + 0.05*i, 2) for i in range(1, 10)] 
tile_sizes = [128, 416]
overlaps = [30, 100]
iou_threshold = 0.4 # Representation
iou_thr = 0.25 # Metric
models = [YOLO("best128.pt"), YOLO("best416.pt")]

# ===== Evaluation / Fusion Controls =====
# Single-scale controls
SINGLE_MANUAL_THR = 0.25    # used only when SINGLE_USE_BEST=False

# Two-scale controls (base + add-on)
MS_BASE_SCALE = 128            # which scale is the base (e.g., 128 or 416)
MS_BASE_MANUAL_THR = 0.25      # used only when MS_BASE_USE_BEST=False

MS_ADD_SCALE = 416             # the second scale to add on top of base
MS_ADD_MANUAL_THR = 0.25       # used only when MS_ADD_USE_BEST=False

# --- Border suppression ---
APPLY_BORDER_FILTER = True
MARGIN_128 = 10  
MARGIN_416 = 20  

CONS_IOU_PARTNER = 0.40   # IoU threshold to consider two boxes (same class) as partners
CONS_LOW  = 0.25          # lower bound for "usable" confidence in fusion stage
CONS_HIGH = 0.70          # high-confidence threshold for solo-keep and high-vs-high

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
}


# Add threshold for each class
if calculate_metrics:
    CLASS_THRESHOLDS = {i: 0.25 for i in CLASS_NAMES.keys()}
else:
    CLASS_THRESHOLDS = {
        0: 0.6,  # Landslide 1
        1: 0.8,  # Strike
        2: 0.8,  # Spring 1
        3: 0.8,  # Minepit 1
        4: 0.8,  # Hillside
        5: 0.7,  # Feuchte
        6: 0.7,  # Torf
        7: 0.92,  # Bergsturz
        8: 0.8,  # Landslide 2
        9: 0.7,  # Spring 2
        10: 0.7,  # Spring 3
        11: 0.6,  # Minepit 2
    }
    
# Classes to exclude completely (will not be shown on the image)
EXCLUDED_CLASSES = {}
all_dets_per_image = {}  

def get_imgsz(model, default_size):
    """
    Get the effective input size (imgsz) for this YOLO model.
    Falls back to the tile_size if unavailable.
    """
    try:
        sz = model.model.args.get('imgsz', None)
        if isinstance(sz, (list, tuple)):
            sz = max(sz)
        return int(sz) if sz else int(default_size)
    except Exception:
        return int(default_size)

def margin_for(tile_size: int) -> int:
    return MARGIN_128 if tile_size <= 128 else MARGIN_416

def box_center_from_xyxyxyxy(points8):
    """
    points8: [x1,y1,x2,y2,x3,y3,x4,y4] in *global* coordinates.
    Returns (cx, cy) as the average of the four vertices.
    """
    cx = (points8[0] + points8[2] + points8[4] + points8[6]) / 4.0
    cy = (points8[1] + points8[3] + points8[5] + points8[7]) / 4.0
    return cx, cy

def center_inside_safe_region(points8, crop_x0, crop_y0, crop_w, crop_h, margin_px: int) -> bool:
    """
    Check if the box center is at least `margin_px` away from each crop border.
    crop_* are in *global* coordinates except crop_w/h which are sizes.
    """
    cx, cy = box_center_from_xyxyxyxy(points8)
    cx_rel = cx - crop_x0
    cy_rel = cy - crop_y0
    return (margin_px <= cx_rel <= (crop_w - margin_px)) and (margin_px <= cy_rel <= (crop_h - margin_px))


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
        return 0.0 

    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union > 0 else 0.0

def _center_of_poly8(poly8):
    # poly8: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = poly8[0::2]; ys = poly8[1::2]
    return (sum(xs) / 4.0, sum(ys) / 4.0)


def detect_symbols(image, model, tile_size: int, overlap: int):
    """
    Run tiled detection and return global OBB detections.
    Assumes model(crop) returns objects with .obb where each item has:
        - xyxyxyxy (tensor shape [1,8])
        - conf (tensor shape [1])
        - cls (tensor shape [1])
    """
    H, W = image.shape[:2]
    step = max(1, tile_size - overlap)
    detections = []

    margin_px = margin_for(tile_size)

    for y in range(0, H, step):
        for x in range(0, W, step):
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            crop = image[y:y2, x:x2]
            crop_h, crop_w = crop.shape[:2]

            if crop_h == 0 or crop_w == 0:
                continue

            results = model(crop)
            crop_dets = []

            for det in results[0].obb:
                points = [float(v) for v in det.xyxyxyxy[0].flatten().tolist()]
                cls_id = int(det.cls[0])
                conf = float(det.conf[0])

                # per-class conf threshold
                if conf < CLASS_THRESHOLDS.get(cls_id, 0.05):
                    continue

                # convert local tile coords -> global image coords
                gx = [points[0] + x, points[2] + x, points[4] + x, points[6] + x]
                gy = [points[1] + y, points[3] + y, points[5] + y, points[7] + y]
                global_points8 = [
                    gx[0], gy[0], gx[1], gy[1],
                    gx[2], gy[2], gx[3], gy[3]
                ]

                # OPTIONAL border suppression (disabled for fair eval)
                if APPLY_BORDER_FILTER and margin_px > 0:
                    if not center_inside_safe_region(
                        global_points8,
                        crop_x0=x, crop_y0=y,
                        crop_w=crop_w, crop_h=crop_h,
                        margin_px=margin_px
                    ):
                        continue

                # angle only for Strike
                if CLASS_NAMES.get(cls_id, f"Class{cls_id}") == "Strike":
                    angle = compute_angle_from_bbox(points)
                else:
                    angle = 0.0

                crop_dets.append((
                    global_points8[0], global_points8[1],
                    global_points8[2], global_points8[3],
                    global_points8[4], global_points8[5],
                    global_points8[6], global_points8[7],
                    cls_id, conf, angle
                ))

            # local NMS merge for this crop
            detections.extend(merge_detections(crop_dets, iou_threshold))

    return detections

def merge_detections(detections, iou_threshold=0.5, exclude_check=True):
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
    
    detections.sort(key=lambda x: x[9], reverse=True)  
    merged = []
    excluded_boxes = [det[:11] for det in detections if det[8] in EXCLUDED_CLASSES]
    
    for det1 in detections:
        box1, cls1, conf1 = det1[:8], det1[8], det1[9]
        if cls1 in EXCLUDED_CLASSES:
            continue 
        
        # poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
        keep = True

        if exclude_check:
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

def int_point(x, y, width, height):
    """Clamp and cast (x, y) to a valid integer tuple inside the image."""
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(width - 1, xi))
    yi = max(0, min(height - 1, yi))
    return (xi, yi)
    
def _best_partner(det, pool, iou_thr):
    """Return the best same-class partner of `det` in `pool` based on IoU."""
    box1, cls1 = det[:8], int(det[8])
    best = None
    best_iou = 0.0
    for d in pool:
        if int(d[8]) != cls1:
            continue
        iou = compute_polygon_iou(box1, d[:8])
        if iou >= iou_thr and iou > best_iou:
            best = d
            best_iou = iou
    return best, best_iou

def cross_scale_consensus_filter(dets_by_scale):
    """
    Input: dict {scale -> [detections]}, detection = (x1..y4, cls, score, angle)
    Rules:
      - SOLO: if no partner -> keep only if conf >= CONS_HIGH (else drop)
      - BOTH in [CONS_LOW, CONS_HIGH): keep the higher, drop the other
      - BOTH >= CONS_HIGH: keep the higher, drop the other
      - MIXED: one >= CONS_HIGH and other in [CONS_LOW, CONS_HIGH) -> keep the higher (the high one)
      - A partner with conf < CONS_LOW is ignored (treated as no partner)
    Output: list of detections after applying the above consensus (no WBF).
    """
    scales = sorted(dets_by_scale.keys())
    if len(scales) == 1:
        return list(dets_by_scale[scales[0]])

    kept = []
    visited = {s: [False]*len(dets_by_scale[s]) for s in scales}

    # flatten for single pass
    flat = []
    for s in scales:
        for idx, d in enumerate(dets_by_scale[s]):
            flat.append((s, idx, d))

    # map other scales for each scale
    others = {sc: [t for t in scales if t != sc] for sc in scales}

    for s, i, d in flat:
        if visited[s][i]:
            continue

        cls_d = int(d[8])
        conf_d = float(d[9])

        # find best partner across other scales (same class, IoU >= CONS_IOU_PARTNER)
        best_partner = None
        best_partner_tuple = None
        best_partner_conf = -1.0
        best_partner_iou = 0.0

        for t in others[s]:
            pool = dets_by_scale[t]
            for j, p in enumerate(pool):
                if visited[t][j]:
                    continue
                if int(p[8]) != cls_d:
                    continue
                iou = compute_polygon_iou(d[:8], p[:8])
                if iou >= CONS_IOU_PARTNER:
                    conf_p = float(p[9])
                    # choose partner primarily by higher confidence (tie-breaker: higher IoU)
                    if (conf_p > best_partner_conf) or (conf_p == best_partner_conf and iou > best_partner_iou):
                        best_partner = p
                        best_partner_tuple = (t, j, p)
                        best_partner_conf = conf_p
                        best_partner_iou = iou

        if best_partner is None or best_partner_conf < CONS_LOW:
            # SOLO case (no usable partner)
            if conf_d >= CONS_HIGH:
                kept.append(d)  # keep strong solo
            # else drop
            visited[s][i] = True
            continue

        # we have a usable partner (best_partner_conf >= CONS_LOW)
        conf_p = best_partner_conf

        # define buckets for clarity
        d_low   = (CONS_LOW <= conf_d  < CONS_HIGH)
        d_high  = (conf_d >= CONS_HIGH)
        p_low   = (CONS_LOW <= conf_p  < CONS_HIGH)
        p_high  = (conf_p  >= CONS_HIGH)

        # CASES:
        # 1) both in low-range -> keep higher
        if d_low and p_low:
            if conf_d >= conf_p:
                kept.append(d)
            else:
                kept.append(best_partner)
            visited[s][i] = True
            t, j, _ = best_partner_tuple
            visited[t][j] = True
            continue

        # 2) both high -> keep higher
        if d_high and p_high:
            if conf_d >= conf_p:
                kept.append(d)
            else:
                kept.append(best_partner)
            visited[s][i] = True
            t, j, _ = best_partner_tuple
            visited[t][j] = True
            continue

        # 3) mixed: one high, one low -> keep the higher (the high one)
        if d_high and p_low:
            kept.append(d)
        elif d_low and p_high:
            kept.append(best_partner)
        else:
            # fallback: if any weird edge, keep the higher
            if conf_d >= conf_p:
                kept.append(d)
            else:
                kept.append(best_partner)

        visited[s][i] = True
        t, j, _ = best_partner_tuple
        visited[t][j] = True

    return kept


def process_image(image_path, output_dir):
    """
    Process one image:
      - Run tiled detection for each (tile_size, overlap, model).
      - If multi-scale and SIZE_ROUTING_ENABLED: do size-based routing
        (small boxes from the smaller scale, base scale handles the rest).
      - Else (single-scale): keep the original merging behavior.
      - Draw, save image and Excel as before.
    """
    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Warn] Could not read image: {image_path}")
        return

    dets_by_scale = {}
    for tile_size, overlap, model in zip(tile_sizes, overlaps, models):
        dets = detect_symbols(image, model, tile_size, overlap)  
        dets_by_scale[tile_size] = dets

    consensus_dets = cross_scale_consensus_filter(dets_by_scale)
    merged_detections = merge_detections(consensus_dets, iou_threshold, False)

    print(f"--- {time.time() - t0:.3f} seconds ---")

    # 3) Drawing and export
    result_image = image.copy()
    image_name = os.path.basename(image_path)
    excel_path = os.path.join(
        output_dir,
        image_name.replace(".jpg", ".xlsx").replace(".png", ".xlsx")
    )

    rows = []
    H, W = result_image.shape[:2]
    for (x1, y1, x2, y2, x3, y3, x4, y4, cls_id, conf, angle) in merged_detections:
        if cls_id in EXCLUDED_CLASSES:
            continue

        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        color = tuple(int(c) for c in color)  
        label = CLASS_NAMES.get(cls_id, f"Class{cls_id}")

        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=2)

        tx = int(max(0, min(W - 1, round(min(x1, x2, x3, x4)))))
        ty = int(max(0, min(H - 1, round(min(y1, y2, y3, y4) - 10))))
        cv2.putText(result_image, f"{label} {conf:.2f}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)
        
        # text_x = min(x1, x2, x3, x4)
        # text_y = min(y1, y2, y3, y4) - 10  # Shift text above the box
        # cv2.putText(result_image, f"{label} {conf:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw polygon
        # points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        # cv2.polylines(result_image, [points], isClosed=True, color=color, thickness=2)
        
        # # Annotate corners with numbers 1 to 4
        # corner_coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # for idx, (px, py) in enumerate(corner_coords, start=1):
        #     cv2.putText(result_image, str(idx), (px + 3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        #     cv2.putText(result_image, str(idx), (px + 3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # # Place label text above the topmost y-coordinate of the box
        # text_x = min(x1, x2, x3, x4)
        # text_y = min(y1, y2, y3, y4) - 15  # Shift text above the corner numbers
        # cv2.putText(result_image, f"{label} {conf:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        rows.append([label, x1, y1, x2, y2, x3, y3, x4, y4, conf, angle])

    # save visual
    out_jpg = os.path.join(
        output_dir,
        image_name.replace(".jpg", "_detected.jpg").replace(".png", "_detected.jpg")
    )
    cv2.imwrite(out_jpg, result_image)

    # save excel
    df = pd.DataFrame(rows, columns=["Class", "X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "Confidence", "Angle"])
    df.to_excel(excel_path, index=False)

    all_dets_per_image[image_path] = merged_detections


def _label_path_for_image(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    cand1 = os.path.join(os.path.dirname(image_path), base)
    if os.path.exists(cand1):
        return cand1
    labels_dir = os.path.join(os.path.dirname(image_path), "Labels")
    cand2 = os.path.join(labels_dir, base)
    if os.path.exists(cand2):
        return cand2
    return None

def _load_gt_as_pixels(image_path):
    lp = _label_path_for_image(image_path)
    gts = []
    if lp is None or not os.path.exists(lp):
        return gts
    img = cv2.imread(image_path)
    if img is None:
        return gts
    h, w = img.shape[:2]
    with open(lp, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:])) 
            pts_pix = [(vals[i]*w, vals[i+1]*h) for i in range(0,8,2)]
            gts.append({"cls": cls_id, "pts": pts_pix})
    return gts

def _match_dets_to_gts_pixel(dets, gts, iou_thr):
    """
    dets: list of tuples (x1..y4, cls, conf, angle) in pixels
    gts:  list of dicts {"cls": int, "pts": [(x,y)*4]} in pixels
    returns: (TP, FP, FN)
    """
    used = [False]*len(gts)
    tp = 0
    for det in dets:
        box1 = det[:8]
        cls1 = int(det[8])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if used[j] or cls1 != g["cls"]:
                continue
            box2 = [coord for pt in g["pts"] for coord in pt]
            iou = compute_polygon_iou(box1, box2)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            used[best_j] = True
            tp += 1
    fp = len(dets) - tp
    fn = used.count(False)
    return tp, fp, fn

def _prec_rec_f1(tp, fp, fn):
    P = tp / (tp + fp + 1e-9)
    R = tp / (tp + fn + 1e-9)
    F1 = 2 * P * R / (P + R + 1e-9)
    return P, R, F1

def _evaluate_dataset(all_images, conf_thr, iou_thr):
    tot_tp = tot_fp = tot_fn = 0
    for img_path in all_images:
        dets_all = all_dets_per_image.get(img_path, [])
        try:
            filtered = [d for d in dets_all if (d[9] >= conf_thr and (d[8] not in EXCLUDED_CLASSES))]
        except NameError:
            filtered = [d for d in dets_all if d[9] >= conf_thr]
        gts = _load_gt_as_pixels(img_path)
        try:
            gts = [g for g in gts if g["cls"] not in EXCLUDED_CLASSES]
        except NameError:
            pass
        tp, fp, fn = _match_dets_to_gts_pixel(filtered, gts, iou_thr=iou_thr)
        tot_tp += tp; tot_fp += fp; tot_fn += fn
    return _prec_rec_f1(tot_tp, tot_fp, tot_fn)

def _classwise_report(dets_source, all_images, conf_thr, iou_thr):
    rows = []
    # collect class ids from source
    all_cids = set()
    for dets in dets_source.values():
        for d in dets:
            all_cids.add(int(d[8]))
    try:
        all_cids = [cid for cid in sorted(all_cids) if cid not in EXCLUDED_CLASSES]
    except NameError:
        all_cids = sorted(all_cids)

    for cid in all_cids:
        tp=fp=fn=0
        for img_path in all_images:
            dets_all = dets_source.get(img_path, [])
            dets_c = [d for d in dets_all if (int(d[8])==cid and d[9]>=conf_thr)]
            gts = _load_gt_as_pixels(img_path)
            gts_c = [g for g in gts if g["cls"]==cid]
            tpp,fpp,fnn = _match_dets_to_gts_pixel(dets_c, gts_c, iou_thr=iou_thr)
            tp += tpp; fp += fpp; fn += fnn
        P,R,F1 = _prec_rec_f1(tp,fp,fn)
        cname = CLASS_NAMES.get(cid, str(cid)) if 'CLASS_NAMES' in globals() else str(cid)
        rows.append([cid, cname, tp, fp, fn, P, R, F1])

    df = pd.DataFrame(rows, columns=["cls_id","class","TP","FP","FN","Precision","Recall","F1"])
    out_path = os.path.join(output_dir, "fusion_classwise_metrics.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df

# ---- functions to compute AP / mAP across IoU thresholds ----
def compute_ap_from_pr(recall, precision):
    """
    Compute AP as area under precision-recall curve using trapezoidal rule.
    recall and precision arrays must be sorted increasing recall.
    """
    # ensure monotonic precision envelope (common VOC/COCO step)
    # For stability, append endpoints
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # make precision monotonically decreasing
    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    # compute area
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return ap

def gather_detections_and_gts(dets_source, all_images, cls_id):
    try:
        if cls_id in EXCLUDED_CLASSES:
            return [], {}
    except NameError:
        pass
    dets, gts = [], {}
    for img_path in all_images:
        img_dets = [d for d in dets_source.get(img_path, []) if int(d[8]) == cls_id]
        for d in img_dets:
            if d[9] >= MAP_MIN_SCORE:
                dets.append({"image_id": img_path, "score": float(d[9]), "bbox": d[:8]})
        gt_boxes = _load_gt_as_pixels(img_path)
        gts[img_path] = [[c for pt in g["pts"] for c in pt] for g in gt_boxes if g["cls"] == cls_id]
    return dets, gts

def compute_pr_for_class(dets, gts, iou_thr=0.5):
    """
    dets: list of {'image_id','score','bbox'}
    gts:  dict image_id -> list of gt bboxes (list of 8 coords)
    returns:
      precision array,
      recall array,
      ap (area under PR),
      mean_precision (avg precision over curve),
      mean_recall (avg recall over curve),
      total_TP, total_FP, total_FN
    """
    npos = sum(len(v) for v in gts.values())  # total GT count
    if npos == 0:
        return np.array([0.0]), np.array([0.0]), None, 0.0, 0.0, 0, 0, 0

    dets_sorted = sorted(dets, key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(dets_sorted))
    fp = np.zeros(len(dets_sorted))
    matched = {img: np.zeros(len(gts.get(img, [])), dtype=bool) for img in gts.keys()}

    for i, det in enumerate(dets_sorted):
        img = det["image_id"]
        box_det = det["bbox"]
        best_iou, best_j = 0.0, -1
        gt_list = gts.get(img, [])

        for j, gt_box in enumerate(gt_list):
            if matched[img][j]:
                continue
            iou = compute_polygon_iou(box_det, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1
            matched[img][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (npos + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = compute_ap_from_pr(recall, precision)

    # mean precision / recall across curve
    mean_precision = float(np.mean(precision)) if len(precision) else 0.0
    mean_recall    = float(np.mean(recall))    if len(recall)    else 0.0

    # total TP, FP, FN for this class at the end of sweep
    total_TP = int(tp_cum[-1])
    total_FP = int(fp_cum[-1])
    total_FN = int(npos - total_TP)

    return precision, recall, ap, mean_precision, mean_recall, total_TP, total_FP, total_FN

def _gt_class_ids(all_images):
    s = set()
    for img in all_images:
        for g in _load_gt_as_pixels(img):
            s.add(int(g["cls"]))
    try:
        s = {cid for cid in s if cid not in EXCLUDED_CLASSES}
    except NameError:
        pass
    return sorted(s)

def evaluate_map(dets_source, all_images, iou_list=None):
    if iou_list is None:
        iou_list = IOU_LIST_FOR_MAP

    class_ids = _gt_class_ids(all_images)

    per_iou_map = {}
    P_lists = {}
    R_lists = {}

    total_TP_sum = 0
    total_FP_sum = 0
    total_FN_sum = 0

    for iou in iou_list:
        ap_list = []
        P_lists[iou] = []
        R_lists[iou] = []

        for cid in class_ids:
            dets, gts = gather_detections_and_gts(dets_source, all_images, cid)

            precision, recall, ap, mean_P, mean_R, TP, FP, FN = compute_pr_for_class(
                dets, gts, iou_thr=iou
            )

            if ap is not None:
                ap_list.append(ap)

            P_lists[iou].append(mean_P)
            R_lists[iou].append(mean_R)

            # only accumulate totals for IoU = 0.5 (standard report)
            if abs(iou - 0.5) < 1e-6:
                total_TP_sum += TP
                total_FP_sum += FP
                total_FN_sum += FN

        per_iou_map[iou] = float(np.mean(ap_list)) if ap_list else 0.0

    map50 = per_iou_map.get(0.5, 0.0)
    map5095 = float(np.mean([per_iou_map[i] for i in iou_list])) if iou_list else 0.0

    if 0.5 in P_lists and len(P_lists[0.5]) > 0:
        mean_precision_global = float(np.mean(P_lists[0.5]))
        mean_recall_global    = float(np.mean(R_lists[0.5]))
    else:
        first_iou = iou_list[0]
        mean_precision_global = float(np.mean(P_lists[first_iou])) if P_lists[first_iou] else 0.0
        mean_recall_global    = float(np.mean(R_lists[first_iou])) if R_lists[first_iou] else 0.0

    return {
        "mAP@0.5": map50,
        "mAP@[0.5:0.95]": map5095,
        "per_iou": per_iou_map,
        "mean_precision": mean_precision_global,
        "mean_recall": mean_recall_global,
        "TP": int(total_TP_sum),
        "FP": int(total_FP_sum),
        "FN": int(total_FN_sum),
    }

def evaluate_center_hit(all_images, conf_thr=0.5):
    """
    Center-Hit metric: a detection is TP if its center falls inside a GT polygon of the same class.
    Uses EXCLUDED_CLASSES just like other metrics.
    Returns (P, R, F1) and prints a summary.
    """
    tp = fp = fn = 0

    for img_path in all_images:
        # detections (apply conf threshold, ignore excluded classes)
        dets = [d for d in all_dets_per_image.get(img_path, [])
                if (d[9] >= conf_thr and int(d[8]) not in EXCLUDED_CLASSES)]

        # ground truths (ignore excluded classes)
        gts = [g for g in _load_gt_as_pixels(img_path) if g["cls"] not in EXCLUDED_CLASSES]

        used = [False] * len(gts)  # each GT can be matched at most once

        for d in dets:
            cls = int(d[8])
            cx, cy = _center_of_poly8(d[:8])
            p_center = Point(cx, cy)

            matched = False
            for j, g in enumerate(gts):
                if used[j] or g["cls"] != cls:
                    continue
                poly = Polygon(g["pts"])
                if not poly.is_valid:
                    continue
                if poly.contains(p_center):
                    tp += 1
                    used[j] = True
                    matched = True
                    break

            if not matched:
                fp += 1

        # any GTs left unmatched are FNs
        fn += sum(1 for u in used if not u)

    P, R, F1 = _prec_rec_f1(tp, fp, fn)
    print(f"[Center-Hit @ conf≥{conf_thr:.2f}] P={P:.3f} R={R:.3f} F1={F1:.3f} (TP={tp}, FP={fp}, FN={fn})")
    return P, R, F1


def run_fusion_eval(input_dir, iou_thr):
    global all_dets_per_image
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    if not all_images:
        print("[Eval] No images found for evaluation.")
        return

    print(f"Tile size: {tile_sizes}, Overlap: {overlaps}")

    # ===== SINGLE-SCALE =====
    if len(models) == 1 or len(tile_sizes) == 1:
        thr = float(SINGLE_MANUAL_THR)
        print(f"[Single-scale] Using manual threshold: {thr:.2f}")
        P, R, F1 = _evaluate_dataset(all_images, conf_thr=thr, iou_thr=iou_thr)
        print(f"[Report @ {thr:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")

        _classwise_report(all_dets_per_image, all_images, conf_thr=thr, iou_thr=iou_thr)
        evaluate_center_hit(all_images, conf_thr=thr)

        maps = evaluate_map(all_dets_per_image, all_images, iou_list=list(np.arange(0.5, 0.96, 0.05)))
        print("[mAP Results]")
        print(f"mAP@0.5 = {maps['mAP@0.5']:.4f}")
        print(f"mAP@[0.5:0.95] = {maps['mAP@[0.5:0.95]']:.4f}")
        print(f"Mean Precision (PR-sweep, IoU=0.5) = {maps['mean_precision']:.4f}")
        print(f"Mean Recall    (PR-sweep, IoU=0.5) = {maps['mean_recall']:.4f}")
        print(f"TP={maps['TP']}, FP={maps['FP']}, FN={maps['FN']}")

        maps_soft = evaluate_map(all_dets_per_image, all_images, iou_list=[0.30,0.40,0.50,0.60,0.70])
        print("[mAP (soft) Results]")
        print(f"mAP@0.3 = {maps_soft['per_iou'][0.30]:.4f}")
        soft_avg = float(np.mean([maps_soft['per_iou'][i] for i in [0.30,0.40,0.50,0.60,0.70]]))
        print(f"mAP@[0.3:0.7] = {soft_avg:.4f}")
        return

    # ===== MULTI-SCALE =====
    print("[Fusion] scale-agnostic merge (late fusion).")
    thr = float(MS_BASE_MANUAL_THR)  
    print(f"[Fusion] Using manual threshold: {thr:.2f}")

    P, R, F1 = _evaluate_dataset(all_images, conf_thr=thr, iou_thr=iou_thr)
    print(f"[Fusion @ {thr:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")

    _classwise_report(all_dets_per_image, all_images, conf_thr=thr, iou_thr=iou_thr)
    evaluate_center_hit(all_images, conf_thr=thr)

    maps = evaluate_map(all_dets_per_image, all_images, iou_list=list(np.arange(0.5, 0.96, 0.05)))
    print("[mAP Results]")
    print(f"mAP@0.5 = {maps['mAP@0.5']:.4f}")
    print(f"mAP@[0.5:0.95] = {maps['mAP@[0.5:0.95]']:.4f}")
    print(f"Mean Precision (PR-sweep, IoU=0.5) = {maps['mean_precision']:.4f}")
    print(f"Mean Recall    (PR-sweep, IoU=0.5) = {maps['mean_recall']:.4f}")
    print(f"TP={maps['TP']}, FP={maps['FP']}, FN={maps['FN']}")

    maps_soft = evaluate_map(all_dets_per_image, all_images, iou_list=[0.30,0.40,0.50,0.60,0.70])
    print("[mAP (soft) Results]")
    print(f"mAP@0.3 = {maps_soft['per_iou'][0.30]:.4f}")
    soft_avg = float(np.mean([maps_soft['per_iou'][i] for i in [0.30,0.40,0.50,0.60,0.70]]))
    print(f"mAP@[0.3:0.7] = {soft_avg:.4f}")


input_dir = "Input"
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

for image_file in os.listdir(input_dir):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        print(f"Processing {image_file}...")
        process_image(os.path.join(input_dir, image_file), output_dir)
        print(f"Results saved for {image_file}")

print("--- %s seconds ---" % (time.time() - start_time))

if calculate_metrics:
    try:
        run_fusion_eval(input_dir, iou_thr=iou_thr)
    except Exception as e:
        print(f"[Eval] Skipped due to error: {e}")
