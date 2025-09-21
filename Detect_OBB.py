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

start_time = time.time()

calculate_metrics = True
MAP_MIN_SCORE = 0.001                   
IOU_LIST_FOR_MAP = [0.5] + [round(0.5 + 0.05*i, 2) for i in range(1, 10)] 
tile_sizes = [128, 416]
overlaps = [20, 50]
iou_threshold = 0.2 # Representation
iou_thr = 0.2 # Metric
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
        12: 0.05,  # Spring B2
        13: 0.05,  # Hillside B2
    }
    
# Classes to exclude completely (will not be shown on the image)
EXCLUDED_CLASSES = {12, 13}

all_dets_per_image = {}  

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


def detect_symbols(image, model, tile_size, overlap):
    """
    Detect objects in an image using a given model, tile size, and overlap.
    """
    h, w, _ = image.shape
    step = tile_size - overlap
    detections = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            crop_detections = []
            crop = image[y:y + tile_size, x:x + tile_size]
            # if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
            #     continue
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
                crop_detections.append((x1 + x, y1 + y, x2 + x, y2 + y, x3 + x, y3 + y,\
                                        x4 + x, y4 + y, cls, conf, angle))
            detections.extend(merge_detections(crop_detections, iou_threshold))
                    
    return detections

def merge_detections(detections, iou_threshold=0.5, excluse_check=True):
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

        if excluse_check:
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
    
    merged_detections = merge_detections(all_detections, iou_threshold, False)
    result_image = image.copy()
    image_name = os.path.basename(image_path)
    excel_path = os.path.join(output_dir, image_name.replace(".jpg", ".xlsx").replace(".png", ".xlsx"))
    
    data = []
    for x1, y1, x2, y2, x3, y3, x4, y4, cls, conf, angle in merged_detections:
        if cls in EXCLUDED_CLASSES:
            continue  
        
        color = CLASS_COLORS.get(cls, (0, 255, 255))
        label = CLASS_NAMES.get(cls, f"Class{cls}")
        
        # Draw polygon
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        cv2.polylines(result_image, [points], isClosed=True, color=color, thickness=2)
        
        # Place label text above the box
        text_x = min(x1, x2, x3, x4)
        text_y = min(y1, y2, y3, y4) - 10  # Shift text above the box
        cv2.putText(result_image, f"{label} {conf:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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


        data.append([label, x1, y1, x2, y2, x3, y3, x4, y4, conf, angle])
    
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_detected.jpg").replace(".png", "_detected.jpg"))
    cv2.imwrite(output_path, result_image)
    
    df = pd.DataFrame(data, columns=["Class", "X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "Confidence", "Angle"])
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

def _find_best_conf_threshold(all_images, iou_thr=0.5):
    best = {"thr": 0.25, "P": 0.0, "R": 0.0, "F1": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        P, R, F1 = _evaluate_dataset(all_images, conf_thr=float(thr), iou_thr=iou_thr)
        if F1 > best["F1"]:
            best = {"thr": float(thr), "P": float(P), "R": float(R), "F1": float(F1)}
    return best

def _classwise_report(all_images, conf_thr, iou_thr):
    rows = []
    all_cids = set()
    for dets in all_dets_per_image.values():
        for d in dets:
            all_cids.add(int(d[8]))
    try:
        all_cids = [cid for cid in sorted(all_cids) if cid not in EXCLUDED_CLASSES]
    except NameError:
        all_cids = sorted(all_cids)

    for cid in all_cids:
        tp=fp=fn=0
        for img_path in all_images:
            dets_all = all_dets_per_image.get(img_path, [])
            dets_c = [d for d in dets_all if (int(d[8])==cid and d[9]>=conf_thr)]
            gts = _load_gt_as_pixels(img_path)
            gts_c = [g for g in gts if g["cls"]==cid]
            tpp,fpp,fnn = _match_dets_to_gts_pixel(dets_c, gts_c, iou_thr=iou_thr)
            tp += tpp; fp += fpp; fn += fnn
        P,R,F1 = _prec_rec_f1(tp,fp,fn)
        try:
            cname = CLASS_NAMES.get(cid, str(cid))
        except NameError:
            cname = str(cid)
        rows.append([cid, cname, tp, fp, fn, P, R, F1])

    if pd is None:
        print("\n[Classwise metrics]")
        for r in rows:
            print(f"cls={r[0]:>2} {r[1]:<20} TP={r[2]:<5} FP={r[3]:<5} FN={r[4]:<5} P={r[5]:.3f} R={r[6]:.3f} F1={r[7]:.3f}")
        return None
    else:
        df = pd.DataFrame(rows, columns=["cls_id","class","TP","FP","FN","Precision","Recall","F1"])
        try:
            save_dir = output_dir
        except NameError:
            save_dir = "."
        out_path = os.path.join(save_dir, "fusion_classwise_metrics.xlsx")
        df.to_excel(out_path, index=False)
        print(f"[Saved] {out_path}")
        return df

# ---- New: functions to compute AP / mAP across IoU thresholds ----
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

def gather_detections_and_gts(all_images, cls_id):
    try:
        if cls_id in EXCLUDED_CLASSES:
            return [], {}
    except NameError:
        pass

    dets, gts = [], {}
    for img_path in all_images:
        img_dets_all = all_dets_per_image.get(img_path, [])

        img_dets_cls = [d for d in img_dets_all if int(d[8]) == cls_id]

        for d in img_dets_cls:
            if d[9] >= MAP_MIN_SCORE:
                dets.append({"image_id": img_path, "score": float(d[9]), "bbox": d[:8]})

        gt_boxes = _load_gt_as_pixels(img_path)
        gts[img_path] = [[c for pt in g["pts"] for c in pt] for g in gt_boxes if g["cls"] == cls_id]

    return dets, gts



def compute_pr_for_class(dets, gts, iou_thr=0.5):
    """
    dets: list of {'image_id','score','bbox'}
    gts: dict image_id -> list of gt bboxes (list of 8 coords)
    returns: precision array, recall array, ap
    """
    # count total positives
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return np.array([0.0]), np.array([0.0]), None
    # sort detections by score desc
    dets_sorted = sorted(dets, key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(dets_sorted))
    fp = np.zeros(len(dets_sorted))
    # keep track of matched GTs per image
    matched = {img: np.zeros(len(gts.get(img,[])), dtype=bool) for img in gts.keys()}
    for i, det in enumerate(dets_sorted):
        img = det["image_id"]
        box_det = det["bbox"]
        best_iou = 0.0
        best_j = -1
        gt_list = gts.get(img, [])
        for j, gt_box in enumerate(gt_list):
            if matched[img][j]:
                continue
            iou = compute_polygon_iou(box_det, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr:
            tp[i] = 1
            matched[img][best_j] = True
        else:
            fp[i] = 1
    # cumulative
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (npos + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = compute_ap_from_pr(recall, precision)
    return precision, recall, ap

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

def evaluate_map(all_images, iou_list=None):
    """
    Compute mAP for all classes across IoU thresholds in iou_list.
    By default calculates mAP@0.5 and mAP@[0.5:0.95]
    """
    if iou_list is None:
        iou_list = IOU_LIST_FOR_MAP
    # class_ids = sorted({int(d[8]) for dets in all_dets_per_image.values() for d in dets if int(d[8]) not in EXCLUDED_CLASSES})
    class_ids = _gt_class_ids(all_images)
    per_iou_map = {}
    # for each IoU threshold compute AP per class
    for iou in iou_list:
        ap_list = []
        for cid in class_ids:
            dets, gts = gather_detections_and_gts(all_images, cid)
            precision, recall, ap = compute_pr_for_class(dets, gts, iou_thr=iou)
            if ap is not None:              
                ap_list.append(ap)
        per_iou_map[iou] = float(np.mean(ap_list)) if ap_list else 0.0
    map50 = per_iou_map.get(0.5, 0.0)
    map5095 = float(np.mean([per_iou_map[i] for i in iou_list])) if iou_list else 0.0
    return {"mAP@0.5": map50, "mAP@[0.5:0.95]": map5095, "per_iou": per_iou_map}

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
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    if not all_images:
        print("[Eval] No images found for evaluation.")
        return

    print(f"Tile size: {tile_sizes}, Overlap: {overlaps}")    
    best = _find_best_conf_threshold(all_images, iou_thr=iou_thr)
    print(f"[Fusion] Best confidence threshold (by F1): {best['thr']:.2f} | P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f}")
    P, R, F1 = _evaluate_dataset(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    print(f"[Fusion @ {best['thr']:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")
    _classwise_report(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    evaluate_center_hit(all_images, conf_thr=best['thr'])
    # compute mAPs
    maps = evaluate_map(all_images, iou_list=list(np.arange(0.5, 0.96, 0.05)))
    print("[mAP Results]")
    print(f"mAP@0.5 = {maps['mAP@0.5']:.4f}")
    print(f"mAP@[0.5:0.95] = {maps['mAP@[0.5:0.95]']:.4f}")
    
    maps_soft = evaluate_map(all_images, iou_list=[0.30, 0.40, 0.50, 0.60, 0.70])
    print("[mAP (soft) Results]")
    print(f"mAP@0.3 = {maps_soft['per_iou'][0.30]:.4f}")
    soft_avg = float(np.mean([maps_soft['per_iou'][i] for i in [0.30,0.40,0.50,0.60,0.70]]))
    print(f"mAP@[0.3:0.7] = {soft_avg:.4f}")
    
    # save per-IoU mAP
    try:
        save_dir = output_dir
    except NameError:
        save_dir = "."
    per_iou_rows = [{"iou": k, "mAP": v} for k, v in maps["per_iou"].items()]
    pd.DataFrame(per_iou_rows).to_excel(os.path.join(save_dir, "fusion_map_per_iou.xlsx"), index=False)
    print(f"[Saved] mAP per IoU table to {os.path.join(save_dir, 'fusion_map_per_iou.xlsx')}")
    
    soft_rows = [{"iou": k, "mAP": v} for k, v in maps_soft["per_iou"].items()]
    pd.DataFrame(soft_rows).to_excel(os.path.join(save_dir, "fusion_map_per_iou_soft.xlsx"), index=False)
    print(f"[Saved] soft mAP per IoU table to {os.path.join(save_dir, 'fusion_map_per_iou_soft.xlsx')}")
    

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
