# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 13:14:46 2025

Multi-channel OBB Detection

@author: amoslemi
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import copy

# =========================
# Config
# =========================
calculate_metrics = False
tile_sizes = [128, 416]
overlaps = [30, 100]
models = [YOLO("best128.pt"), YOLO("best416.pt")]

channels = 3        # 3 or 4      
MS_SIGMAS = (0, 0.6, 1.2, 2.4)
DT_BIN_METHOD = "percentile"
DT_P_HI, DT_P_LO = 90, 65
DT_MORPH_OPEN = 1

MAP_MIN_SCORE = 0.001
iou_thr = 0.25   # Metrics
iou_threshold = 0.4 # Merge 

APPLY_BORDER_FILTER = True
MARGIN_128 = 10
MARGIN_416 = 20

all_dets_per_image = {}

CLASS_NAMES = {
    0: "Landslide 1",
    1: "Strike",
    2: "Spring 1",
    3: "Minepit 1",
    4: "Hillside", #Colluvial Deposits
    5: "Feuchte", #Wetlands
    6: "Torf", #Peat
    7: "Bergsturz", #Rockfall
    8: "Landslide 2",
    9: "Spring 2",
    10: "Spring 3",
    11: "Minepit 2",
}

CLASS_COLORS = {
    0: (255, 0, 0),    # Landslide
    1: (0, 255, 0),    # Strike
    2: (0, 0, 255),    # Spring
    3: (255, 255, 0),  # Minepit
    4: (255, 0, 255),  # Hillside
    5: (0, 255, 255),  # Feuchte
    6: (0, 0, 0),      # Torf
    7: (240, 34, 0),   # Bergsturz
    8: (50, 20, 60),   # Landslide 2
    9: (60, 50, 20),   # Spring 2
    10: (200, 150, 80),# Spring 3
    11: (100, 200, 150),# Minepit 2
}

start_time = time.time()

def run_inference_on_crop(crop_bgr, model):
    net_input = build_multich(crop_bgr, out_channels=channels)

    with torch.no_grad():
        if calculate_metrics:
            results = model(net_input, conf=0.001)  
        else:
            results = model(net_input, conf=0.25)

    return results

def build_multich(bgr: np.ndarray, out_channels: int = channels) -> np.ndarray:
    
    assert out_channels in (3, 4), f"Unsupported out_channels={out_channels}"

    # --- 3-channel ---
    if out_channels == 3:
        return np.ascontiguousarray(bgr)
        
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # --- 4-channel: RGB + DT-edge ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    acc = None
    for s in MS_SIGMAS:
        blur = cv2.GaussianBlur(gray, (0, 0), s, s, borderType=cv2.BORDER_REFLECT_101) if s > 0 else gray
        mag = cv2.magnitude(
            cv2.Scharr(blur, cv2.CV_32F, 1, 0),
            cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        )
        acc = mag if acc is None else np.maximum(acc, mag)

    if DT_BIN_METHOD == "otsu":
        acc8 = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(acc8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        lo, hi = np.percentile(acc, [DT_P_LO, DT_P_HI])
        edges = (acc >= hi).astype(np.uint8) * 255

    if DT_MORPH_OPEN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k, iterations=DT_MORPH_OPEN)

    non_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)[1]
    dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3).astype(np.float32)
    lo, hi = np.percentile(dist, [1, 99])
    dist = np.clip((dist - lo) / max(1e-6, (hi - lo)), 0, 1)

    tau = 3.0
    soft = np.exp(-dist / tau)
    acc8_nrm = cv2.normalize(acc, None, 0, 1, cv2.NORM_MINMAX)
    soft = 0.7 * soft + 0.3 * acc8_nrm
    soft = np.clip(soft, 0, 1)
    dt_edge = (soft * 255).astype(np.uint8)

    out4 = np.dstack([rgb_raw, dt_edge]).astype(np.uint8)
    return np.ascontiguousarray(out4)

def compute_angle_from_bbox(points):
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    angle = np.arctan2(x4 - x1, y4 - y1) * (180.0 / np.pi)
    if angle > 0:
        angle = 180 - angle
    else:
        angle = abs(angle)
    return angle

def compute_polygon_iou(box1, box2):
    """
    Compute IoU between two rotated bounding boxes.
    """
    poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
    poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0.0

def margin_for(tile_size: int) -> int:
    return MARGIN_128 if tile_size <= 128 else MARGIN_416

def box_center_from_xyxyxyxy(points8):
    """
    points8: [x1,y1,x2,y2,x3,y3,x4,y4] in *global* coordinates.
    """
    cx = (points8[0] + points8[2] + points8[4] + points8[6]) / 4.0
    cy = (points8[1] + points8[3] + points8[5] + points8[7]) / 4.0
    return cx, cy

def center_inside_safe_region(points8, crop_x0, crop_y0, crop_w, crop_h, margin_px: int) -> bool:
    """
    Check if the box center is at least `margin_px` away from each crop border.
    """
    cx, cy = box_center_from_xyxyxyxy(points8)
    cx_rel = cx - crop_x0
    cy_rel = cy - crop_y0
    return (margin_px <= cx_rel <= (crop_w - margin_px)) and (margin_px <= cy_rel <= (crop_h - margin_px))

def merge_detections(detections, iou_threshold=0.5):
    """
    Merge overlapping detections while considering confidence and class types.
    """
    if not detections:
        return []

    detections.sort(key=lambda x: x[9], reverse=True)
    merged = []

    for det1 in detections:
        box1, cls1, conf1 = det1[:8], det1[8], det1[9]

        keep = True

        for det2 in merged:
            box2, cls2 = det2[:8], det2[8]
            if cls1 == cls2 and compute_polygon_iou(box1, box2) >= iou_threshold:
                keep = False
                break

        if keep:
            merged.append(det1)

    return merged

def detect_symbols(image, model, tile_size: int, overlap: int):
    """
    Tiled detection. Works for both:
      - 3-channel
      - 4-channel
    Output format for each det:
      (x1,y1,x2,y2,x3,y3,x4,y4, cls_id, conf, angle)
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

            results = run_inference_on_crop(crop, model)
            crop_dets = []

            for det in results[0].obb:
                points = [float(v) for v in det.xyxyxyxy[0].flatten().tolist()]
                cls_id = int(det.cls[0])
                conf = float(det.conf[0])

                gx = [points[0] + x, points[2] + x, points[4] + x, points[6] + x]
                gy = [points[1] + y, points[3] + y, points[5] + y, points[7] + y]
                global_points8 = [
                    gx[0], gy[0],
                    gx[1], gy[1],
                    gx[2], gy[2],
                    gx[3], gy[3],
                ]

                if APPLY_BORDER_FILTER and margin_px > 0:
                    if not center_inside_safe_region(
                        global_points8,
                        crop_x0=x, crop_y0=y,
                        crop_w=crop_w, crop_h=crop_h,
                        margin_px=margin_px
                    ):
                        continue

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

            detections.extend(merge_detections(crop_dets, iou_threshold))

    return detections

def process_image(image_path, output_dir):
    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Warn] Could not read image: {image_path}")
        return

    # 1) run detection on each scale
    dets_by_scale = {}
    for tile_size, overlap, model in zip(tile_sizes, overlaps, models):
        dets = detect_symbols(image, model, tile_size, overlap)
        dets_by_scale[tile_size] = dets

    if calculate_metrics:
        dets_all_scales = []
        for s in dets_by_scale:
            dets_all_scales.extend(dets_by_scale[s])
        merged_for_map = merge_detections(dets_all_scales, iou_threshold)
        dets_by_scale_for_pr = copy.deepcopy(dets_by_scale)
        consensus_dets = cross_scale_consensus_filter(dets_by_scale_for_pr)
        merged_for_pr = merge_detections(consensus_dets, iou_threshold)
    else:
        consensus_dets = cross_scale_consensus_filter(dets_by_scale)
        merged_for_pr = merge_detections(consensus_dets, iou_threshold)

    print(f"--- {time.time() - t0:.3f} seconds ---")

    result_image = image.copy()
    image_name = os.path.basename(image_path)
    excel_path = os.path.join(
        output_dir,
        image_name.replace(".jpg", ".xlsx").replace(".png", ".xlsx")
    )

    rows = []
    H, W = result_image.shape[:2]
    for (x1, y1, x2, y2, x3, y3, x4, y4, cls_id, conf, angle) in merged_for_pr:

        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        color = tuple(int(c) for c in color)
        label = CLASS_NAMES.get(cls_id, f"Class{cls_id}")

        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=2)

        tx = int(max(0, min(W - 1, round(min(x1, x2, x3, x4)))))
        ty = int(max(0, min(H - 1, round(min(y1, y2, y3, y4) - 10))))
        cv2.putText(result_image, f"{label} {conf:.2f}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)

        rows.append([label, x1, y1, x2, y2, x3, y3, x4, y4, conf, angle])

    out_jpg = os.path.join(
        output_dir,
        image_name.replace(".jpg", "_detected.jpg").replace(".png", "_detected.jpg")
    )
    cv2.imwrite(out_jpg, result_image)

    df = pd.DataFrame(
        rows,
        columns=["Class","X1","Y1","X2","Y2","X3","Y3","X4","Y4","Confidence","Angle"]
    )
    df.to_excel(excel_path, index=False)

    if calculate_metrics:
        global all_dets_per_image_pr
        global all_dets_per_image_map
    
        if "all_dets_per_image_pr" not in globals():
            all_dets_per_image_pr = {}
        if "all_dets_per_image_map" not in globals():
            all_dets_per_image_map = {}
    
        all_dets_per_image_pr[image_path]  = merged_for_pr  
        all_dets_per_image_map[image_path] = merged_for_map  

    global all_dets_per_image
    all_dets_per_image[image_path] = merged_for_pr 

def cross_scale_consensus_filter(dets_by_scale):

    CONS_IOU_PARTNER = 0.40   # IoU threshold to consider same object
    CONS_LOW = 0.25           # detections below this are ignored in fusion
    CONS_HIGH = 0.70          # solo keep threshold if no partner

    kept = []
    scales = sorted(dets_by_scale.keys())

    # single-scale 
    if len(scales) == 1:
        return list(dets_by_scale[scales[0]])

    # multi-scale 
    dets_by_scale_f = {
        s: [d for d in dets_by_scale[s] if d[9] >= CONS_LOW]
        for s in scales
    }

    visited = {s: [False] * len(dets_by_scale_f[s]) for s in scales}

    flat = []
    for s in scales:
        for idx, d in enumerate(dets_by_scale_f[s]):
            flat.append((s, idx, d))

    others = {sc: [t for t in scales if t != sc] for sc in scales}

    for s, i, d in flat:
        if visited[s][i]:
            continue

        cls_d = int(d[8])
        conf_d = float(d[9])

        best_partner = None
        best_partner_tuple = None
        best_partner_conf = -1.0
        best_partner_iou = 0.0

        for t in others[s]:
            pool = dets_by_scale_f[t]
            for j, p in enumerate(pool):
                if visited[t][j]:
                    continue
                if int(p[8]) != cls_d:
                    continue
                iou = compute_polygon_iou(d[:8], p[:8])
                if iou >= CONS_IOU_PARTNER:
                    conf_p = float(p[9])
                    if (conf_p > best_partner_conf) or (
                        conf_p == best_partner_conf and iou > best_partner_iou
                    ):
                        best_partner = p
                        best_partner_tuple = (t, j, p)
                        best_partner_conf = conf_p
                        best_partner_iou = iou

        # no partner found
        if best_partner is None or best_partner_conf < CONS_LOW:
            if conf_d >= CONS_HIGH:
                kept.append(d) 
            visited[s][i] = True
            continue

        # partner found - keep stronger one
        conf_p = best_partner_conf
        if conf_d >= conf_p:
            kept.append(d)
        else:
            kept.append(best_partner)

        visited[s][i] = True
        t, j, _ = best_partner_tuple
        visited[t][j] = True

    return kept

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

def _match_dets_to_gts_pixel(dets, gts, iou_thr=0.5):
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

# Functions to compute AP / mAP across IoU thresholds 
def compute_ap_from_pr(recall, precision):
    """
    Compute AP as area under precision-recall curve using trapezoidal rule.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return ap

def gather_detections_and_gts(dets_source, all_images, cls_id):
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
    Compute precision, recall, and AP
    """

    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return (np.array([0.0]),
                np.array([0.0]),
                0.0, 0.0, 0.0, 0, 0, 0)

    dets_sorted = sorted(dets, key=lambda x: x["score"], reverse=True)

    if len(dets_sorted) == 0:
        return (np.array([0.0]),
                np.array([0.0]),
                0.0, 0.0, 0.0, 0, 0, npos)

    tp = np.zeros(len(dets_sorted))
    fp = np.zeros(len(dets_sorted))

    matched = {img: np.zeros(len(gts.get(img, [])), dtype=bool) for img in gts.keys()}

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
    total_TP = int(tp_cum[-1])
    total_FP = int(fp_cum[-1])
    total_FN = npos - total_TP

    return precision, recall, ap, total_TP, total_FP, total_FN

def _gt_class_ids(all_images):
    s = set()
    for img in all_images:
        for g in _load_gt_as_pixels(img):
            s.add(int(g["cls"]))
    return sorted(s)

def evaluate_map(dets_source, all_images, iou_list=None):
    if iou_list is None:
        IOU_LIST_FOR_MAP = [0.5] + [round(0.5 + 0.05*i, 2) for i in range(1, 10)]
        iou_list = IOU_LIST_FOR_MAP

    # prefer wide (map) source
    use_source = globals().get("all_dets_per_image_map", dets_source)

    class_ids = _gt_class_ids(all_images)

    per_iou_map = {}
    for iou in iou_list:
        ap_list = []

        for cid in class_ids:
            dets, gts = gather_detections_and_gts(use_source, all_images, cid)

            precision, recall, ap, TP, FP, FN = compute_pr_for_class(
                dets, gts, iou_thr=iou
            )

            if ap is not None:
                ap_list.append(ap)

        per_iou_map[iou] = float(np.mean(ap_list)) if ap_list else 0.0

    map50 = per_iou_map.get(0.5, 0.0)
    map5095 = float(np.mean([per_iou_map[i] for i in iou_list])) if iou_list else 0.0

    return {
        "mAP@0.5": map50,
        "mAP@[0.5:0.95]": map5095,
        "per_iou": per_iou_map,
    }

def evaluate_center_hit(all_images, conf_thr=0.5):
    """
    Center-Hit metric: a detection is TP if its center falls inside a GT polygon of the same class.
    """
    tp = fp = fn = 0

    for img_path in all_images:
        dets = [d for d in all_dets_per_image.get(img_path, [])
                if (d[9] >= conf_thr)]

        gts = [g for g in _load_gt_as_pixels(img_path)]

        used = [False] * len(gts)  
        for d in dets:
            cls = int(d[8])
            cx, cy = box_center_from_xyxyxyxy(d[:8])
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

def _evaluate_dataset(all_images, conf_thr, iou_thr):
    tot_tp = tot_fp = tot_fn = 0
    for img_path in all_images:
        dets_all = all_dets_per_image.get(img_path, [])
        filtered = [d for d in dets_all if d[9] >= conf_thr]
        gts = _load_gt_as_pixels(img_path)
        tp, fp, fn = _match_dets_to_gts_pixel(filtered, gts, iou_thr=iou_thr)
        tot_tp += tp; tot_fp += fp; tot_fn += fn
    return _prec_rec_f1(tot_tp, tot_fp, tot_fn)

def _classwise_report(dets_source, all_images, conf_thr, iou_thr):
    rows = []
    all_cids = set()
    for dets in dets_source.values():
        for d in dets:
            all_cids.add(int(d[8]))

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
        cname = CLASS_NAMES.get(cid, str(cid))
        rows.append([cid, cname, tp, fp, fn, P, R, F1])

    df = pd.DataFrame(rows, columns=["cls_id","class","TP","FP","FN","Precision","Recall","F1"])
    out_path = os.path.join(output_dir, "fusion_classwise_metrics.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[Saved] {out_path}")
    return df

def run_fusion_eval(input_dir, iou_thr=iou_thr):
    global all_dets_per_image
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    if not all_images:
        print("[Eval] No images found for evaluation.")
        return

    print(f"Tile size: {tile_sizes}, Overlap: {overlaps}")

    # ===== SINGLE-SCALE =====
    if len(models) == 1 or len(tile_sizes) == 1:
        thr = float(iou_thr)
        print(f"[Single-scale] Using manual threshold: {thr:.2f}")
        P, R, F1 = _evaluate_dataset(all_images, conf_thr=thr, iou_thr=iou_thr)
        print(f"[Report @ {thr:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")

        _classwise_report(all_dets_per_image, all_images, conf_thr=thr, iou_thr=iou_thr)
        evaluate_center_hit(all_images, conf_thr=thr)

        maps = evaluate_map(all_dets_per_image, all_images, iou_list=list(np.arange(0.5, 0.96, 0.05)))
        print("[mAP Results]")
        print(f"mAP@0.5 = {maps['mAP@0.5']:.4f}")
        print(f"mAP@[0.5:0.95] = {maps['mAP@[0.5:0.95]']:.4f}")

        maps_soft = evaluate_map(all_dets_per_image, all_images, iou_list=[0.30,0.40,0.50,0.60,0.70])
        print("[mAP (soft) Results]")
        print(f"mAP@0.3 = {maps_soft['per_iou'][0.30]:.4f}")
        soft_avg = float(np.mean([maps_soft['per_iou'][i] for i in [0.30,0.40,0.50,0.60,0.70]]))
        print(f"mAP@[0.3:0.7] = {soft_avg:.4f}")
        return

    # ===== Dual-SCALE =====
    print("[Fusion] scale-agnostic merge (late fusion).")
    thr = float(iou_thr)
    print(f"[Fusion] Using manual threshold: {thr:.2f}")

    P, R, F1 = _evaluate_dataset(all_images, conf_thr=thr, iou_thr=iou_thr)
    print(f"[Fusion @ {thr:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")

    _classwise_report(all_dets_per_image, all_images, conf_thr=thr, iou_thr=iou_thr)
    evaluate_center_hit(all_images, conf_thr=thr)

    maps = evaluate_map(all_dets_per_image, all_images, iou_list=list(np.arange(0.5, 0.96, 0.05)))
    print("[mAP Results]")
    print(f"mAP@0.5 = {maps['mAP@0.5']:.4f}")
    print(f"mAP@[0.5:0.95] = {maps['mAP@[0.5:0.95]']:.4f}")

    maps_soft = evaluate_map(all_dets_per_image, all_images, iou_list=[0.30,0.40,0.50,0.60,0.70])
    print("[mAP (soft) Results]")
    print(f"mAP@0.3 = {maps_soft['per_iou'][0.30]:.4f}")
    soft_avg = float(np.mean([maps_soft['per_iou'][i] for i in [0.30,0.40,0.50,0.60,0.70]]))
    print(f"mAP@[0.3:0.7] = {soft_avg:.4f}")
    
# =========================
# Main
# =========================
input_dir = "Input"
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith((".jpg",".png",".jpeg",".tif",".tiff")):
        print(f"Processing {fname}...")
        process_image(os.path.join(input_dir,fname), output_dir)
        print(f"Results saved for {fname}")

print(f"--- {time.time() - start_time:.2f} seconds ---")

if calculate_metrics:
    try:
        run_fusion_eval(input_dir, iou_thr=iou_thr)
    except Exception as e:
        print(f"[Eval] Skipped due to error: {e}")
