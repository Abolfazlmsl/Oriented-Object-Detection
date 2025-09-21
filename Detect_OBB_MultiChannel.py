# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 13:14:46 2025

Multi-channel OBB Detection (6 channels)

@author: amoslemi
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon
from ultralytics.models.yolo.obb.predict import OBBPredictor

# =========================
# Config
# =========================
channels = 3         # set to 3, 4, or 6 
APPLY_FILTER_3CH = False  
FOURCH_MODE = "dtedge"   # "dtedge" | "msedge" | "Lproc" | "palette_per_image"
MS_SIGMAS = (0, 0.6, 1.2, 2.4)
DT_BIN_METHOD = "otsu"  
DT_P_HI, DT_P_LO = 90, 65
DT_MORPH_OPEN = 1
PALETTE_K = 15                      
PALETTE_SAMPLE_DOWN = 2           
KMEANS_MAX_ITER = 20
KMEANS_EPS = 1.0
KMEANS_ATTEMPTS = 1

USM_RADIUS = 7.0
USM_WEIGHT = 1.40
NLM_H = 7
NLM_T = 7
NLM_S = 21

calculate_metrics = True
tile_sizes = [128, 416]
overlaps = [20, 50]
iou_threshold = 0.2
models = [YOLO("best128.pt"), YOLO("best416.pt")]


# Classes/colors/names
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
    12: (12, 52, 83),  # Spring B2
    13: (123, 232, 23) # Hillside B2
}

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

if calculate_metrics:
    CLASS_THRESHOLDS = {cid: 0.0 for cid in CLASS_NAMES.keys()}
    EXCLUDED_CLASSES = {}
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
    EXCLUDED_CLASSES = {12, 13}

all_dets_per_image = {}
start_time = time.time()

# =========================
# Extra channel builders
# =========================

def _unsharp_imagej(bgr_img, radius=USM_RADIUS, weight=USM_WEIGHT, border=cv2.BORDER_REFLECT_101):
    lab32 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab32)
    L_blur  = cv2.GaussianBlur(L, (0, 0), sigmaX=radius, sigmaY=radius, borderType=border)
    L_sharp = np.clip(L + weight * (L - L_blur), 0, 255)
    lab_s   = cv2.merge([L_sharp, A, B]).astype(np.uint8)
    return cv2.cvtColor(lab_s, cv2.COLOR_LAB2BGR)

def _nlm_rgb_to_rgb(bgr_img, h=NLM_H, t=NLM_T, s=NLM_S):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_d = np.empty_like(rgb)
    for c in range(3):
        rgb_d[:, :, c] = cv2.fastNlMeansDenoising(rgb[:, :, c], None, h=h, templateWindowSize=t, searchWindowSize=s)
    return rgb_d  # uint8 RGB


def letterbox_ultra(img, new_shape=640, color=(114, 114, 114), stride=32, auto=True, scaleFill=False, scaleup=True):
    """
    Returns:
      lb_img, ratio(tuple), dwdh(tuple)
    """
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # make padding color match channel count
    C = 1 if img.ndim == 2 else img.shape[2]
    if isinstance(color, tuple) and len(color) != C:
        color = tuple([114] * C)

    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if not np.isscalar(color) and img.ndim == 3 and img.shape[2] > 4:
        color = 114
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)

def ensure_predictor(model, imgsz):
    """
    Create a predictor only to reuse Ultralytics postprocess/NMS.
    """
    if getattr(model, "predictor", None) is not None:
        return model.predictor
    pred = OBBPredictor(overrides={"imgsz": imgsz, "conf": 0.25})
    pred.setup_model(model.model)
    # neutralize warmup flags to avoid 3ch warmup:
    if not hasattr(model.model, "warmup"):
        model.model.warmup = lambda *a, **k: None
    if not hasattr(model.model, "pt"):
        model.model.pt = True
    if not hasattr(model.model, "triton"):
        model.model.triton = False
    model.predictor = pred
    return pred

def run_inference_on_crop(crop_bgr, model, imgsz):
    """
    Unified path for 6 channels:
      build_multich -> letterbox -> tensor -> model.model() -> predictor.postprocess()
    Returns: list[Results] (Ultralytics)
    """
    # build multi-channel (even if CHANNELS==3 returns BGR)
    multi = build_multich(crop_bgr, out_channels=channels)  # HxWxC

    # letterbox like Ultralytics
    stride = int(getattr(model.model, "stride", torch.tensor([32])).max().item())
    lb_img, _, _ = letterbox_ultra(multi, new_shape=imgsz, color=114, auto=True, stride=stride)

    # to tensor (NCHW, float/255)
    im = torch.from_numpy(lb_img).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0
    device = next(model.model.parameters()).device
    dtype  = next(model.model.parameters()).dtype  
    
    # if im.shape[1] >= 3:
    #     rgb3 = im[:, [2, 1, 0], :, :]  
    #     if im.shape[1] == 3:
    #         im = rgb3
    #     else:
    #         # im = torch.cat([rgb3, im[:, 3:, :, :]], dim=1)  
    #         pass
    
    im = im.to(device, non_blocking=True).type(dtype)

    # forward (no extra preprocess)
    with torch.no_grad():
        raw = model.model(im)

    # make list preds and ensure predictor
    preds = raw if isinstance(raw, (list, tuple)) else [raw]
    predictor = ensure_predictor(model, imgsz)

    # predictor.batch[0] is used in construct_results; provide a dummy list of paths
    predictor.batch = (['in_memory'] * len(preds), None, None, None)

    # IMPORTANT: pass orig_imgs as BGR crop (not multi) to keep scaling identical to baseline
    results = predictor.postprocess(preds, im, [crop_bgr])
    return results

def build_multich(bgr: np.ndarray, out_channels: int = channels) -> np.ndarray:
    """
    Build per-crop multi-channel exactly like training:
      3ch: if APPLY_FILTER_3CH -> RGB_proc (Unsharp(L)->NLM on RGB), else RGB_raw
      4ch: [RGB_raw, L_proc] where L_proc = NLM( Unsharp(L in LAB) )
      6ch: [RGB_raw, RGB_proc] where RGB_proc = NLM( Unsharp(L in LAB) -> RGB )
    Returns HxWxC uint8.
    """
    assert out_channels in (3, 4, 6), f"Unsupported out_channels={out_channels}"
    rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if out_channels == 3:
        if APPLY_FILTER_3CH:
            bgr_usm = _unsharp_imagej(bgr, radius=USM_RADIUS, weight=USM_WEIGHT)
            rgb_proc = _nlm_rgb_to_rgb(bgr_usm, h=NLM_H, t=NLM_T, s=NLM_S)  # (H,W,3) uint8
            multich = rgb_proc
        else:
            multich = rgb_raw

    elif out_channels == 4:
        if FOURCH_MODE == "dtedge":
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dt   = dt_edge_channel_from_bgr(
                      bgr,
                      sigmas=MS_SIGMAS,
                      bin_method=DT_BIN_METHOD,
                      p_hi=DT_P_HI, p_lo=DT_P_LO,
                      morph_open=DT_MORPH_OPEN
                  )
            multich = np.dstack([rgb, dt]).astype(np.uint8)
    
        elif FOURCH_MODE == "msedge":
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            edge = ms_edge_channel_from_bgr(bgr, sigmas=MS_SIGMAS)
            multich = np.dstack([rgb, edge]).astype(np.uint8)
    
        elif FOURCH_MODE == "palette_per_image":
            multich = _build_4ch_rgb_plus_index_per_image(bgr, K=PALETTE_K)
        else:  # "Lproc"
            lab32 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            L, A, B = cv2.split(lab32)
            L_blur  = cv2.GaussianBlur(L, (0, 0), sigmaX=USM_RADIUS, sigmaY=USM_RADIUS, borderType=cv2.BORDER_REFLECT_101)
            L_sharp = np.clip(L + USM_WEIGHT * (L - L_blur), 0, 255).astype(np.uint8)
            L_proc  = cv2.fastNlMeansDenoising(L_sharp, None, h=NLM_H, templateWindowSize=NLM_T, searchWindowSize=NLM_S)
            multich = np.dstack([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), L_proc]).astype(np.uint8)
    
    else:  # out_channels == 6
        bgr_usm  = _unsharp_imagej(bgr, radius=USM_RADIUS, weight=USM_WEIGHT)
        rgb_proc = _nlm_rgb_to_rgb(bgr_usm, h=NLM_H, t=NLM_T, s=NLM_S)  # (H,W,3) uint8
        multich  = np.dstack([rgb_raw, rgb_proc]).astype(np.uint8)      # (H,W,6)

    return np.ascontiguousarray(multich)

def _index_per_image_kmeans(bgr: np.ndarray, K=PALETTE_K, sample_down=PALETTE_SAMPLE_DOWN) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if sample_down > 1:
        small = cv2.resize(rgb, (rgb.shape[1] // sample_down, rgb.shape[0] // sample_down),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    Z = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_MAX_ITER, KMEANS_EPS)
    _compactness, labels_small, centers = cv2.kmeans(
        Z, K, None, criteria, KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.float32)  # (K,3) RGB

    flat = rgb.reshape(-1, 3).astype(np.float32)
    dists = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (N,K)
    labels = np.argmin(dists, axis=1).astype(np.uint8)
    return labels.reshape(rgb.shape[:2])  # (H,W) uint8

def _build_4ch_rgb_plus_index_per_image(bgr: np.ndarray, K=PALETTE_K) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    idx = _index_per_image_kmeans(bgr, K=K)  # (H,W) uint8
    return np.dstack([rgb, idx])

def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def ms_edge_channel_from_bgr(bgr: np.ndarray, sigmas=MS_SIGMAS) -> np.uint8:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    acc = None
    for s in sigmas:
        blur = cv2.GaussianBlur(gray, (0, 0), s, s, borderType=cv2.BORDER_REFLECT_101) if s > 0 else gray
        mag  = _grad_mag(blur)
        acc  = mag if acc is None else np.maximum(acc, mag)
    lo, hi = np.percentile(acc, [1, 99])
    acc = np.clip((acc - lo) / max(1e-6, (hi - lo)), 0, 1)
    return (acc * 255).astype(np.uint8)

def _scharr_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def dt_edge_channel_from_bgr(
    bgr: np.ndarray,
    sigmas=MS_SIGMAS,
    bin_method: str = DT_BIN_METHOD,
    p_hi: int = DT_P_HI, p_lo: int = DT_P_LO,
    morph_open: int = DT_MORPH_OPEN
) -> np.uint8:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    acc = None
    for s in sigmas:
        blur = cv2.GaussianBlur(gray, (0,0), s, s, borderType=cv2.BORDER_REFLECT_101) if s>0 else gray
        mag  = _scharr_mag(blur)
        acc  = mag if acc is None else np.maximum(acc, mag)

    if bin_method == "otsu":
        acc8 = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(acc8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        lo, hi = np.percentile(acc, [p_lo, p_hi])
        edges = (acc >= hi).astype(np.uint8) * 255

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, k, iterations=morph_open)

    non_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)[1]
    dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3).astype(np.float32)

    lo, hi = np.percentile(dist, [1, 99])
    dist = np.clip((dist - lo) / max(1e-6, (hi - lo)), 0, 1)
    tau = 3.0  
    soft = np.exp(-dist / tau)          

    acc8 = cv2.normalize(acc, None, 0, 1, cv2.NORM_MINMAX)
    soft = 0.7 * soft + 0.3 * acc8
    soft = np.clip(soft, 0, 1)
    return (soft * 255).astype(np.uint8)

# =========================
# Utils
# =========================
def compute_angle_from_bbox(points):
    x1, y1, x2, y2, x3, y3, x4, y4 = points
    angle = np.arctan2(x4 - x1, y4 - y1) * (180.0 / np.pi)
    if angle > 0:
        angle = 180 - angle
    else:
        angle = abs(angle)
    return angle

def compute_polygon_iou(box1, box2):
    poly1 = Polygon([(box1[i], box1[i+1]) for i in range(0, 8, 2)])
    poly2 = Polygon([(box2[i], box2[i+1]) for i in range(0, 8, 2)])
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0.0

def merge_detections(detections, iou_thr=0.5, excluse_check=True):
    if not detections:
        return []
    detections.sort(key=lambda x: x[9], reverse=True)
    merged = []
    excluded_boxes = [det[:11] for det in detections if det[8] in EXCLUDED_CLASSES]

    for det1 in detections:
        box1, cls1, conf1 = det1[:8], det1[8], det1[9]
        if cls1 in EXCLUDED_CLASSES:
            continue
        keep = True

        if excluse_check:
            for det_excl in excluded_boxes:
                excl_box, excl_cls, excl_conf = det_excl[:8], det_excl[8], det_excl[9]
                iou = compute_polygon_iou(box1, excl_box)
                if iou > 0.3:
                    if conf1 > 0.85 or excl_conf < 0.5:
                        continue
                    else:
                        keep = False
                        break

        for det2 in merged:
            box2, cls2 = det2[:8], det2[8]
            if cls1 == cls2 and compute_polygon_iou(box1, box2) >= iou_thr:
                keep = False
                break

        if keep:
            merged.append(det1)

    return merged

# =========================
# Detection
# =========================
def detect_symbols(image, model, tile_size, overlap):
    h, w, _ = image.shape
    step = tile_size - overlap
    detections = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            crop_detections = []
            crop = image[y:y+tile_size, x:x+tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                continue
                
            results = run_inference_on_crop(crop, model, imgsz=tile_size)

            for box in results[0].obb:
                points = [int(v) for v in box.xyxyxyxy[0].flatten().tolist()]
                x1, y1, x2, y2, x3, y3, x4, y4 = points
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CLASS_THRESHOLDS.get(cls, 0.05):
                    continue
                angle = compute_angle_from_bbox(points) if CLASS_NAMES.get(cls) == "Strike" else 0
                crop_detections.append((x1+x, y1+y, x2+x, y2+y, x3+x, y3+y, x4+x, y4+y, cls, conf, angle))
            
            detections.extend(merge_detections(crop_detections, iou_threshold))
            
    return detections

def process_image(image_path, output_dir):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    all_detections = []
    for tile_size, overlap, model in zip(tile_sizes, overlaps, models):
        all_detections.extend(detect_symbols(image, model, tile_size, overlap))

    print(f"--- {time.time() - start_time:.2f} seconds ---")

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
        pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], np.int32)
        cv2.polylines(result_image, [pts], isClosed=True, color=color, thickness=2)
        tx, ty = min(x1,x2,x3,x4), min(y1,y2,y3,y4)-10
        cv2.putText(result_image, f"{label} {conf:.2f}", (tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        data.append([label, x1,y1,x2,y2,x3,y3,x4,y4,conf,angle])

    out_path = os.path.join(output_dir, image_name.replace(".jpg","_detected.jpg").replace(".png","_detected.jpg"))
    cv2.imwrite(out_path, result_image)

    df = pd.DataFrame(data, columns=["Class","X1","Y1","X2","Y2","X3","Y3","X4","Y4","Confidence","Angle"])
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

def _evaluate_dataset(all_images, conf_thr=0.25, iou_thr=0.5):
    tot_tp = tot_fp = tot_fn = 0
    for img_path in all_images:
        dets_all = all_dets_per_image.get(img_path, [])
        try:
            filtered = [d for d in dets_all if (d[9] >= conf_thr and (d[8] not in EXCLUDED_CLASSES))]
        except NameError:
            filtered = [d for d in dets_all if d[9] >= conf_thr]
        gts = _load_gt_as_pixels(img_path)
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

def _classwise_report(all_images, conf_thr=0.25, iou_thr=0.5):
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

def run_fusion_eval(input_dir, iou_thr=0.5):
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
    if not all_images:
        print("[Eval] No images found for evaluation.")
        return

    print(f"Channels: {channels}, APPLY_FILTER_3CH: {APPLY_FILTER_3CH}")
    print(f"Tile size: {tile_sizes}, Overlap: {overlaps}")    
    best = _find_best_conf_threshold(all_images, iou_thr=iou_thr)
    print(f"[Fusion] Best confidence threshold (by F1): {best['thr']:.2f} | P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f}")
    P, R, F1 = _evaluate_dataset(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    print(f"[Fusion @ {best['thr']:.2f}] Precision={P:.3f} | Recall={R:.3f} | F1={F1:.3f}")
    _classwise_report(all_images, conf_thr=best['thr'], iou_thr=iou_thr)
    
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
        run_fusion_eval(input_dir, iou_thr=iou_threshold)
    except Exception as e:
        print(f"[Eval] Skipped due to error: {e}")
