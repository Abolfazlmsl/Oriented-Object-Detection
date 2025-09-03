# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 18:29:09 2025

6-Channel YOLO11-OBB training:
- Tile into crops + OBB label adjust
- Build 6ch per-tile at load time: [R,G,B,L_CLAHE,edge_soft,L_NLM]
- Patch first Conv to in_channels=6 and train

@author: amoslemi
"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb import OBBTrainer
import sys, subprocess, inspect

# ==============================
# Config
# ==============================
# Tiling / dataset
TILE_SIZE = 128
OVERLAP = 50
NEED_CROPPING = False
NEED_AUGMENTATION = False
Dual_GPU = True

# Training
EPOCHS = 1
BATCH_SIZE = 32
WORKERS = 2
CACHE = False       
RECT = False  
if Dual_GPU:     
    DEVICE = "0,1" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "0" if torch.cuda.is_available() else "cpu"

# Balancing
CLASS_BALANCE_THRESHOLD = 400
AUGMENTATION_REPEATS = 2

# Paths (match your structure)
TRAIN_IMG_DIR = "datasets/GeoMap/images/train"
TRAIN_LBL_DIR = "datasets/GeoMap/labels/train"
TRAIN_OUT_IMG_DIR = "datasets/GeoMap/cropped/images/train"
TRAIN_OUT_LBL_DIR = "datasets/GeoMap/cropped/labels/train"
TRAIN_LIST = "datasets/GeoMap/train.txt"
TRAIN_LIST_CROPPED = "datasets/GeoMap/train_cropped.txt"

VAL_IMG_DIR = "datasets/GeoMap/images/val"
VAL_LBL_DIR = "datasets/GeoMap/labels/val"
VAL_OUT_IMG_DIR = "datasets/GeoMap/cropped/images/val"
VAL_OUT_LBL_DIR = "datasets/GeoMap/cropped/labels/val"
VAL_LIST = "datasets/GeoMap/val.txt"
VAL_LIST_CROPPED = "datasets/GeoMap/val_cropped.txt"

DATA_YAML = "datasets/GeoMap/data.yaml"   
PRETRAINED = "yolo11x-obb.pt"              
    
# ==============================
# Helpers
# ==============================
def write_list(txt_file, paths):
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    with open(txt_file, "w") as f:
        for p in paths:
            f.write(p + "\n")

def append_list(txt_file, paths):
    with open(txt_file, "a") as f:
        for p in paths:
            f.write(p + "\n")


# ==============================
# Tiling + OBB label adjust 
# ==============================
def crop_images_and_labels(image_dir, label_dir, out_img_dir, out_lbl_dir, list_out,
                           tile_size=512, overlap=0):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    new_paths = []

    step = tile_size - overlap

    for image_file in image_files:
        ipath = os.path.join(image_dir, image_file)
        bgr = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Cannot read image: {image_file}")
            continue

        h, w = bgr.shape[:2]
        label_file = os.path.splitext(image_file)[0] + ".txt"
        lpath = os.path.join(label_dir, label_file)
        if not os.path.exists(lpath):
            print(f"[WARN] Label not found: {label_file}")
            continue

        df = pd.read_csv(lpath, sep=" ", header=None)
        df.columns = ["class", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        # denorm to pixels
        for col in ["x1","x2","x3","x4"]:
            df[col] *= w
        for col in ["y1","y2","y3","y4"]:
            df[col] *= h

        tid = 0
        # centroid-based selection (x1,y1 to x4,y4 ordering assumed consistent)
        cx = (df["x1"] + df["x3"]) / 2.0
        cy = (df["y1"] + df["y3"]) / 2.0

        for y in range(0, h, step):
            for x in range(0, w, step):
                crop = bgr[y:y+tile_size, x:x+tile_size]
                if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                    continue

                mask = (cx >= x) & (cx < x + tile_size) & \
                       (cy >= y) & (cy < y + tile_size)
                tl = df[mask].copy()
                if tl.empty:
                    continue

                # shift to tile coords
                for col in ["x1","x2","x3","x4"]:
                    tl[col] -= x
                for col in ["y1","y2","y3","y4"]:
                    tl[col] -= y

                # clip to tile
                for col in ["x1","x2","x3","x4"]:
                    tl[col] = tl[col].clip(0, tile_size)
                for col in ["y1","y2","y3","y4"]:
                    tl[col] = tl[col].clip(0, tile_size)

                # normalize to [0,1]
                for col in ["x1","x2","x3","x4"]:
                    tl[col] /= tile_size
                for col in ["y1","y2","y3","y4"]:
                    tl[col] /= tile_size

                # save
                base = f"{os.path.splitext(image_file)[0]}_tile_{tid}"
                img_out = os.path.join(out_img_dir, base + ".jpg")
                lbl_out = os.path.join(out_lbl_dir, base + ".txt")
                cv2.imwrite(img_out, crop)
                tl.to_csv(lbl_out, sep=" ", header=False, index=False)

                new_paths.append(img_out)
                tid += 1

        print(f"[INFO] Tiled: {image_file}")

    write_list(list_out, new_paths)


# ==============================
# Simple per-class balancing 
# ==============================
def apply_single_class_augmentation(image, labels):
    height, width = image.shape[:2]
    aug_results = []
    x_cols = [1,3,5,7]
    y_cols = [2,4,6,8]

    def dup(df): return df.copy()

    # Scale 1.2x
    scaled = cv2.resize(image, (int(width*1.2), int(height*1.2)))
    sl = dup(labels)
    sl[x_cols] *= width; sl[y_cols] *= height
    sl[x_cols] *= 1.2;   sl[y_cols] *= 1.2
    nw, nh = scaled.shape[1], scaled.shape[0]
    sl[x_cols] /= nw; sl[y_cols] /= nh
    sl.iloc[:,1:] = sl.iloc[:,1:].clip(0,1)
    aug_results.append(("scale", scaled, sl))

    # Shift
    sx = random.randint(-40, 40); sy = random.randint(-40, 40)
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(image, M, (width, height))
    sh = dup(labels)
    sh[x_cols] *= width; sh[y_cols] *= height
    sh[x_cols] += sx;    sh[y_cols] += sy
    sh[x_cols] /= width; sh[y_cols] /= height
    sh.iloc[:,1:] = sh.iloc[:,1:].clip(0,1)
    aug_results.append(("shift", shifted, sh))

    # HSV jitter
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:,:,1] *= random.uniform(0.7, 1.3)
    hsv[:,:,2] *= random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hs = dup(labels)
    hs.iloc[:,1:] = hs.iloc[:,1:].clip(0,1)
    aug_results.append(("hsv", hsv, hs))

    return aug_results

def balance_classes(image_dir, label_dir, list_file, class_balance_threshold=100, augmentation_repeats=5):
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    counts = {}
    for lf in label_files:
        df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
        for cid in df[0]:
            counts[cid] = counts.get(cid, 0) + 1
    print(f"[INFO] Initial class distribution: {counts}")

    new_paths = []
    uid = 0
    for cid, cnt in counts.items():
        if cnt >= class_balance_threshold:
            continue
        print(f"[INFO] Balancing class {cid} (count={cnt})")
        files_with = []
        for lf in label_files:
            df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
            if cid in df[0].values:
                files_with.append(lf)

        for _ in range(augmentation_repeats):
            for lf in files_with:
                ip = os.path.join(image_dir, lf.replace(".txt",".jpg"))
                img = cv2.imread(ip)
                if img is None: continue
                df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
                aug_list = apply_single_class_augmentation(img, df)
                for a_type, a_img, a_lbl in aug_list:
                    base = os.path.splitext(lf)[0]
                    out_img = os.path.join(image_dir, f"{base}_aug_{a_type}_{uid}.jpg")
                    out_lbl = os.path.join(label_dir, f"{base}_aug_{a_type}_{uid}.txt")
                    cv2.imwrite(out_img, a_img)
                    a_lbl.to_csv(out_lbl, sep=" ", header=False, index=False)
                    new_paths.append(out_img)
                    uid += 1

    append_list(list_file, new_paths)

    # report final counts
    counts = {}
    for lf in os.listdir(label_dir):
        if not lf.endswith(".txt"): continue
        df = pd.read_csv(os.path.join(label_dir, lf), sep=" ", header=None)
        for cid in df[0]:
            counts[cid] = counts.get(cid, 0) + 1
    print(f"[INFO] Balanced class distribution: {counts}")


# ==============================
# Edge & channel builders (used AFTER tiling, inside dataset)
# ==============================
def _norm01(x, eps=1e-6):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return np.zeros_like(x, dtype=np.float32) if mx - mn < eps else (x - mn) / (mx - mn)

def soft_edge_from_L(L_uint8, method="scharr", pre_blur_sigma=0.6):
    img = L_uint8
    if pre_blur_sigma > 0:
        k = int(2 * round(3 * pre_blur_sigma) + 1)
        img = cv2.GaussianBlur(img, (k, k), pre_blur_sigma)

    if method == "scharr":
        gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
        mag = np.abs(gx) + np.abs(gy)
        return _norm01(mag)
    elif method == "morphgrad":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dil = cv2.dilate(img, k, iterations=1)
        ero = cv2.erode(img, k, iterations=1)
        return _norm01(cv2.subtract(dil, ero))
    else:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return _norm01(np.sqrt(gx*gx + gy*gy))


# ==============================
# 6-Channel Dataset (build channels per tile)
# ==============================
class MultiChannelYOLODataset(YOLODataset):
    """YOLODataset that returns HxWx6 uint8 images (pads/trims channels as needed)."""

    def _ensure_six(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3:
            raise ValueError(f"Unexpected image ndim={img.ndim}; expected 2 or 3.")
        h, w, c = img.shape
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if c == 6:
            return img
        if c > 6:
            return img[:, :, :6]
        pad = np.zeros((h, w, 6 - c), dtype=np.uint8)
        return np.concatenate([img, pad], axis=2)

    def load_image(self, i: int):
        p = self.im_files[i]
        im = cv2.imread(p, cv2.IMREAD_COLOR)  # force 3ch
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        h0, w0 = im.shape[:2]
    
        # Build extra channels
        L = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # CLAHE on L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        L_clahe = clahe.apply(L)
        # Soft edges on L
        edge = (soft_edge_from_L(L, method="scharr") * 255).astype(np.uint8)
        # Denoised L
        L_nlm = cv2.fastNlMeansDenoising(L, None, 10, 7, 21)
    
        six = np.dstack([im, L_clahe, edge, L_nlm])
        if not six.flags["C_CONTIGUOUS"]:
            six = np.ascontiguousarray(six)
        return six, (h0, w0), six.shape[:2]

def patch_first_conv(model: nn.Module, in_ch: int = 6) -> bool:
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3 and m.groups == 1:
            old = m
            dev = old.weight.device
            dt  = old.weight.dtype

            new = nn.Conv2d(
                in_ch, old.out_channels, old.kernel_size, old.stride,
                old.padding, old.dilation, groups=1, bias=(old.bias is not None)
            ).to(device=dev, dtype=dt)

            with torch.no_grad():
                reps = in_ch // 3
                rem  = in_ch % 3
                inflated = old.weight.detach().clone().repeat(1, reps, 1, 1)
                if rem:
                    inflated = torch.cat([inflated, old.weight[:, :rem]], dim=1)
                inflated *= 3.0 / float(in_ch)
                new.weight.copy_(inflated.to(dev, dt))
                if old.bias is not None:
                    new.bias.copy_(old.bias.detach().to(dev, dt))

            parent = model
            *parents, last = name.split(".")
            for p in parents:
                parent = getattr(parent, p)
            setattr(parent, last, new)

            print(f"[INFO] Patched first Conv2d: in_channels 3 → {in_ch} ✔ ({name}) on {dev}, {dt}")
            return True
    print("[WARN] Could not find a 3-channel Conv2d to patch.")
    return False



# =========================
# Trainer: OBB + 6-channel safe
# =========================
class MultiChannelTrainer(OBBTrainer):
    """OBB trainer configured for 6-channel inputs and safe augments."""

    def final_eval(self):
        print("[INFO] Skipping post-training final_eval() to avoid post-epoch errors.")
        return
    
    # turn off color & multi-image augments that assume 3 channels
    def build_dataset(self, img_path, mode="train", batch=None):
        self.args.task = "obb"
        self.args.hsv_h = self.args.hsv_s = self.args.hsv_v = 0.0
        self.args.mosaic = 0.0
        self.args.mixup = 0.0
        self.args.copy_paste = 0.0
        self.args.plots = False

        return MultiChannelYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=self.args.batch if batch is None else batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        try:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose, in_channels=6)
            print("[check] get_model accepted in_channels=6")
        except TypeError:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            print("[check] get_model does NOT accept in_channels; will patch Conv.")
    
        try:
            first = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
            print(f"[check] first Conv2d in_channels BEFORE patch = {first.in_channels}")
        except StopIteration:
            print("[check] No Conv2d found for sanity check.")
    
        ok = patch_first_conv(model, in_ch=6)
        print(">>> PATCH MAIN RESULT =", ok)
        dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device={dev}, RANK={os.getenv('RANK')}, LOCAL_RANK={os.getenv('LOCAL_RANK')}")
        
        return model


def on_fit_start_cb(trainer):
    if getattr(trainer, "ema", None) and getattr(trainer.ema, "ema", None): 
        try:
            patch_first_conv(trainer.ema.ema, in_ch=6)
        except Exception:
            pass
        
def on_preprocess_batch_end(trainer, batch):
    imgs = batch.get("img", None)
    if imgs is not None and imgs.dtype != torch.float32:
        batch["img"] = imgs.float()

def on_val_batch_start(*args, **kwargs):
    if len(args) >= 3:
        _, batch, _ = args[:3]
    elif len(args) == 2:
        _, batch = args
    else:
        batch = kwargs.get("batch")

    import torch
    if isinstance(batch, dict):
        x = batch.get("img", None) or batch.get("imgs", None)
        if isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] == 3:
            batch["img"] = torch.cat([x, x], dim=1)[:, :6]
            
# ==============================
# Main
# ==============================
if __name__ == "__main__":
            
    if Dual_GPU:
        if os.getenv("RANK") is None:  
            DEV = "0,1"  
            if "," in DEV:  
                nproc = len(DEV.split(","))
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEV)
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29501")  
                os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
                os.environ.setdefault("NCCL_IB_DISABLE", "1")
                os.environ.setdefault("NCCL_P2P_DISABLE", "1")
                this = os.path.abspath(inspect.getfile(sys.modules[__name__]))
                cmd = [sys.executable, "-m", "torch.distributed.run",
                       "--nproc_per_node", str(nproc),
                       "--master_port", os.environ["MASTER_PORT"],
                       this]
                raise SystemExit(subprocess.run(cmd, check=True).returncode)
            
    # 1) Tiling 
    if NEED_CROPPING:
        crop_images_and_labels(
            image_dir=TRAIN_IMG_DIR,
            label_dir=TRAIN_LBL_DIR,
            out_img_dir=TRAIN_OUT_IMG_DIR,
            out_lbl_dir=TRAIN_OUT_LBL_DIR,
            list_out=TRAIN_LIST_CROPPED,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )
        crop_images_and_labels(
            image_dir=VAL_IMG_DIR,
            label_dir=VAL_LBL_DIR,
            out_img_dir=VAL_OUT_IMG_DIR,
            out_lbl_dir=VAL_OUT_LBL_DIR,
            list_out=VAL_LIST_CROPPED,
            tile_size=TILE_SIZE,
            overlap=OVERLAP
        )

    # 2) Class balancing on CROPPED tiles
    if NEED_AUGMENTATION:
        balance_classes(
            image_dir=TRAIN_OUT_IMG_DIR,
            label_dir=TRAIN_OUT_LBL_DIR,
            list_file=TRAIN_LIST_CROPPED,
            class_balance_threshold=CLASS_BALANCE_THRESHOLD,
            augmentation_repeats=AUGMENTATION_REPEATS
        )
        balance_classes(
            image_dir=VAL_OUT_IMG_DIR,
            label_dir=VAL_OUT_LBL_DIR,
            list_file=VAL_LIST_CROPPED,
            class_balance_threshold=CLASS_BALANCE_THRESHOLD,
            augmentation_repeats=AUGMENTATION_REPEATS
        )

    overrides = dict(
        model=PRETRAINED,
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
        task='obb',
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        amp=False
    )
    
    trainer = MultiChannelTrainer(overrides=overrides)
    trainer.add_callback("on_fit_start", on_fit_start_cb)
    trainer.add_callback("on_preprocess_batch_end", on_preprocess_batch_end)
    trainer.add_callback("on_val_batch_start", on_val_batch_start)
    trainer.train()
