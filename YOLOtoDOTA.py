#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 09:00:58 2025

@author: abolfazl
"""

import os
from glob import glob
import cv2

data = "val" # or val

YOLO_LABEL_DIR = f"datasets/GeoMap/cropped/labels/{data}"  # or val
IMAGE_DIR = f"datasets/GeoMap/cropped/images/{data}"        # or val

DOTA_LABEL_OUT = f"mmdata/GeoMap/{data}/labelTxt"           # or val/labelTxt
DOTA_IMAGE_OUT = f"mmdata/GeoMap/{data}/images"             # or val/images

CLASS_NAMES = {
    0: "Landslide1",
    1: "Strike",
    2: "Spring1",
    3: "Minepit1",
    4: "Hillside",
    5: "Feuchte",
    6: "Torf",
    7: "Bergsturz",
    8: "Landslide2",
    9: "Spring2",
    10: "Spring3",
    11: "Minepit2",
    12: "SpringB2",
    13: "HillsideB2"
}

os.makedirs(DOTA_LABEL_OUT, exist_ok=True)
os.makedirs(DOTA_IMAGE_OUT, exist_ok=True)

label_files = glob(os.path.join(YOLO_LABEL_DIR, "*.txt"))

for txt_path in label_files:
    basename = os.path.splitext(os.path.basename(txt_path))[0]
    image_ext = ".jpg" if os.path.exists(os.path.join(IMAGE_DIR, basename + ".jpg")) else ".png"
    image_path = os.path.join(IMAGE_DIR, basename + image_ext)

    os.system(f'cp "{image_path}" "{os.path.join(DOTA_IMAGE_OUT, basename + image_ext)}"')

    with open(txt_path, "r") as fin, open(os.path.join(DOTA_LABEL_OUT, basename + ".txt"), "w") as fout:
        for line in fin:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:9]))

            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            coords_pixel = [
                coords[i] * w if i % 2 == 0 else coords[i] * h
                for i in range(8)
            ]
            class_name = CLASS_NAMES[cls_id]
            difficult = 0  

            fout.write(" ".join(f"{x:.2f}" for x in coords_pixel) + f" {class_name} {difficult}\n")

print("Done")
