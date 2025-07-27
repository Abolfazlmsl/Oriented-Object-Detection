# 🛰️ Oriented Object Detection Using YOLOv11-OBB

This project utilizes AI (YOLOv11) to identify geological symbols in images using Oriented Bounding Boxes (OBB). It includes preprocessing (cropping, augmentation), training on multiple sizes, and inference with merging strategies.

---

## 📁 Project Structure

```
├── Train_OBB.py 
├── Detect_OBB.py 
├── datasets/
│   └── GeoMaps/
│       ├── images/        # Not included - download separately
│       │   ├── train/
│       │   └── val/
│       ├── labels/ 
│       │   ├── train/
│       │   └── val/
├── best128.pt             # Trained model at 128 tile size (via Google Drive)
├── best416.pt             # Trained model at 416 tile size (via Google Drive)
├── Input/                 # Sample images for detection
├── Output/                # Saved detection results (ignored in git)
├── requirements.txt
└── README.md
```

---

## 🚀 How It Works

### Training

The pipeline:
- Crops the dataset into tiles (`128x128` and `416x416`)
- Applies data augmentation to balance classes
- Trains two models using the Ultralytics YOLOv11-OBB engine

Run training:

```bash
python Train_OBB.py
```

> Change `tile_size` and other config inside the script for crop size, training parameters and augmentation control.

---

### Detection

Runs both trained models on input images, then merges detections using IOU-based filtering and class-specific thresholds.

Run detection:

```bash
python Detect_OBB.py
```

The script reads images from `Input/` and writes results to `Output/`, including:
- Visual results (`_detected.jpg`)
- An Excel file with coordinates, class, confidence, and orientation

---

## 🧠 Sample Images

### 📥 Input Example

![Input Image](./sample_input.png)

### 🧾 Detection Output

![Output Image](./sample_output.jpg)

---

## 📦 Dataset

Access to the dataset will be granted upon reasonable request due to data sensitivity and privacy constraints.

> After obtaining and extracting the dataset, please ensure the directory structure matches the following:

```
datasets/
└── GeoMaps/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/   ✅ Included in this repo
    │   └── val/     ✅ Included in this repo
```

---

## 📦 Pretrained Models

Download the trained models below and place them in the root folder:

- 🔗 [best128.pt](https://drive.google.com/uc?export=download&id=1xAPPtuVRvJ27rRkIBvpNmngCQrKFnt1S)
- 🔗 [best416.pt](https://drive.google.com/uc?export=download&id=1zo8XXpEcBf-SbMt9A4fJ209X3zAOX8j0)

---

## 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Required:
- Python 3.12.7
- ultralytics==8.3.78
- opencv-python==4.11.0.86
- numpy==1.26.4
- shapely==2.0.7
- openpyxl==3.1.5
- pandas==2.3.1

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this code under the terms of this license.

See the [LICENSE](./LICENSE) file for full details.
