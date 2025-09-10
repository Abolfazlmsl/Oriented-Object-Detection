# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:15:12 2025

@author: amoslemi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

# ====== Config ======
CSV_PATH = "results.csv"      # path to results.csv
OUT_DIR  = "Train Curves"     # output folder
SMOOTH_WINDOW = 0             # moving average window

# Different line styles and markers for distinction in B/W printing
LINESTYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+']

def moving_average(series: pd.Series, w: int) -> pd.Series:
    if w is None or w <= 1:
        return series
    return series.rolling(window=w, min_periods=1).mean()

def detect_epoch_column(columns):
    candidates = ["epoch", "Epoch", "epochs", "step", "Step", "iter", "iteration"]
    for c in candidates:
        if c in columns:
            return c
    return None

def plot_group(df, epoch_col, columns, title, filename, out_dir, smooth=0):
    if not columns:
        return None
    plt.figure(figsize=(10, 6), dpi=140)
    x = df[epoch_col]
    for i, c in enumerate(columns):
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        y = moving_average(df[c].astype(float), smooth)
        ls = LINESTYLES[i % len(LINESTYLES)]
        mk = MARKERS[i % len(MARKERS)]
        plt.plot(x, y, linestyle=ls, marker=mk, markevery=max(1, len(x)//20), label="\n".join(wrap(c, 40)))
    plt.xlabel("Epoch" if "epoch" in epoch_col.lower() else epoch_col)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    epoch_col = detect_epoch_column(df.columns)
    if epoch_col is None:
        epoch_col = "epoch"
        df[epoch_col] = np.arange(len(df), dtype=int)

    df = df.dropna(how="all").reset_index(drop=True)

    loss_cols   = [c for c in df.columns if "loss" in c.lower() and c != epoch_col]
    lr_cols     = [c for c in df.columns if c.lower().startswith("lr") or "lr/" in c.lower() or c.lower() == "lr"]
    metric_cols = [c for c in df.columns if any(k in c.lower() for k in ["map", "precision", "recall", "f1", "acc"])]
    time_cols   = [c for c in df.columns if any(k in c.lower() for k in ["time", "t/"])]

    exclude = set([epoch_col] + loss_cols + lr_cols + metric_cols + time_cols)
    extra_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    groups = [
        ("Training Losses", loss_cols, "losses.png"),
        ("Validation/Test Metrics", metric_cols, "metrics.png"),
        ("Learning Rates", lr_cols, "lrs.png"),
        ("Timing (per epoch)", time_cols, "timings.png"),
        ("Other Numeric Metrics", extra_cols, "others.png"),
    ]

    saved = []
    for title, cols, fname in groups:
        path = plot_group(df, epoch_col, cols, title, fname, OUT_DIR, smooth=SMOOTH_WINDOW)
        if path:
            saved.append(path)

    print("\n[OK] Plots saved in:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()

