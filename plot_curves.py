# plot_curves.py
import pandas as pd
import matplotlib.pyplot as plt
import os

proj = "results"
name = "cli_train_fast"
csv_path = f"{proj}/{name}/results.csv"
outdir = f"{proj}/{name}"
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(csv_path)

def save_curve(y, ylabel, fname):
    plt.figure()
    plt.plot(y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Epoch")
    plt.grid(True)
    plt.savefig(f"{outdir}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()

mp50_cols   = [c for c in df.columns if c.lower() in ("metrics/map50", "metrics/map50", "metrics/mAP_0.5")]
mp5095_cols = [c for c in df.columns if c.lower() in ("metrics/map50-95", "metrics/map50-95", "metrics/mAP_0.5:0.95")]
prec_cols   = [c for c in df.columns if "precision" in c.lower()]
rec_cols    = [c for c in df.columns if "recall" in c.lower()]

if mp50_cols:   save_curve(df[mp50_cols[0]], "mAP@0.5", "map50_curve.png")
if mp5095_cols: save_curve(df[mp5095_cols[0]], "mAP@0.5:0.95", "map5095_curve.png")
if prec_cols:   save_curve(df[prec_cols[0]], "Precision", "precision_curve.png")
if rec_cols:    save_curve(df[rec_cols[0]], "Recall", "recall_curve.png")

print("Saved curves to:", outdir)
