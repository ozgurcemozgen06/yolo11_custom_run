# eval_test.py
from ultralytics import YOLO
import pandas as pd
import os

proj = "results"
name = "cli_train_fast"
yaml = "/Users/ozgurcem/yolo11_custom/custom_dataset.yaml"

m = YOLO(f"{proj}/{name}/weights/best.pt")
res = m.val(data=yaml, split="test", imgsz=512, device="cpu", workers=0,
            project=proj, name=f"{name}_test")

outdir = f"{proj}/{name}"
os.makedirs(outdir, exist_ok=True)

df = pd.DataFrame([res.results_dict])
df.to_csv(f"{outdir}/test_metrics.csv", index=False)
print("Saved:", f"{outdir}/test_metrics.csv")
