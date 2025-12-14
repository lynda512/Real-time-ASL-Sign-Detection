import os
import random
import shutil
from pathlib import Path

# -------- CONFIG --------
ROOT = Path("src/gesture_active_learning/data/ASL.v1i.yolov8")
OUT  = Path("src/gesture_active_learning/data/ASL_small.yolov8")
FRACTION = 0.2      # keep 20% of images; change to 0.1, 0.05, ...
RANDOM_SEED = 42
# ------------------------

random.seed(RANDOM_SEED)

def make_subset(split):
    in_img_dir = ROOT / split / "images"
    in_lbl_dir = ROOT / split / "labels"

    out_img_dir = OUT / split / "images"
    out_lbl_dir = OUT / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in in_img_dir.iterdir()
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    keep_n = max(1, int(len(image_files) * FRACTION))
    keep = random.sample(image_files, keep_n)

    for img_path in keep:
        # copy image
        shutil.copy2(img_path, out_img_dir / img_path.name)

        # copy matching label if it exists
        lbl_name = img_path.with_suffix(".txt").name
        lbl_path = in_lbl_dir / lbl_name
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl_dir / lbl_name)

    print(f"{split}: kept {keep_n} / {len(image_files)} images")

for split in ["train", "valid", "test"]:
    make_subset(split)

print("Done. Small dataset at:", OUT)
