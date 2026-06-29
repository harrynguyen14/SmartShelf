"""Analyze the SKU110K dataset (headerless CSV bounding-box annotations)."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\industry-item-dataset\SKU110K\SKU110K_fixed")
ANN = ROOT / "annotations"
IMAGES = ROOT / "images"

# CSV columns: image_name,x1,y1,x2,y2,class,image_width,image_height
SPLITS = ["annotations_train.csv", "annotations_val.csv", "annotations_test.csv"]


def analyze(csv_path: Path):
    boxes_per_img = Counter()
    classes = Counter()
    n_rows = 0
    bad = 0
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) != 8:
                bad += 1
                continue
            name, x1, y1, x2, y2, cls, _, _ = row
            n_rows += 1
            boxes_per_img[name] += 1
            classes[cls] += 1

    counts = list(boxes_per_img.values())
    print(f"\n=== {csv_path.name} ===")
    print(f"  annotations:  {n_rows}" + (f"  (malformed rows skipped: {bad})" if bad else ""))
    print(f"  images:       {len(boxes_per_img)}")
    print(f"  classes:      {dict(classes)}")
    if counts:
        print(f"  boxes/img:    min={min(counts)} max={max(counts)} "
              f"avg={sum(counts)/len(counts):.1f}")


if __name__ == "__main__":
    n_img_files = sum(1 for _ in IMAGES.glob("*.jpg")) if IMAGES.exists() else 0
    print(f"SKU110K image files on disk: {n_img_files}")
    for s in SPLITS:
        p = ANN / s
        if p.exists():
            analyze(p)
        else:
            print(f"  (missing: {s})")
