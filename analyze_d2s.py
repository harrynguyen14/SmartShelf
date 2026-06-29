"""Analyze the D2S dataset (COCO-format annotations)."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\industry-item-dataset\D2S")
ANN = ROOT / "annotations"
IMAGES = ROOT / "images"

# The COCO json splits worth reporting on (skip the *_info.json image-only files).
SPLITS = [
    "D2S_training.json",
    "D2S_validation.json",
    "D2S_augmented.json",
]


def analyze(json_path: Path):
    d = json.loads(json_path.read_text())
    imgs, anns, cats = d["images"], d["annotations"], d["categories"]
    id2name = {c["id"]: c["name"] for c in cats}

    per_img = Counter(a["image_id"] for a in anns)
    per_cat = Counter(id2name.get(a["category_id"], a["category_id"]) for a in anns)
    widths = [im["width"] for im in imgs]
    heights = [im["height"] for im in imgs]
    obj_counts = [per_img.get(im["id"], 0) for im in imgs]

    print(f"\n=== {json_path.name} ===")
    print(f"  images:       {len(imgs)}")
    print(f"  annotations:  {len(anns)}")
    print(f"  categories:   {len(cats)}")
    print(f"  resolution:   {set(zip(widths, heights))}")
    print(f"  objects/img:  min={min(obj_counts)} max={max(obj_counts)} "
          f"avg={sum(obj_counts)/len(obj_counts):.1f}")
    print("  top 5 categories by instance count:")
    for name, n in per_cat.most_common(5):
        print(f"    {n:6d}  {name}")
    print("  bottom 5 categories:")
    for name, n in per_cat.most_common()[-5:]:
        print(f"    {n:6d}  {name}")


if __name__ == "__main__":
    n_img_files = sum(1 for _ in IMAGES.glob("*.jpg")) if IMAGES.exists() else 0
    print(f"D2S image files on disk: {n_img_files}")
    for s in SPLITS:
        p = ANN / s
        if p.exists():
            analyze(p)
        else:
            print(f"  (missing: {s})")
