"""Convert SKU-110K (SKU110K_fixed) into a standard HuggingFace object-detection
dataset and push it to the Hub.

Source layout
-------------
    SKU110K_fixed/
        images/                       train_*.jpg, val_*.jpg, test_*.jpg  (11,743)
        annotations/
            annotations_train.csv     one row PER BOX:
            annotations_val.csv         image_name,x1,y1,x2,y2,class,img_w,img_h
            annotations_test.csv

Output schema (one row PER IMAGE) — the COCO-style layout HF detection models
(DETR, etc.) expect, matching datasets like `cppe-5`:

    image       : datasets.Image()           # decoded PIL image
    image_id    : int                         # stable per-image index
    width,height: int
    objects : {
        id        : list[int]                 # per-box running id
        bbox      : list[[x, y, w, h]]        # COCO xywh, absolute pixels
        category  : list[int]                 # always 0 (single class)
        area      : list[float]               # w * h
    }

Only one category exists in SKU-110K: "object".

Usage
-----
    python sku110k_to_hf.py --root D:/industry-item-dataset/SKU110K/SKU110K_fixed \
        --repo <hf-username>/sku110k --token $HF_TOKEN

    # local only (no upload), saves Arrow shards:
    python sku110k_to_hf.py --root ... --out ./sku110k_hf

Needs only `datasets` + `Pillow` (no torch). Heavy step is uploading ~13 GB of
images; run where the images + bandwidth live (e.g. Kaggle).
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Image as HFImage,
    Sequence,
    Value,
)

CATEGORIES = ["object"]  # SKU-110K is single-class

SPLITS = {
    "train": "annotations_train.csv",
    "validation": "annotations_val.csv",
    "test": "annotations_test.csv",
}

FEATURES = Features(
    {
        "image": HFImage(),
        "image_id": Value("int64"),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence(
            {
                "id": Value("int64"),
                "bbox": Sequence(Value("float32"), length=4),  # COCO xywh
                "category": Value("int64"),
                "area": Value("float32"),
            }
        ),
    }
)


def parse_csv(csv_path: str) -> Dict[str, dict]:
    """Group per-box rows into per-image records. Clips degenerate
    (x2<=x1 or y2<=y1) boxes; skips the header / malformed rows."""
    by_image: Dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) != 8:
                continue
            name, x1, y1, x2, y2, _cls, w, h = row
            try:
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w, h = int(w), int(h)
            except ValueError:
                continue  # header row ("x1", ...) or junk
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            rec = by_image.setdefault(
                name, {"width": w, "height": h, "bbox": [], "area": []}
            )
            rec["bbox"].append([x1, y1, bw, bh])  # xywh
            rec["area"].append(bw * bh)
    return by_image


def build_split(root: str, csv_name: str) -> Dataset:
    images_dir = os.path.join(root, "images")
    by_image = parse_csv(os.path.join(root, "annotations", csv_name))

    rows: List[dict] = []
    box_id = 0
    for image_id, (name, rec) in enumerate(sorted(by_image.items())):
        path = os.path.join(images_dir, name)
        if not os.path.isfile(path):
            continue
        n = len(rec["bbox"])
        ids = list(range(box_id, box_id + n))
        box_id += n
        rows.append(
            {
                "image": path,  # HFImage() loads from this path on cast
                "image_id": image_id,
                "width": rec["width"],
                "height": rec["height"],
                "objects": {
                    "id": ids,
                    "bbox": rec["bbox"],
                    "category": [0] * n,
                    "area": rec["area"],
                },
            }
        )

    return Dataset.from_list(rows, features=FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="path to SKU110K_fixed")
    ap.add_argument("--repo", help="HF Hub repo id, e.g. user/sku110k")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--out", help="local dir for save_to_disk (instead of push)")
    ap.add_argument("--private", action="store_true", help="private Hub repo")
    args = ap.parse_args()

    if not args.repo and not args.out:
        ap.error("give --repo (push) or --out (local save)")

    dd = DatasetDict()
    for split, csv_name in SPLITS.items():
        print(f"building {split} from {csv_name} ...")
        dd[split] = build_split(args.root, csv_name)
        print(f"  {split}: {len(dd[split])} images")

    if args.out:
        dd.save_to_disk(args.out)
        print(f"saved to {args.out} -> datasets.load_from_disk('{args.out}')")

    if args.repo:
        print(f"pushing to {args.repo} (uploads images — the slow part) ...")
        dd.push_to_hub(args.repo, token=args.token, private=args.private)
        print(f"done -> load_dataset('{args.repo}')")


if __name__ == "__main__":
    main()
