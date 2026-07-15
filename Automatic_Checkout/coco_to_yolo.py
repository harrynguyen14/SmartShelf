"""Convert D2S (COCO RLE instance-seg) -> YOLO-seg dataset.

D2S masks are compressed-RLE, YOLO-seg wants normalized polygons, so we
decode each RLE and trace its contours. Categories are contiguous ids 1..60,
so the YOLO class index is simply category_id - 1.

Outputs under --out:
  images/{train,val}/*.jpg   (copied)
  labels/{train,val}/*.txt   (YOLO-seg: "cls x1 y1 x2 y2 ...", normalized)
  data.yaml                  (ready for `yolo train data=...`)
  supercategories.json       (class-index -> {name, supercategory})

Kaggle example:
  python coco_to_yolo.py --d2s-root /kaggle/input/<d2s-slug>/D2S \
                         --out /kaggle/working/d2s_yolo
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils

# ponytail: approxPolyDP epsilon as a fraction of contour perimeter; smaller = more
# points/tighter masks. 0.002 keeps curved produce (bananas, apples) faithful.
APPROX_EPS_FRAC = 0.002
MIN_CONTOUR_AREA = 20  # px^2, drops decode speckle


def rle_to_polygons(seg, h, w):
    """Decode one COCO segmentation to a list of normalized polygons."""
    if isinstance(seg, list):  # already polygon(s)
        rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):  # uncompressed RLE
        rle = maskUtils.frPyObjects(seg, h, w)
    else:  # compressed RLE (D2S case) — counts is bytes/str
        rle = seg
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        eps = APPROX_EPS_FRAC * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        norm = c.astype(np.float64)
        norm[:, 0] /= w
        norm[:, 1] /= h
        polys.append(norm.reshape(-1).clip(0, 1))
    return polys


def convert_split(coco_path, images_src, out, split, limit=0):
    coco = json.load(open(coco_path, encoding="utf-8"))
    if limit:
        coco["images"] = coco["images"][:limit]
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    n_img = n_inst = n_poly = 0
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        h, w = im["height"], im["width"]
        lines = []
        for a in anns:
            cls = a["category_id"] - 1  # ids are 1..60 -> 0..59
            for p in rle_to_polygons(a["segmentation"], h, w):
                lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in p))
                n_poly += 1
            n_inst += 1
        stem = Path(im["file_name"]).stem
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        dst = img_dir / im["file_name"]
        if not dst.exists():
            shutil.copy2(images_src / im["file_name"], dst)  # ponytail: copy = portable local/Kaggle; symlink if disk-bound
        n_img += 1
    print(f"[{split}] {n_img} images, {n_inst} instances, {n_poly} polygons")
    return coco["categories"]


def write_meta(out, categories):
    cats = sorted(categories, key=lambda c: c["id"])  # -> index = id-1
    # Class name IS the hierarchy label, so the model predicts it directly (no lookup).
    names = [f"{c['supercategory']}: {c['name']}" for c in cats]

    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val",
            f"nc: {len(names)}", "names:"]
    yaml += [f'  {i}: "{n}"' for i, n in enumerate(names)]  # quote: names contain ": "
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"wrote data.yaml ({len(names)} classes, names = 'supercategory: product')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2s-root", default=r"D:\industry-item-dataset\D2S")
    ap.add_argument("--out", default="d2s_yolo")
    ap.add_argument("--train-json", default="D2S_training.json")
    ap.add_argument("--val-json", default="D2S_validation.json")
    ap.add_argument("--limit", type=int, default=0, help="cap images/split for a smoke test (0 = all)")
    args = ap.parse_args()

    root = Path(args.d2s_root)
    ann = root / "annotations"
    images_src = root / "images"
    out = Path(args.out)

    cats = convert_split(ann / args.train_json, images_src, out, "train", args.limit)
    convert_split(ann / args.val_json, images_src, out, "val", args.limit)
    write_meta(out, cats)


if __name__ == "__main__":
    main()
