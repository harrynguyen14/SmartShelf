"""Danh gia YOLO11n da train + export NCNN cho Raspberry Pi.

Metric (Ultralytics tu tinh): mAP50, mAP50-95, precision, recall.
Voi bai dem san pham + suy ke trong, RECALL la quan trong nhat (miss item = sai).

Chay:  python smartshelf/validate.py --weights runs_smartshelf/yolo11n_sku110k/weights/best.pt
"""
import argparse
import os
from ultralytics import YOLO

SRC = r"D:/industry-item-dataset/SKU110K/SKU110K_fixed"
DATA = os.path.join(SRC, "yolo", "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt sau train")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--export-ncnn", action="store_true", help="export NCNN cho Pi")
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=DATA, split=args.split, imgsz=args.imgsz, max_det=600)

    b = m.box
    print("\n=== Metric (", args.split, ") ===")
    print(f"  mAP50     : {b.map50:.4f}")
    print(f"  mAP50-95  : {b.map:.4f}")
    print(f"  precision : {b.mp:.4f}")
    print(f"  recall    : {b.mr:.4f}   <- quan trong nhat (miss item)")

    if args.export_ncnn:
        path = model.export(format="ncnn", imgsz=args.imgsz)
        print(f"\nNCNN -> {path}  (copy sang Pi)")


if __name__ == "__main__":
    main()
