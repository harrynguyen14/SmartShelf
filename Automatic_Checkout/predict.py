"""Run a trained D2S YOLO11-seg model. Class names are "supercategory: product",
so the model predicts the full hierarchy label directly (no post-processing).

  python predict.py --weights runs/d2s_seg/weights/best.pt --source path/to/img_or_dir --save
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/d2s_seg/weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true", help="save annotated images")
    args = ap.parse_args()

    model = YOLO(args.weights)
    for r in model.predict(source=args.source, conf=args.conf, save=args.save):
        print(f"\n{r.path}: {len(r.boxes)} products")
        for b in r.boxes:
            print(f"  {model.names[int(b.cls)]}  conf={float(b.conf):.2f}")


if __name__ == "__main__":
    main()
