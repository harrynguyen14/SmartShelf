"""Train YOLO11-seg on the D2S YOLO-seg dataset (run coco_to_yolo.py first).

Kaggle (GPU) example:
  python train.py --data /kaggle/working/d2s_yolo/data.yaml --model yolo11s-seg.pt \
                  --epochs 100 --imgsz 960 --batch 16
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="d2s_yolo/data.yaml")
    ap.add_argument("--model", default="yolo11s-seg.pt")  # n/s/m/l/x-seg
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)  # 960 fits seg on a 15GB T4; 1280 needs P100/A100
    ap.add_argument("--batch", type=int, default=16)  # T4 @960 OK; for 1280 use bigger GPU or --batch 4
    ap.add_argument("--device", default=0)  # 0 for GPU, "cpu" otherwise
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="d2s_seg")
    # Production robustness: D2S is shot near top-down, so augment angle/orientation/
    # lighting hard. Override any knob from the CLI (e.g. --degrees 20).
    ap.add_argument("--degrees", type=float, default=25.0)   # free product rotation
    ap.add_argument("--flipud", type=float, default=0.5)     # any orientation from above
    ap.add_argument("--perspective", type=float, default=0.0005)  # tilted camera
    ap.add_argument("--scale", type=float, default=0.5)      # near/far distance
    ap.add_argument("--hsv_v", type=float, default=0.5)      # store lighting variance
    ap.add_argument("--close_mosaic", type=int, default=15)  # drop mosaic last N epochs to fine-tune real layouts
    # D2S train = 1 object/img, val = 4.4 objects/img with occlusion. copy_paste pastes
    # instances into scenes -> simulates the crowded multi-object val, the main recall gap.
    ap.add_argument("--copy_paste", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=50)  # early-stop after 50 epochs w/o val improvement
    args = ap.parse_args()

    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        degrees=args.degrees, flipud=args.flipud, perspective=args.perspective,
        scale=args.scale, hsv_v=args.hsv_v, close_mosaic=args.close_mosaic,
        copy_paste=args.copy_paste, patience=args.patience,
    )


if __name__ == "__main__":
    main()
