# D2S Fine-grained Automatic Checkout (Instance Segmentation)

YOLO11-seg segments each product and predicts one of **60 fine-grained classes**.
Each class name is `"supercategory: product"` (e.g. `tea: gepa_bio_und_fair_kamillentee`),
so the model outputs the full hierarchy label **directly** — on the annotated image
and in results — with no post-processing.

> Flat 60-class model, hierarchy baked into the class names. Each product maps to
> exactly one supercategory, so a separate two-stage head buys nothing here.

## Pipeline (run on Kaggle GPU)

```bash
pip install -r requirements.txt

# 1. Convert D2S (COCO RLE) -> YOLO-seg. Decodes RLE -> polygons, copies images,
#    writes data.yaml (class names = "supercategory: product"). --limit 20 for a smoke test.
python coco_to_yolo.py --d2s-root /kaggle/input/<d2s-slug>/D2S --out /kaggle/working/d2s_yolo

# 2. Train
python train.py --data /kaggle/working/d2s_yolo/data.yaml --model yolo11s-seg.pt \
                --epochs 100 --imgsz 960 --batch 16 --device 0

# 3. Predict (labels are "supercategory: product" natively)
python predict.py --weights runs/d2s_seg/weights/best.pt --source some_image.jpg --save
```

## Notes
- D2S splits used: `D2S_training.json` (train, 4380 imgs) / `D2S_validation.json` (val, 3600 imgs).
- Images are 1920×1440; `imgsz=960` is a speed/accuracy balance — raise toward 1280
  if small labels (tea, cereal bars) are confused.
- Swap `yolo11s-seg.pt` for `m`/`l` if the GPU allows; larger helps fine-grained separation.
- **Production robustness:** D2S is shot near top-down, so `train.py` augments angle
  hard by default — rotation (`degrees=25`), vertical flip (`flipud=0.5`), perspective
  tilt, scale, and lighting (`hsv_v`). `close_mosaic=15` disables mosaic for the last
  15 epochs so it fine-tunes on realistic single-scene layouts. Override any via CLI.
