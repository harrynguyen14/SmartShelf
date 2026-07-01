"""Convert SKU110K CSV -> YOLO format (1 class 'product') + data.yaml.

SKU110K CSV (khong header): name, x1, y1, x2, y2, class, W, H  (pixel)
YOLO can: 1 file .txt / anh, moi dong  `cls cx cy w h`  (normalized 0-1).

Ultralytics tim label bang cach doi '/images/' -> '/labels/' trong duong dan anh
(cung stem, .txt). Anh goc o .../images/<name>; ta ghi label o .../labels/<stem>.txt
va liet ke duong dan anh trong <split>.txt. KHONG copy anh (ton dia).
3 split tron chung 1 thu muc anh -> tach bang danh sach <split>.txt, label dung chung
thu muc labels/ (stem train_*/val_*/test_* khong dung nhau).

Chay:  python smartshelf/load_dataset.py
"""
import csv
import os
from collections import defaultdict

SRC = r"D:/industry-item-dataset/SKU110K/SKU110K_fixed"
IMG_DIR = os.path.join(SRC, "images")
ANN = os.path.join(SRC, "annotations")
OUT = os.path.join(SRC, "yolo")           # noi ghi labels + danh sach + data.yaml

SPLITS = {"train": "annotations_train.csv",
          "val": "annotations_val.csv",
          "test": "annotations_test.csv"}


def convert_split(split, csv_name):
    rows_by_img = defaultdict(list)
    dims = {}
    with open(os.path.join(ANN, csv_name), newline="") as f:
        for r in csv.reader(f):
            if len(r) < 8:
                continue
            name = r[0]
            x1, y1, x2, y2 = map(float, (r[1], r[2], r[3], r[4]))
            W, H = float(r[6]), float(r[7])
            rows_by_img[name].append((x1, y1, x2, y2))
            dims[name] = (W, H)

    # label phai o thu muc mirror cua anh: .../images/x.jpg -> .../labels/x.txt
    lbl_dir = os.path.join(SRC, "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    img_list = []
    n_box = 0
    for name, boxes in rows_by_img.items():
        W, H = dims[name]
        lines = []
        for x1, y1, x2, y2 in boxes:
            # clamp vao trong anh roi -> normalized center/w/h
            x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
            y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
        n_box += len(lines)
        stem = os.path.splitext(name)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        # path TUYET DOI -> Ultralytics resolve dung o moi cwd. Tren Colab sua
        # IMG_DIR (dau file) thanh /content/drive/.../images roi chay lai.
        img_list.append(os.path.join(IMG_DIR, name).replace("\\", "/"))

    list_path = os.path.join(OUT, f"{split}.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(img_list))
    print(f"{split}: {len(img_list)} anh, {n_box} box -> {lbl_dir}")
    return list_path


def main():
    os.makedirs(OUT, exist_ok=True)
    for s, c in SPLITS.items():
        convert_split(s, c)
    yaml = (
        "# SKU110K, 1 class product. 'path' PHAI tuyet doi (Ultralytics resolve\n"
        "# path tuong doi theo datasets_dir, KHONG theo vi tri file nay).\n"
        "# Tren Colab: sua dong 'path:' thanh /content/drive/.../yolo\n"
        f"path: {OUT}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n  0: product\n"
    )
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(yaml)
    print(f"data.yaml -> {os.path.join(OUT, 'data.yaml')}")


def _selfcheck():
    # box giua anh 100x100: (25,25,75,75) -> cx=cy=0.5, w=h=0.5
    W = H = 100.0
    x1, y1, x2, y2 = 25, 25, 75, 75
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    bw, bh = (x2 - x1) / W, (y2 - y1) / H
    assert (cx, cy, bw, bh) == (0.5, 0.5, 0.5, 0.5), (cx, cy, bw, bh)
    print("selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
    main()
