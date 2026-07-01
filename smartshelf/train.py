"""Train YOLO11n detect 1 class 'product' tren SKU110K (cloud GPU).

Edge dich = Raspberry Pi -> nano + sau nay export NCNN (xem validate.py / deploy).
imgsz cao (960) de giu vat nho day dac cua SKU110K; chong miss them o INFERENCE
bang tiling + da-frame (post-process rieng), KHONG phai o day.

Chay:  python smartshelf/train.py
Truoc do:  python smartshelf/load_dataset.py   (sinh data.yaml)
"""
import os
from ultralytics import YOLO

SRC = r"D:/industry-item-dataset/SKU110K/SKU110K_fixed"
DATA = os.path.join(SRC, "yolo", "data.yaml")

# SKU110K vat nho day dac -> imgsz cao. Chinh xuong neu GPU het VRAM.
IMGSZ = 960
EPOCHS = 100
BATCH = 16          # giam neu OOM
MODEL = "yolo11n.pt"   # nano cho edge Pi
# QUAN TRONG (Colab): de checkpoint vao DRIVE, khong phai /content (mat khi disconnect).
# Doi sang /content/drive/MyDrive/... khi chay tren Colab.
PROJECT = "runs_smartshelf"


def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        # SKU110K co the >300 box/anh -> noi long gioi han detection
        max_det=600,
        # vat nho day dac: tat mosaic 10 epoch cuoi de hoc box that
        close_mosaic=10,
        save_period=10,    # luu ckp moi 10 epoch (ngoai best.pt + last.pt)
        project=PROJECT,
        name="yolo11n_sku110k",
        resume=False,      # doi True + MODEL=last.pt de chay tiep khi dut
    )


if __name__ == "__main__":
    main()
