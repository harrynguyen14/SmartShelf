"""Extract the D2S and SKU110K tar archives in place."""
import sys, tarfile
from pathlib import Path

ROOT = Path(r"D:\industry-item-dataset")
ARCHIVES = [
    ROOT / "D2S" / "d2s_annotations_v1.1.tar.xz",
    ROOT / "D2S" / "d2s_images_v1.tar.xz",
    ROOT / "SKU110K" / "SKU110K_fixed.tar.gz",
]


def extract(archive: Path):
    dest = archive.parent
    print(f"Extracting {archive.name} -> {dest} ...", flush=True)
    with tarfile.open(archive) as tar:
        # filter='data' (Python 3.12+) blocks path-traversal/absolute members
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)  # ponytail: older Python, no filter arg
    print(f"  done: {archive.name}", flush=True)


if __name__ == "__main__":
    missing = [a for a in ARCHIVES if not a.exists()]
    if missing:
        sys.exit("Missing: " + ", ".join(str(m) for m in missing))
    for a in ARCHIVES:
        extract(a)
    print("All done.")
