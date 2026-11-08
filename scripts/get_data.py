"""Download the two public datasets into data/ (they are not stored in this repository).

    python scripts/get_data.py

Needs git. Copies come from public GitHub mirrors of the original datasets; you can also download the originals
yourself (links in data/SOURCES.md) and arrange them like this:

    data/neu/train/<class>/*.jpg     data/neu/test/<class>/*.jpg      (6 classes, 240 / 60 images each)
    data/casting/ok/*.jpeg           data/casting/defect/*.jpeg       (casting_512x512 version, 519 / 781)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
NEU_MIRROR = "https://github.com/Dabi0228/NEU-Surface-Defect-Dataset.git"
CASTING_MIRROR = "https://github.com/kwakjm/casting_products.git"


def clone(url: str, dest: Path) -> None:
    subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(dest)], check=True)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if not (DATA / "neu").exists():
            print("Downloading NEU steel surface defects ...")
            clone(NEU_MIRROR, tmp / "neu")
            shutil.copytree(tmp / "neu" / "train" / "images", DATA / "neu" / "train")
            shutil.copytree(tmp / "neu" / "validation" / "images", DATA / "neu" / "test")
        if not (DATA / "casting").exists():
            print("Downloading casting product images ...")
            clone(CASTING_MIRROR, tmp / "casting")
            src = tmp / "casting" / "data" / "Casting material" / "casting_512x512" / "casting_512x512"
            shutil.copytree(src / "ok_front", DATA / "casting" / "ok")
            shutil.copytree(src / "def_front", DATA / "casting" / "defect")
    for d in ("neu", "casting"):
        n = sum(1 for _ in (DATA / d).rglob("*.jp*g"))
        print(f"{d}: {n} images")


if __name__ == "__main__":
    main()
