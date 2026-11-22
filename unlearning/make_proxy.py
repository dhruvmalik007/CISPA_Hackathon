"""Build an external retain-proxy set as npz (key "images", uint8 [N, 224, 224, 3]) for --proxy-npz.

Sources (pick one):
  --folder PATH       any folder of images, searched recursively (jpg/jpeg/png/webp/bmp)
  --torchvision NAME  a torchvision dataset downloaded to --root: CIFAR100, CIFAR10, STL10
                      (download on a login node: compute nodes usually have no internet)

Images are resized (shorter side 224) and centre-cropped to 224x224.

Examples:
  python make_proxy.py --folder /path/to/imagenet/val --n 5000 --out proxy.npz
  python make_proxy.py --torchvision STL10 --root data --n 5000 --out proxy_stl10.npz
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def to_224(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder")
    src.add_argument("--torchvision", choices=["CIFAR100", "CIFAR10", "STL10"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="proxy.npz")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    out, labels = [], []
    if a.folder:
        files = sorted(p for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXTS)
        if not files:
            raise SystemExit(f"no images found under {a.folder}")
        files = rng.sample(files, min(a.n, len(files)))
        for i, f in enumerate(files):
            try:
                out.append(to_224(Image.open(f)))
                labels.append(f.parent.name)  # folder name, often the class (e.g. ImageNet synset)
            except Exception as e:  # skip unreadable files
                print(f"skip {f}: {e}")
            if (i + 1) % 1000 == 0:
                print(f"{i+1}/{len(files)}", flush=True)
    else:
        import torchvision.datasets as D
        if a.torchvision == "STL10":
            ds = D.STL10(a.root, split="test", download=True)
        else:
            ds = getattr(D, a.torchvision)(a.root, train=False, download=True)
        idx = rng.sample(range(len(ds)), min(a.n, len(ds)))
        for i in idx:
            img, lab = ds[i]
            out.append(to_224(img))
            labels.append(str(lab))

    images = np.stack(out)
    np.savez(a.out, images=images, source_label=np.array(labels))
    print(f"wrote {a.out}: {images.shape} uint8 ({images.nbytes/2**20:.0f} MB)")


if __name__ == "__main__":
    main()