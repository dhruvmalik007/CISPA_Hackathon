"""Save a contact sheet of forget images grouped by the original model's predicted class,
so you can see what the classes are (and guess which public dataset they resemble).

  python show_forget.py --diag diag/per_image.npz --classes 12 --per-class 8 --out forget_grid.jpg
Rows = classes (most frequent first), labelled with class id and forget count.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw

from task_template import load_forget_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forget", default="forget_images.npz")
    ap.add_argument("--diag", default="diag/per_image.npz")
    ap.add_argument("--classes", type=int, default=12, help="number of classes (rows); 0 = all present")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--tile", type=int, default=112)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="forget_grid.jpg")
    a = ap.parse_args()

    imgs = load_forget_images().numpy() if a.forget == "forget_images.npz" else np.load(a.forget)["images"]
    y = np.load(a.diag)["pseudo_label"]
    counts = np.bincount(y, minlength=100)
    order = [c for c in np.argsort(-counts, kind="stable") if counts[c] > 0]
    if a.classes:
        order = order[: a.classes]
    rng = np.random.default_rng(a.seed)

    t, label_w = a.tile, 90
    sheet = Image.new("RGB", (label_w + t * a.per_class, t * len(order)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, c in enumerate(order):
        idx = np.flatnonzero(y == c)
        pick = rng.choice(idx, size=min(a.per_class, len(idx)), replace=False)
        draw.text((6, r * t + t // 2 - 12), f"class {c}\nn={counts[c]}", fill="black")
        for j, i in enumerate(pick):
            sheet.paste(Image.fromarray(imgs[i]).resize((t, t)), (label_w + j * t, r * t))
    sheet.save(a.out, quality=90)
    print(f"wrote {a.out} ({len(order)} classes x {a.per_class})")


if __name__ == "__main__":
    main()