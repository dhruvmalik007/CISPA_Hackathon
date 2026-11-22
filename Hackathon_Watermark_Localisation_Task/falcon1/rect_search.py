"""
Rectangle search: find the rectangle whose crop decodes the known message best.
    PYTHONPATH=.. python -u rect_search.py --data ../release --check 30
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from step2_shapes import METHODS, kind_thresholds, mask128, bits_of, to_tensor
from step3_train import miou_batch
import wmloc.watermarks as W

GRID = 64
REFINE = (32, 16, 8, 4)
MAX_ASPECT = {"pixelseal": 16, "stegastamp": 16, "trustmark": 2}
BATCH = 64
CACHE = "cache_rect"


def candidates(kind, meth):
    pts = list(range(0, 513, GRID))
    out = []
    for y0 in pts:
        for y1 in pts:
            if y1 - y0 < 32:
                continue
            for x0 in pts:
                for x1 in pts:
                    if x1 - x0 < 32:
                        continue
                    h, w = y1 - y0, x1 - x0
                    if max(h / w, w / h) > MAX_ASPECT[meth] * 1.5:
                        continue
                    area = h * w / 512 ** 2
                    if (kind == 0 and area > 0.4) or (kind == 1 and (area < 0.25 or area > 0.97)):
                        continue
                    out.append((y0, x0, y1, x1))
    return out


class Scorer:
    def __init__(self, wm, x, sgn):
        self.wm, self.x, self.sgn = wm, x, sgn
        self.R = int(getattr(wm, "resolution", 256))
        z = wm.logits(x).float().cpu()
        self.L = min(z.shape[1], len(sgn))
        self.scale = z[:, : self.L].abs().median().item() + 1e-6

    @torch.no_grad()
    def __call__(self, boxes):
        soft, acc = [], []
        for i in range(0, len(boxes), BATCH):
            crops = [F.interpolate(self.x[:, :, y0:y1, x0:x1], size=(self.R, self.R), mode="bilinear",
                                   align_corners=False, antialias=True) for (y0, x0, y1, x1) in boxes[i:i + BATCH]]
            z = self.wm.logits(torch.cat(crops)).float().cpu()[:, : self.L]
            a = z * self.sgn[: self.L]
            soft.append(torch.tanh(a / self.scale).mean(1))
            acc.append((a > 0).float().mean(1))
        return torch.cat(soft), torch.cat(acc)


def search(wm, it, kind, meth, device):
    x = to_tensor(it["image"])[None].to(device)
    sgn = torch.tensor(bits_of(it["message"]) * 2 - 1, dtype=torch.float32)
    sc = Scorer(wm, x, sgn)
    cands = candidates(kind, meth)
    if not cands:
        return None
    soft, acc = sc(cands)
    order = soft.argsort(descending=True).tolist()
    starts = []
    for j in order:
        if all(max(abs(a - b) for a, b in zip(cands[j], s[0])) > GRID for s in starts):
            starts.append((list(cands[j]), soft[j].item(), acc[j].item()))
        if len(starts) == 3:
            break
    best = None
    for box, s, ac in starts:
        for st in REFINE:
            while True:
                moves = []
                for side in range(4):
                    for d in (st, -st):
                        nb = box.copy()
                        nb[side] = min(512, max(0, nb[side] + d))
                        if nb[2] - nb[0] >= 16 and nb[3] - nb[1] >= 16:
                            moves.append(tuple(nb))
                ms, ma = sc(moves)
                k = int(ms.argmax())
                if ms[k].item() > s + 1e-4:
                    box, s, ac = list(moves[k]), ms[k].item(), ma[k].item()
                else:
                    break
        if best is None or s > best[1]:
            best = (tuple(box), s, ac)
    return best


def box_mask(box):
    m = torch.zeros(512, 512)
    y0, x0, y1, x1 = box
    m[y0:y1, x0:x1] = 1
    return (m.view(128, 4, 128, 4).mean((1, 3)) >= 0.5)


def run(items, accs, thr, device, tag):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for meth in METHODS:
        its = [it for it in items if str(it["method"]).lower() == meth]
        path = os.path.join(CACHE, f"{tag}_{meth}.pt")
        cache = torch.load(path) if os.path.exists(path) else {}
        todo = [it for it in its if it["image_name"] not in cache]
        if todo:
            wm = W._CLASSES[meth](Path("models").resolve(), device)
            t0 = time.time()
            for n, it in enumerate(todo):
                kind = int(accs[it["image_name"]] > thr[meth])
                cache[it["image_name"]] = search(wm, it, kind, meth, device)
                if (n + 1) % 10 == 0 or n + 1 == len(todo):
                    torch.save(cache, path)
                    left = (len(todo) - n - 1) * (time.time() - t0) / (n + 1) / 60
                    print(f"  {tag} {meth}: {n + 1}/{len(todo)}  ({(time.time() - t0) / (n + 1):.1f}s per image, "
                          f"~{left:.0f} min left for this part)", flush=True)
            del wm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out.update(cache)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--check", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    items = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(items, accs)
    sub = sum([[it for it in items if str(it["method"]).lower() == m][: a.check] for m in METHODS], [])
    print(f"=== rectangle search on {len(sub)} train images ===", flush=True)
    res = run(sub, accs, thr, a.device, "train")

    oof, cut = None, {}
    if os.path.exists("analysis_oof.pt"):
        d = torch.load("analysis_oof.pt")
        oof = dict(zip(d["names"], d["oof"].float()))
        byname = {it["image_name"]: it for it in items}
        for m in METHODS:
            for k in (0, 1):
                ns = [n for n in oof if n in byname and str(byname[n]["method"]).lower() == m
                      and int(accs[n] > thr[m]) == k]
                if ns:
                    P = torch.stack([oof[n] for n in ns])
                    G = torch.stack([torch.from_numpy(mask128(byname[n]["mask128"])).bool() for n in ns])
                    cut[(m, k)] = max((miou_batch(P > t, G).mean().item(), t) for t in np.arange(0.05, 0.96, 0.05))[1]

    rows = []
    for it in sub:
        name, meth = it["image_name"], str(it["method"]).lower()
        kind = int(accs[name] > thr[meth])
        g = torch.from_numpy(mask128(it["mask128"])).bool()
        r = res.get(name)
        if r is None:
            continue
        bm = box_mask(r[0])
        ys, xs = torch.nonzero(g, as_tuple=True)
        is_rect = len(ys) > 0 and len(ys) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)).item() > 0.9
        model_mask = (oof[name] > cut.get((meth, kind), 0.5)) if oof is not None and name in oof else torch.full_like(g, bool(kind))
        rows.append(dict(meth=meth, kind=kind, rect=is_rect, acc=r[2], soft=r[1],
                         s_box=miou_batch(bm[None], g[None]).item(),
                         s_model=miou_batch(model_mask[None], g[None]).item(),
                         s_const=miou_batch(torch.full_like(g, bool(kind))[None], g[None]).item()))

    print("\n=== A. How well does the found rectangle match the truth? (score of predicting just the rectangle) ===")
    print(f"{'group':<34}{'n':>4}{'bit acc of best crop':>22}{'step1':>8}{'model':>8}{'rectangle':>11}")
    for m in METHODS:
        for k in (0, 1):
            for rect in (True, False):
                g_ = [r for r in rows if r["meth"] == m and r["kind"] == k and r["rect"] == rect]
                if not g_:
                    continue
                lab = f"{m} {'mostly clean' if k == 0 else 'mostly wm'} {'RECT' if rect else 'other'}"
                print(f"{lab:<34}{len(g_):>4}{np.mean([r['acc'] for r in g_]):22.3f}"
                      f"{np.mean([r['s_const'] for r in g_]):8.3f}{np.mean([r['s_model'] for r in g_]):8.3f}"
                      f"{np.mean([r['s_box'] for r in g_]):11.3f}")

    print("\n=== B. Rule: use the rectangle when its crop decodes with bit accuracy >= T, otherwise the model ===")
    base = np.mean([r["s_model"] for r in rows])
    print(f"model alone (step 3 CV predictions): {base:.3f}")
    for m in METHODS:
        for k in (0, 1):
            g_ = [r for r in rows if r["meth"] == m and r["kind"] == k]
            best = max(((np.mean([r["s_box"] if r["acc"] >= T else r["s_model"] for r in g_]), T)
                        for T in np.arange(0.6, 1.001, 0.025)), key=lambda z: z[0])
            used = np.mean([r["acc"] >= best[1] for r in g_])
            print(f"  {m:<10} {'mostly clean' if k == 0 else 'mostly wm   '}: model {np.mean([r['s_model'] for r in g_]):.3f}"
                  f" -> with rectangles {best[0]:.3f}  (T={best[1]:.3f}, rectangle used on {used:.0%} of images)")
    print("If 'with rectangles' is clearly above 'model', the rectangle search is worth running on everything.")


if __name__ == "__main__":
    main()
