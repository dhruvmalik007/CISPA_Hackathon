"""test2_template.py - can the EMBEDDER tell us where the watermark is?

Idea: we know the message m. Re-embed the image X with m and with the opposite
message; the difference d = E(X,m) - E(X,1-m) is the pattern this message
would leave on this image. Where X really carries the watermark, X's own
mid-frequency detail correlates with d; in holes and clean areas it does not.

The pattern is computed on the WHOLE image, so it is on the right grid for
mostly-watermarked images (their watermark box is the whole image). For
mostly-clean images the result tells us whether the grid is the whole image
or the small rectangle.

Run:   PYTHONPATH=.. python -u test2_template.py --data ../release
Out:   printed tables + cache_tmpl/maps.npz (maps reusable as U-Net inputs)
GPU is used only for the embed calls (4 per image); the rest is numpy.
"""
import argparse, os
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

# ============================ ADAPTER ========================================
# The only part that touches the starter kit. Put load_train() and embed()
# (and optionally remove()) in a file kit_adapter.py next to this script;
# they replace the stubs below.

def load_train(data_dir):
    """Return a list with one dict per TRAIN image:
        name   : str
        method : 'pixelseal' | 'trustmark' | 'stegastamp'
        image  : float32 numpy [512,512,3], values 0..1
        msg    : numpy vector of 0/1 (the embedded message)
        gt     : numpy [128,128] of 0/1 (true mask)
        full   : bool (the full_watermark flag)
    """
    raise NotImplementedError("fill from step4_embed.py")


def embed(method, image, msg):
    """Watermark the WHOLE 512x512 image with msg, using the kit embedder of
    `method` exactly as step4_embed.py does. image: float32 [512,512,3] 0..1.
    Returns the watermarked image, same shape and range (numpy)."""
    raise NotImplementedError("fill from step4_embed.py")


def remove(image):
    """Optional: TrustMark's watermark remover on the whole image
    (same format in and out). Return None if not available."""
    return None

try:                                   # preferred: keep your two functions in kit_adapter.py,
    from kit_adapter import *          # so replacing this script never loses them
except ImportError:
    pass

# ============================ CORE ===========================================
SIGMAS = (1.5, 4.0)     # band-pass scale in px (watermarks are upsampled, so not only fine detail)
WINDOWS = (16, 32)      # local window in px


def bandpass(a, s):
    return a - ndi.gaussian_filter(a, (s, s, 0))


def pool128(a):
    return a.reshape(128, 4, 128, 4).mean((1, 3))


def template_maps(X, T, tag):
    """Local agreement between image X and template T. Returns {variant: [128,128]}."""
    out = {}
    for s in SIGMAS:
        y, t = bandpass(X, s), bandpass(T, s)
        ty, tt, yy = (t * y).sum(2), (t * t).sum(2), (y * y).sum(2)
        for w in WINDOWS:
            bty, btt, byy = (ndi.uniform_filter(v, w) for v in (ty, tt, yy))
            out[f"{tag}|s{s}|w{w}|ncc"] = pool128(bty / (np.sqrt(btt * byy) + 1e-12))
            out[f"{tag}|s{s}|w{w}|beta"] = pool128(bty / (btt + 1e-12))
    return out


def energy_maps(X):
    """Control: local noise level of the image alone (no watermark knowledge)."""
    return {f"energy|s{s}": pool128(np.log(ndi.uniform_filter((bandpass(X, s) ** 2).sum(2), 16) + 1e-12))
            for s in SIGMAS}


def auc(score, gt):
    g = gt.ravel() > 0
    n1, n0 = g.sum(), (~g).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score.ravel())
    return (r[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def miou(P, G):
    P = P.reshape(len(P), -1); G = G.reshape(len(G), -1)
    tot = 0
    for c in (False, True):
        inter = ((P == c) & (G == c)).sum(1).astype(float)
        union = ((P == c) | (G == c)).sum(1).astype(float)
        tot = tot + np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return tot / 2


def best_cut(S, G):
    """Best single threshold for a group of images -> (mean mIoU, threshold)."""
    qs = np.quantile(S, np.linspace(0.01, 0.99, 50))
    sc = [miou(S > q, G).mean() for q in qs]
    i = int(np.argmax(sc))
    return sc[i], float(qs[i])


def shape_of(gt, full):
    if full:
        return "mostly-wm (holes)"
    ys, xs = np.where(gt > 0)
    if len(ys) == 0:
        return "clean object"
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return "clean rectangle" if gt.sum() >= 0.97 * box else "clean object"


def compute(items, embed_fn, remove_fn=None, seed=0, log=print):
    rng = np.random.default_rng(seed)
    rows = []
    for k, it in enumerate(items):
        X = np.asarray(it["image"], np.float32)
        m = np.asarray(it["msg"]).astype(np.float32).ravel()
        Wm = embed_fn(it["method"], X, m)
        d = Wm - embed_fn(it["method"], X, 1 - m)
        mr = rng.integers(0, 2, len(m)).astype(np.float32)          # wrong message (control)
        dr = embed_fn(it["method"], X, mr) - embed_fn(it["method"], X, 1 - mr)
        maps = {}
        maps.update(template_maps(X, d, "diff"))                      # E(m) - E(1-m)
        maps.update(template_maps(X, Wm - X, "plain"))                # E(m) - X
        maps.update(template_maps(X, dr, "ctrl-diff"))                # same recipes with the wrong message
        maps.update(template_maps(X, embed_fn(it["method"], X, mr) - X, "ctrl-plain"))
        maps.update(energy_maps(X))
        if remove_fn is not None and it["method"] == "trustmark":
            C = remove_fn(X)
            if C is not None:
                maps.update(template_maps(X - C, d, "removed"))       # estimated present residual vs template
        rows.append(maps)
        if (k + 1) % 50 == 0:
            log(f"  {k + 1}/{len(items)} images")
    return rows


def selftest(items, embed_fn, n=10, seed=0, log=print):
    """Positive control: watermark clean-ish covers OURSELVES with the kit embedder, cut a hole,
    and run the same maps. 'exact' = our watermark on the 512 grid; 'zoom1.1' = the same image
    enlarged by 10% and centre-cropped (watermark no longer on the 512 grid)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:512, :512]
    log("\n=== SELF-TEST (watermarks we embed ourselves) ===")
    log(f"{'method':<12}{'geometry':<10}{'best variant':<26}{'pixel AUC':>10}{'wrong-msg':>11}")
    for me in sorted({it["method"] for it in items}):
        covers = [it for it in items if it["method"] == me and not it["full"]][:n]
        acc = {"exact": {}, "zoom1.1": {}}
        for it in covers:
            C = np.asarray(it["image"], np.float32)
            m = rng.integers(0, 2, len(np.asarray(it["msg"]).ravel())).astype(np.float32)
            mr = rng.integers(0, 2, len(m)).astype(np.float32)
            W = np.round(np.clip(embed_fn(me, C, m), 0, 1) * 255) / 255
            cy, cx = rng.integers(140, 372, 2); ry, rx = rng.integers(50, 120, 2)
            H = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
            gt = ~(H.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5)
            for geo in acc:
                Wg, Cg = W, C
                if geo == "zoom1.1":
                    z = lambda a: ndi.zoom(a, (1.1, 1.1, 1), order=1)[26:538, 26:538]
                    Wg, Cg = z(W), z(C)
                X = np.where(H[..., None], Cg, Wg).astype(np.float32)
                Em = embed_fn(me, X, m)
                maps = {}
                maps.update(template_maps(X, Em - embed_fn(me, X, 1 - m), "diff"))
                maps.update(template_maps(X, Em - X, "plain"))
                maps.update(template_maps(X, embed_fn(me, X, mr) - embed_fn(me, X, 1 - mr), "ctrl-diff"))
                maps.update(template_maps(X, embed_fn(me, X, mr) - X, "ctrl-plain"))
                for v, mp in maps.items():
                    acc[geo].setdefault(v, []).append(auc(mp, gt))
        for geo, d in acc.items():
            if not d:
                continue
            mean = {v: float(np.mean(a)) for v, a in d.items()}
            best = max((v for v in mean if not v.startswith("ctrl-")), key=mean.get)
            log(f"{me:<12}{geo:<10}{best:<26}{mean[best]:>10.3f}{mean['ctrl-' + best]:>11.3f}")


def report(items, rows, log=print):
    G = np.stack([np.asarray(it["gt"]) > 0 for it in items])
    meth = np.array([it["method"] for it in items])
    shp = np.array([shape_of(np.asarray(it["gt"]), bool(it["full"])) for it in items])
    step1 = miou(np.stack([np.full((128, 128), bool(it["full"])) for it in items]), G)   # all-ones / all-zeros by kind
    summary = {}
    for me in sorted(set(meth)):
        for sh in ("mostly-wm (holes)", "clean rectangle", "clean object"):
            idx = np.where((meth == me) & (shp == sh))[0]
            if len(idx) == 0:
                continue
            variants = sorted(set.intersection(*[set(rows[i]) for i in idx]))
            res = []
            for v in variants:
                S = np.stack([rows[i][v] for i in idx])
                a = np.nanmean([auc(S[j], G[i]) for j, i in enumerate(idx)])
                sign = 1 if a >= 0.5 or not v.startswith("energy") else -1    # energy may point either way
                sc, th = best_cut(sign * S, G[idx])
                res.append((v, a, sc))
            real = [r for r in res if not r[0].startswith(("ctrl-", "energy"))]
            best = max(real, key=lambda r: r[1])
            ctrl = [r for r in res if r[0] == "ctrl-" + best[0]]
            en = max((r for r in res if r[0].startswith("energy")), key=lambda r: abs(r[1] - 0.5))
            log(f"\n### {me} / {sh}   n={len(idx)}   kind-only baseline mIoU={step1[idx].mean():.3f}")
            log(f"{'variant':<28}{'pixel AUC':>10}{'mIoU':>8}")
            for v, a, sc in sorted(real, key=lambda r: -r[1])[:6]:
                log(f"{v:<28}{a:>10.3f}{sc:>8.3f}")
            if ctrl:
                log(f"{'same, WRONG message':<28}{ctrl[0][1]:>10.3f}{ctrl[0][2]:>8.3f}   <- should be ~0.5 AUC")
            log(f"{en[0] + ' (no wm info)':<28}{en[1]:>10.3f}{en[2]:>8.3f}   <- edit artifacts only")
            summary[(me, sh)] = (best, ctrl[0] if ctrl else None, en)
    log("\n=== SUMMARY (best variant per group) ===")
    log(f"{'group':<38}{'AUC':>7}{'wrong-msg':>11}{'energy':>8}{'mIoU':>8}{'baseline':>10}")
    for (me, sh), (b, c, e) in summary.items():
        idx = (meth == me) & (shp == sh)
        log(f"{me + ' / ' + sh:<38}{b[1]:>7.3f}{(c[1] if c else float('nan')):>11.3f}{e[1]:>8.3f}{b[2]:>8.3f}{step1[idx].mean():>10.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images per method (quick check)")
    ap.add_argument("--out", default="cache_tmpl")
    ap.add_argument("--selftest", type=int, default=0, help="run only the positive control on N covers per method")
    a = ap.parse_args()
    items = load_train(a.data)
    if a.selftest:
        selftest(items, embed, a.selftest)
        return
    if a.limit:
        keep, seen = [], {}
        for it in items:                                   # N mostly-wm + N mostly-clean per method
            key = (it["method"], bool(it["full"]))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= a.limit:
                keep.append(it)
        items = keep
    print(f"{len(items)} images")
    rows = compute(items, embed, remove)
    report(items, rows)
    os.makedirs(a.out, exist_ok=True)
    names = sorted(set.intersection(*[set(r) for r in rows]))
    np.savez_compressed(os.path.join(a.out, "maps.npz"),
                        names=np.array([it["name"] for it in items]), variants=np.array(names),
                        maps=np.stack([[r[v] for v in names] for r in rows]).astype(np.float16))
    print(f"\nsaved {a.out}/maps.npz  [images, variants, 128, 128]")


if __name__ == "__main__":
    main()