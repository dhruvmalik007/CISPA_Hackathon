"""
Test 3: train the step 6 network on SYNTHETIC images watermarked with the kit embedder,
then score it on REAL images it never saw. Default group: TrustMark, mostly watermarked.
  PYTHONPATH=.. python -u test3_synth.py --data ../release --check     # pool + realism check, no training
  PYTHONPATH=.. python -u test3_synth.py --data ../release             # + training, synthetic and REAL score per epoch
No submission.
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.stats import rankdata

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128, bits_of, to_tensor, feat_map
from step3_train import loss_fn, miou_batch
import step6_stega as S6

BATCH = 8
GRID = np.arange(0.05, 0.96, 0.05)
REF = {"trustmark": 0.486, "pixelseal": 0.509, "stegastamp": 0.718}


def raw_logits(wm, x):
    fn = getattr(type(wm).logits, "__wrapped__", None)
    return fn(wm, x) if fn is not None else wm.logits(x)


def embed(wm, img_u8, bits):
    w = wm.embed(img_u8, [int(b) for b in bits])
    w = w.detach().float().cpu().numpy() if torch.is_tensor(w) else np.asarray(w).astype(np.float32)
    if w.ndim == 4:
        w = w[0]
    if w.ndim == 3 and w.shape[0] == 3 and w.shape[-1] != 3:
        w = w.transpose(1, 2, 0)
    if w.max() <= 1.5:
        w = w * 255.0
    return np.clip(np.round(w), 0, 255).astype(np.uint8)


def u8(img):
    return np.ascontiguousarray(np.asarray(img).astype(np.uint8))


def load_covers(a, method):
    covers = []
    if a.covers:
        from PIL import Image
        for p in sorted(Path(a.covers).rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                im = Image.open(p).convert("RGB")
                s = 512 / min(im.size)
                im = im.resize((max(512, round(im.size[0] * s)), max(512, round(im.size[1] * s))), Image.BICUBIC)
                l, t = (im.size[0] - 512) // 2, (im.size[1] - 512) // 2
                covers.append(u8(im.crop((l, t, l + 512, t + 512))))
    else:
        for split in ("train", "test"):
            for it in LocalisationDataset(a.data, split):
                if str(it["method"]).lower() != method:
                    covers.append(u8(it["image"]))
    print(f"covers: {len(covers)} images ({'folder ' + a.covers if a.covers else 'other methods, train + test'})", flush=True)
    return covers


def build_pool(a, wm, covers, nbits):
    d = Path(f"cache_synth_{a.method}")
    d.mkdir(exist_ok=True)
    if (d / "done.txt").exists() and np.load(d / "wm.npy", mmap_mode="r").shape[0] == a.pool:
        print(f"pool: reusing {d}/", flush=True)
    else:
        C = np.lib.format.open_memmap(d / "clean.npy", "w+", np.uint8, (a.pool, 512, 512, 3))
        Wm = np.lib.format.open_memmap(d / "wm.npy", "w+", np.uint8, (a.pool, 512, 512, 3))
        rng = np.random.default_rng(1)
        msgs = rng.integers(0, 2, (a.pool, nbits)).astype(np.uint8)
        t0 = time.time()
        for i in range(a.pool):
            c = covers[i % len(covers)]
            f = i // len(covers) + rng.integers(8)
            if f & 1:
                c = c[:, ::-1]
            if f & 2:
                c = c[::-1]
            if f & 4:
                c = c.transpose(1, 0, 2)
            c = np.ascontiguousarray(c)
            C[i] = c
            Wm[i] = embed(wm, c, msgs[i])
            if (i + 1) % 250 == 0:
                print(f"  pool {i + 1}/{a.pool}  ({time.time() - t0:.0f}s)", flush=True)
        C.flush()
        Wm.flush()
        np.save(d / "msgs.npy", msgs)
        np.save(d / "cover_id.npy", np.arange(a.pool) % len(covers))
        (d / "done.txt").write_text("ok")
    C = np.load(d / "clean.npy", mmap_mode="r")
    Wm = np.load(d / "wm.npy", mmap_mode="r")
    msgs = np.load(d / "msgs.npy")
    cid = np.load(d / "cover_id.npy")
    ncov = cid.max() + 1
    val = np.where(cid >= int(0.9 * ncov))[0]
    tr = np.where(cid < int(0.9 * ncov))[0]
    return C, Wm, msgs, tr, val


YY, XX = np.mgrid[0:512, 0:512].astype(np.float32)


def holes(rng):
    k = int(rng.integers(1, 4))
    total = rng.uniform(0.005, 0.30) * 512 * 512
    share = rng.dirichlet(np.ones(k))
    H = np.zeros((512, 512), bool)
    for j in range(k):
        A = total * share[j]
        r = rng.uniform(0.5, 2.0)
        rx = math.sqrt(A * r / math.pi)
        ry = A / (math.pi * rx)
        if rng.random() < 1 / 3:
            t = rng.uniform(0, 512)
            cx, cy = [(t, 0.0), (t, 511.0), (0.0, t), (511.0, t)][int(rng.integers(4))]
            rx, ry = rx * 1.41, ry * 1.41
        else:
            cx, cy = rng.uniform(0, 512), rng.uniform(0, 512)
        th = rng.uniform(0, math.pi)
        dx, dy = XX - cx, YY - cy
        u, v = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
        ang = np.arctan2(v, u)
        rad = 1.0 + sum(rng.uniform(0, 0.15) * np.sin(n * ang + rng.uniform(0, 6.3)) for n in (2, 3, 4))
        H |= (u / rx) ** 2 + (v / ry) ** 2 < rad ** 2
    return H


def sample(i, ids, C, Wm, rng, wm_is_one):
    H = holes(rng)
    al = ndi.gaussian_filter(H.astype(np.float32), rng.uniform(1.0, 2.0))[..., None]
    fill = C[i] if rng.random() < 0.5 else C[int(rng.choice(ids))]
    img = Wm[i].astype(np.float32) * (1 - al) + fill.astype(np.float32) * al
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    y = H.reshape(128, 4, 128, 4).mean((1, 3)) < 0.5
    return img, (y if wm_is_one else ~y)


def auc(score, lab):
    s, l = score.ravel(), lab.ravel().astype(bool)
    n1, n0 = l.sum(), (~l).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return (r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def check_stats(wm, imgs, msgs, labs, dev):
    accs, aucs = [], []
    for img, m, y in zip(imgs, msgs, labs):
        x = to_tensor(img)[None].to(dev)
        z = raw_logits(wm, x).float()[0].detach().cpu().numpy()
        L = min(len(z), len(m))
        accs.append(((z[:L] > 0).astype(int) == m[:L]).mean())
        sgn = torch.tensor(2.0 * m - 1.0, dtype=torch.float32, device=dev)
        fm = feat_map(wm, x, sgn)
        if fm is not None:
            v = auc(fm.numpy(), y)
            if v is not None:
                aucs.append(v)
    return np.array(accs), (np.mean(aucs) if aucs else float("nan"))


def score(net, I, Y, dev):
    P = S6.infer(net, I, torch.zeros(len(I), 1, 128, 128), torch.arange(len(I)), dev)
    return max(miou_batch(P > t, Y.bool()).mean().item() for t in GRID)


def to_input(img, mode):
    return S6.make_input(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1), mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--covers", default="")
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--input", default="hp", choices=["rgb", "hp", "both"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, meth = a.device, a.method
    S6.ENCODER, S6.IN_CH = a.encoder, (6 if a.input == "both" else 3)
    random.seed(0)
    torch.manual_seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    real = [it for it in all_train if str(it["method"]).lower() == meth and accs[it["image_name"]] > thr[meth]]
    R_img = [u8(it["image"]) for it in real]
    R_msg = [bits_of(it["message"]) for it in real]
    R_y = [mask128(it["mask128"]).astype(bool) for it in real]
    share = np.mean([y.mean() for y in R_y])
    wm_is_one = share > 0.5
    print(f"real group: {meth} mostly watermarked, n={len(real)}, mask=1 share {share:.2f} -> "
          f"label 1 = {'watermarked' if wm_is_one else 'clean'}", flush=True)
    nbits = len(R_msg[0])

    wm = W._CLASSES[meth](Path("models").resolve(), dev)
    covers = load_covers(a, meth)
    C, Wm, msgs, tr, val = build_pool(a, wm, covers, nbits)
    del covers
    print(f"pool: {len(tr)} training, {len(val)} held-out (unseen covers)", flush=True)

    rng = np.random.default_rng(7)
    ids = rng.choice(np.concatenate([tr, val]), 200, replace=False)
    S = [sample(int(i), tr, C, Wm, rng, wm_is_one) for i in ids]
    s_acc, s_auc = check_stats(wm, [s[0] for s in S], [msgs[int(i)] for i in ids], [s[1] for s in S], dev)
    r_acc, r_auc = check_stats(wm, R_img, R_msg, R_y, dev)
    s_share = np.mean([s[1].mean() if wm_is_one else 1 - s[1].mean() for s in S])
    r_share = share if wm_is_one else 1 - share
    print("\n=== REALISM CHECK (synthetic should look like real) ===")
    print(f"  {'':28}{'synthetic':>11}{'real':>9}")
    print(f"  {'bit accuracy, mean':28}{s_acc.mean():>11.3f}{r_acc.mean():>9.3f}")
    print(f"  {'bit accuracy, 10th pct':28}{np.percentile(s_acc, 10):>11.3f}{np.percentile(r_acc, 10):>9.3f}")
    print(f"  {'feat AUC':28}{s_auc:>11.3f}{r_auc:>9.3f}")
    print(f"  {'watermarked share':28}{s_share:>11.3f}{r_share:>9.3f}", flush=True)
    if a.check:
        print("\nCheck done. Paste me this table.")
        return

    rng = np.random.default_rng(11)
    vs = [sample(int(i), val, C, Wm, rng, wm_is_one) for i in rng.choice(val, 300, replace=len(val) < 300)]
    VI = torch.stack([to_input(s[0], a.input) for s in vs])
    VY = torch.from_numpy(np.stack([s[1] for s in vs])).float()
    RI = torch.stack([to_input(im, a.input) for im in R_img])
    RY = torch.from_numpy(np.stack(R_y)).float()
    del vs

    net = S6.Net(1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    rng = np.random.default_rng(3)
    extra = torch.zeros(BATCH, 1, 128, 128, device=dev)
    best = -1
    print(f"\n=== TRAINING on synthetic only ({a.epochs} epochs x {a.steps * BATCH} images, input {a.input}) ===")
    print(f"  real score to beat (current CV): {REF.get(meth, float('nan')):.3f}", flush=True)
    for ep in range(a.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for _ in range(a.steps):
            bs = [sample(int(i), tr, C, Wm, rng, wm_is_one) for i in rng.choice(tr, BATCH)]
            img = torch.stack([to_input(s[0], a.input) for s in bs]).to(dev).float() / 255
            y = torch.from_numpy(np.stack([s[1] for s in bs])).float().to(dev)
            f = random.randrange(8)
            img, y = S6.tf(img, f), S6.tf(y, f)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                logit = net(img, extra)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        ss, rs = score(net, VI, VY, dev), score(net, RI, RY, dev)
        if rs > best:
            best = rs
            torch.save(net.state_dict(), f"test3_{meth}_{a.input}.pt")
        print(f"  epoch {ep + 1:2d}/{a.epochs}  loss {tot / a.steps:.3f}   synthetic {ss:.3f}   "
              f"REAL {rs:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nbest real score {best:.3f} (current CV {REF.get(meth, float('nan')):.3f}). Paste me this table.")


if __name__ == "__main__":
    main()
