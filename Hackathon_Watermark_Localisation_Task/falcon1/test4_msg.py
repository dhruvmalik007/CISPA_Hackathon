"""
Test 4: one-hour run for one group (default: TrustMark, mostly watermarked).

Same network as test 3c (DecNet), with five changes:
  1. Message channel: the decoder's own final linear layer is applied at every position of
     layer4, multiplied by the KNOWN bits, and fed to the mask head. Starts at zero, so with
     --init the network begins exactly as the 7b model.
  2. Wrong-message negatives: part of each batch gets random bits and an empty target.
  3. Keep-only windows: part of each batch keeps the watermark only inside a rectangle.
  4. Bit loss: per-position bits must match the message where the watermark is, and be
     undecided where it is not.
  5. Speed: the synthetic pool lives on the GPU and holes are cut there. No flip/rotation
     after embedding (test 3c applied S6.tf to already-watermarked images).

Run inside falcon1/ :
  PYTHONPATH=.. python -u test4_msg.py --data ../release --minutes 40
  PYTHONPATH=.. python -u test4_msg.py --data ../release --minutes 40 --no_msg     # control: speed-up only
  PYTHONPATH=.. python -u test4_msg.py --data ../release --minutes 40 --init none  # start from the decoder, not from 7b
  PYTHONPATH=.. python -u test4_msg.py --data ../release --predict_only            # rewrite the submission file

Writes test4_<method>_msg.pt (checkpoint) and, at the end, --out (a copy of --base with this
group's test masks replaced). It never submits.
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128, bits_of, to_tensor
from step3_train import loss_fn
from test3_synth import build_pool, u8, to_input
from test3b_synth import sample, best_score
from test3c_dec import DecNet, measure_preproc, raw_logits

REF = {"trustmark": 0.578}          # score of the submitted 7b model on the real images


def as_bits(x):
    """Message in any of the usual forms -> float tensor of 0/1."""
    if isinstance(x, str):
        v = np.array([int(c) for c in x.strip()], np.float32)
    else:
        v = np.asarray(x.detach().cpu() if torch.is_tensor(x) else x).astype(np.float32).ravel()
    if not np.isin(v, (0, 1)).all():
        raise SystemExit(f"message is not a 0/1 vector (first values: {v[:8]}). Tell me what bits_of / msgs contain.")
    return torch.from_numpy(v)


def ac(dev):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda"))


class MsgNet(DecNet):
    def __init__(self, backbone, pre, nbits, use_msg=True):
        super().__init__(backbone, pre)
        self.use_msg = use_msg
        dev0 = next(self.bb.parameters()).device
        with torch.no_grad():
            c4 = self.feats(torch.zeros(1, 3, *self.size, device=dev0))[3].shape[1]
        self.bit = nn.Conv2d(c4, nbits, 1)
        self.bit_from_decoder = False
        fc = getattr(self.bb, "fc", None)
        if isinstance(fc, nn.Linear) and fc.in_features == c4 and fc.out_features == nbits:
            with torch.no_grad():
                self.bit.weight.copy_(fc.weight.detach().float()[:, :, None, None])
                self.bit.bias.copy_(fc.bias.detach().float())
            self.bit_from_decoder = True
        self.msg_in = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 128, 1))
        nn.init.zeros_(self.msg_in[2].weight)
        nn.init.zeros_(self.msg_in[2].bias)

    def prep(self, img):
        kw = {} if self.mode in ("area", "nearest") else {"align_corners": False, "antialias": self.aa}
        x = F.interpolate(img[:, self.perm].float(), size=self.size, mode=self.mode, **kw)
        return x * self.a + self.b

    def forward(self, img, msg=None, return_bits=False):
        fs = self.feats(self.prep(img))
        z = self.bit(fs[3])                                         # [B, nbits, h, w] per-position bit logits
        p = self.lat[3](fs[3])
        if self.use_msg and msg is not None:
            agr = z.float() * (2 * msg.float() - 1)[:, :, None, None]
            c = torch.cat([torch.tanh(agr).mean(1, keepdim=True), torch.tanh(agr / 4).mean(1, keepdim=True)], 1)
            p = p + self.msg_in(c).to(p.dtype)
        for k in (2, 1, 0):
            p = F.interpolate(p, size=fs[k].shape[-2:], mode="bilinear", align_corners=False) + self.lat[k](fs[k])
        p = F.interpolate(p, size=(128, 128), mode="bilinear", align_corners=False)
        out = self.out(self.fuse(p))[:, 0]
        return (out, z) if return_bits else out


def bit_loss(z, msg, y):
    """z [B,nbits,h,w]; msg [B,nbits] TRUE bits; y [B,128,128] where the true watermark is."""
    w = F.adaptive_avg_pool2d(y[:, None], z.shape[-2:])
    on, off = (w > 0.75).float(), (w < 0.25).float()
    t = msg[:, :, None, None].expand_as(z)
    l_on = F.binary_cross_entropy_with_logits(z, t, reduction="none").mean(1, keepdim=True)
    l_off = F.binary_cross_entropy_with_logits(z, torch.full_like(z, 0.5), reduction="none").mean(1, keepdim=True)
    return (l_on * on).sum() / on.sum().clamp(min=1) + 0.5 * (l_off * off).sum() / off.sum().clamp(min=1)


@torch.no_grad()
def infer(net, I, M, dev, bs=64):
    """I uint8 [n,3,512,512], M [n,nbits]. Returns mask probability, message map (both [n,128,128]), bit accuracy."""
    net.eval()
    P, A, acc = [], [], []
    for s in range(0, len(I), bs):
        img = I[s:s + bs].to(dev).float() / 255
        m = M[s:s + bs].to(dev).float()
        with ac(dev):
            logit, z = net(img, m, return_bits=True)
        z = z.float()
        P.append(torch.sigmoid(logit.float()).cpu())
        agr = (torch.tanh(z) * (2 * m - 1)[:, :, None, None]).mean(1, keepdim=True)
        A.append(F.interpolate(agr, size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu())
        acc.append(((z.mean((2, 3)) > 0).float() == m).float().mean(1).cpu())
    return torch.cat(P), torch.cat(A), torch.cat(acc)


def pixel_auc(S, Y):
    """Mean over images of the AUC of map S for 'watermarked' (Y=1). Images with one class are skipped."""
    res = []
    for s, y in zip(S.flatten(1), Y.flatten(1) > 0.5):
        npos, nneg = int(y.sum()), int((~y).sum())
        if npos == 0 or nneg == 0:
            continue
        r = torch.empty_like(s)
        r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype)
        res.append(((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)).item())
    return float(np.mean(res)) if res else float("nan")


def pool_to_device(A, idx, dev, chunk=256):
    first = np.asarray(A[int(idx[0])])
    if first.dtype != np.uint8 or first.shape != (512, 512, 3):
        raise SystemExit(f"pool images are {first.dtype} {first.shape}, expected uint8 (512, 512, 3). Paste me test3_synth.py.")
    out = torch.empty((len(idx), 512, 512, 3), dtype=torch.uint8, device=dev)
    for s in range(0, len(idx), chunk):
        out[s:s + chunk] = torch.from_numpy(np.stack([np.asarray(A[int(i)]) for i in idx[s:s + chunk]])).to(dev)
    return out


class Gen:
    """Synthetic batches made on the device: seamless holes (real shapes) or keep-only windows."""

    def __init__(self, Cg, Wg, msgs, holes128, a, dev):
        self.C, self.W, self.msgs, self.holes, self.a, self.dev = Cg, Wg, msgs, holes128, a, dev
        self.ar = torch.arange(128, device=dev)[None]

    def windows(self, B, lo=12, hi=64):                             # 48..256 px, in 4-px cells
        d = self.dev
        ws, hs = torch.randint(lo, hi + 1, (B,), device=d), torch.randint(lo, hi + 1, (B,), device=d)
        x0 = (torch.rand(B, device=d) * (129 - ws)).long()
        y0 = (torch.rand(B, device=d) * (129 - hs)).long()
        wx = (self.ar >= x0[:, None]) & (self.ar < (x0 + ws)[:, None])
        wy = (self.ar >= y0[:, None]) & (self.ar < (y0 + hs)[:, None])
        return wy[:, :, None] & wx[:, None, :]

    def __call__(self, B):
        d, a = self.dev, self.a
        i = torch.randint(len(self.W), (B,), device=d)
        h = self.holes[torch.randint(len(self.holes), (B,), device=d)]          # True = hole
        r = torch.rand(B, 3, device=d) < 0.5                                    # flips of the hole SHAPE only
        h = torch.where(r[:, 0, None, None], h.flip(2), h)
        h = torch.where(r[:, 1, None, None], h.flip(1), h)
        h = torch.where(r[:, 2, None, None], h.transpose(1, 2), h)
        wm = ~h
        is_win = torch.rand(B, device=d) < a.win_frac
        wm = torch.where(is_win[:, None, None], self.windows(B), wm)            # [B,128,128] True = watermarked
        m512 = wm.repeat_interleave(4, 1).repeat_interleave(4, 2)[..., None]
        img = torch.where(m512, self.W[i], self.C[i]).permute(0, 3, 1, 2).float() / 255
        m_true = self.msgs[i]
        neg = torch.rand(B, device=d) < a.neg_frac
        m_in = torch.where(neg[:, None], torch.randint(0, 2, m_true.shape, device=d).float(), m_true)
        return img, wm.float(), m_true, m_in, neg


def write_submission(net, a, accs, kthr, thr, dev):
    meth = a.method
    test = [it for it in LocalisationDataset(a.data, "test") if str(it["method"]).lower() == meth]
    miss = [it["image_name"] for it in test if it["image_name"] not in accs]
    if miss:
        print(f"  [!] accs.npz has no entry for {len(miss)} test images (e.g. {miss[0]}): no file written. "
              f"Paste me step7b_submit.py and I'll match how it picks the group.", flush=True)
        return
    grp = [it for it in test if accs[it["image_name"]] > kthr[meth]]
    TI = torch.stack([to_input(u8(it["image"]), "rgb") for it in grp])
    TM = torch.stack([as_bits(bits_of(it["message"])) for it in grp])
    prob = infer(net, TI, TM, dev)[0]
    wrong = infer(net, TI, torch.randint(0, 2, TM.shape, generator=torch.Generator().manual_seed(1)).float(), dev)[0]
    sub = torch.load(a.base, map_location="cpu", weights_only=False)
    pos = {n: k for k, n in enumerate(list(sub["image_names"]))}
    bad = [it["image_name"] for it in grp if it["image_name"] not in pos]
    if bad:
        print(f"  [!] {len(bad)} names are not in {a.base} (e.g. {bad[0]}): no file written.", flush=True)
        return
    M = sub["masks"].clone()
    for it, p in zip(grp, prob):
        M[pos[it["image_name"]]] = (p > thr).to(M.dtype)
    sub["masks"] = M
    torch.save(sub, a.out)
    print(f"  wrote {a.out}: {len(grp)} {meth} mostly-watermarked test masks replaced (threshold {thr:.2f})")
    print(f"  predicted watermarked share on test: {(prob > thr).float().mean():.3f} (expect about 0.9)   "
          f"with a wrong message: {(wrong > thr).float().mean():.3f}" + (" (expect near 0)" if net.use_msg else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=12)               # unused here; kept because build_pool receives the args
    ap.add_argument("--steps", type=int, default=150)               # unused here
    ap.add_argument("--ft_epochs", type=int, default=20)            # unused here
    ap.add_argument("--ft_lr", type=float, default=1e-4)            # unused here
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bb_mult", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--minutes", type=float, default=40, help="training time, evaluations not counted")
    ap.add_argument("--eval_min", type=float, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--neg_frac", type=float, default=0.25, help="share of each batch with a wrong message and empty target")
    ap.add_argument("--win_frac", type=float, default=0.25, help="share of each batch with the watermark only in a window")
    ap.add_argument("--bit_w", type=float, default=0.5, help="weight of the per-position bit loss")
    ap.add_argument("--no_msg", action="store_true", help="control: no message channel, no negatives, no bit loss")
    ap.add_argument("--init", default="", help="checkpoint to start from (default: test3c_<method>_pretrained.pt if present; "
                                               "'none' = start from the decoder weights)")
    ap.add_argument("--base", default="submission_step7b.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--predict_only", action="store_true")
    a = ap.parse_args()
    dev, meth = a.device, a.method
    if a.no_msg:
        a.neg_frac, a.bit_w = 0.0, 0.0
    tag = f"test4_{meth}_{'nomsg' if a.no_msg else 'msg'}"
    a.out = a.out or f"submission_{tag}.pt"
    random.seed(0)
    torch.manual_seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    kthr = kind_thresholds(all_train, accs)
    mostly_wm = lambda it: accs[it["image_name"]] > kthr[str(it["method"]).lower()]
    real = [it for it in all_train if str(it["method"]).lower() == meth and mostly_wm(it)]
    R_img = [u8(it["image"]) for it in real]
    RY = torch.from_numpy(np.stack([mask128(it["mask128"]).astype(bool) for it in real])).float()
    RI = torch.stack([to_input(im, "rgb") for im in R_img])
    RM = torch.stack([as_bits(bits_of(it["message"])) for it in real])
    RMw = torch.randint(0, 2, RM.shape, generator=torch.Generator().manual_seed(7)).float()      # wrong messages (control)
    nbits = RM.shape[1]
    print(f"real group: {meth} mostly watermarked, n={len(real)}, {nbits} bits", flush=True)

    hole_m = [~mask128(it["mask128"]).astype(bool) for it in all_train
              if str(it["method"]).lower() != meth and mostly_wm(it)]
    holes = [np.repeat(np.repeat(h, 4, 0), 4, 1) for h in hole_m]                                # for test3b's sample()
    print(f"hole shapes: {len(holes)} real masks (other methods)", flush=True)

    wm = W._CLASSES[meth](Path("models").resolve(), dev)
    pre = measure_preproc(wm, R_img[0], dev)
    net = MsgNet(wm.net, pre, nbits, use_msg=not a.no_msg).to(dev)

    # --- check 1: the per-position bit head, averaged, must reproduce the kit decoder
    with torch.no_grad():
        net.eval()
        ref = raw_logits(wm, to_tensor(R_img[0])[None].to(dev).float()).float().flatten()
        mine = net(RI[:1].to(dev).float() / 255, RM[:1].to(dev), return_bits=True)[1].float().mean((2, 3)).flatten()
    if net.bit_from_decoder and ref.numel() == mine.numel():
        print(f"bit head check: max difference to the kit decoder {(ref - mine).abs().max().item():.4f}, "
              f"same bits {((ref > 0) == (mine > 0)).float().mean().item():.3f}  (expect about 0 and 1.000)", flush=True)
    else:
        print("  [!] the decoder does not end in a plain Linear layer on layer4: the bit head starts from scratch "
              "and is learned by the bit loss. The message channel will be slow to help.", flush=True)
    del wm

    init = a.init or f"test3c_{meth}_pretrained.pt"
    if a.predict_only:
        init = f"{tag}.pt"
    warm = False
    if init == "none":
        print("starting from the decoder weights (like test 3c)", flush=True)
    elif Path(init).exists():
        warm = True
        sd = torch.load(init, map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        old = [k for k in missing if not k.startswith(("bit.", "msg_in."))]
        print(f"start from {init}" + (f"   [!] not found in file: {old[:4]}" if old else "")
              + (f"   [!] unused keys: {list(unexpected)[:4]}" if unexpected else ""), flush=True)
    elif a.predict_only or a.init:
        raise SystemExit(f"{init} not found")
    else:
        print(f"{init} not found: starting from the decoder weights (like test 3c)", flush=True)

    def real_eval():
        P, A, acc = infer(net, RI, RM, dev)
        Pw, Aw, _ = infer(net, RI, RMw, dev)
        sc, t = best_score(P, RY)
        return dict(score=float(sc), thr=float(t), share=(P > t).float().mean().item(), wshare=(Pw > t).float().mean().item(),
                    auc=pixel_auc(A, RY), wauc=pixel_auc(Aw, RY), acc=acc.mean().item())

    if a.predict_only:
        r = real_eval()
        print(f"REAL {r['score']:.3f} (threshold {r['thr']:.2f})", flush=True)
        write_submission(net, a, accs, kthr, r["thr"], dev)
        return

    if not (Path(f"cache_synth_{meth}") / "done.txt").exists():
        raise SystemExit("synthetic pool missing: run test3_synth.py --check first")
    C, Wm, msgs, tr, val = build_pool(a, None, [], nbits)
    tr = np.asarray(tr)

    # --- check 2: my image format equals to_input()
    x0 = np.asarray(Wm[int(tr[0])])
    if not torch.equal(to_input(x0, "rgb").float(), torch.from_numpy(x0).permute(2, 0, 1).float()):
        raise SystemExit("to_input(img, 'rgb') is not a plain HWC->CHW uint8 copy. Paste me test3_synth.py.")

    rng = np.random.default_rng(11)
    vi = rng.choice(val, 300, replace=len(val) < 300)
    vs = [sample(int(i), C, Wm, rng, holes) for i in vi]                                          # test 3b's generator
    VI = torch.stack([to_input(s[0], "rgb") for s in vs])
    VY = torch.from_numpy(np.stack([s[1] for s in vs])).float()
    VM = torch.stack([as_bits(msgs[int(i)]) for i in vi])
    del vs

    t_load = time.time()
    Cg, Wg = pool_to_device(C, tr, dev), pool_to_device(Wm, tr, dev)
    Mg = torch.stack([as_bits(msgs[int(i)]) for i in tr]).to(dev)
    Hg = torch.from_numpy(np.stack(hole_m)).to(dev)
    gen = Gen(Cg, Wg, Mg, Hg, a, dev)
    print(f"pool on {dev}: {len(tr)} images, {(Cg.numel() + Wg.numel()) / 2**30:.1f} GB, loaded in {time.time() - t_load:.0f}s", flush=True)

    # --- check 3: the pool's messages are what the decoder reads from the pool's images
    if net.bit_from_decoder:
        k = torch.randperm(len(tr))[:32].to(dev)
        acc = infer(net, Wg[k].permute(0, 3, 1, 2), Mg[k], dev)[2].mean().item()
        print(f"pool message check: bit accuracy {acc:.3f} on fully watermarked pool images "
              + ("(below 0.97 is normal after a warm start: 7b moved the encoder; the bit loss restores it)" if warm
                 else "(expect about 0.97)"), flush=True)
        if acc < 0.8 and not warm:
            raise SystemExit("the pool's messages do not match its images. Paste me test3_synth.py (build_pool).")

    def report(label, loss, speed):
        r = real_eval()
        ss = best_score(infer(net, VI, VM, dev)[0], VY)[0]
        msg = (f"wrong-msg share {r['wshare']:.2f} | msg-map AUC {r['auc']:.3f} (wrong msg {r['wauc']:.3f}) | bit acc {r['acc']:.3f}"
               if net.use_msg else f"msg-map AUC {r['auc']:.3f}")
        print(f"[{label}] {speed}loss {loss:.3f} | synthetic {ss:.3f} | REAL {r['score']:.3f} (thr {r['thr']:.2f}, share {r['share']:.2f}) | {msg}",
              flush=True)
        return r

    ref = REF.get(meth, 0)
    print(f"\n=== training {a.minutes:g} min: batch {a.batch}, wrong-message {a.neg_frac:.0%}, windows {a.win_frac:.0%}, "
          f"bit loss {a.bit_w}, message channel {'on' if net.use_msg else 'OFF'} ===")
    print(f"  score to beat on REAL: {ref:.3f}. 'wrong-msg share' should fall towards 0; 'msg-map AUC' should rise above its start value.", flush=True)
    hist = [report("start ", float("nan"), "")]

    bbp = [p for n, p in net.named_parameters() if n.startswith(("bb.", "bit."))]
    head = [p for n, p in net.named_parameters() if not n.startswith(("bb.", "bit."))]
    base_lr = [a.lr * a.bb_mult, a.lr]
    opt = torch.optim.AdamW([{"params": bbp, "lr": base_lr[0]}, {"params": head, "lr": base_lr[1]}], weight_decay=1e-4)
    T, spent, views, tot, n, next_eval, v0, s0 = a.minutes * 60, 0.0, 0, 0.0, 0, a.eval_min * 60, 0, 0.0
    while spent < T:
        t0 = time.time()
        frac = spent / T
        sc = min(1.0, frac / 0.03 + 0.05) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * sc
        net.train()
        img, y, m_true, m_in, neg = gen(a.batch)
        with ac(dev):
            logit, z = net(img, m_in, return_bits=True)
        logit, z, pos = logit.float(), z.float(), ~neg
        loss = logit.sum() * 0
        if pos.any():
            loss = loss + loss_fn(logit[pos], y[pos]) * pos.float().mean()
        if neg.any():
            loss = loss + F.binary_cross_entropy_with_logits(logit[neg], torch.zeros_like(logit[neg])) * neg.float().mean()
        if a.bit_w > 0:
            loss = loss + a.bit_w * bit_loss(z, m_true, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
        views += len(img)
        spent += time.time() - t0
        if spent >= next_eval or spent >= T:
            speed = f"{views / 1000:.0f}k views, {(views - v0) / max(spent - s0, 1e-6):.0f} img/s | "
            hist.append(report(f"{spent / 60:4.0f} min", tot / max(n, 1), speed))
            torch.save(net.state_dict(), f"{tag}.pt")
            tot, n, v0, s0 = 0.0, 0, views, spent
            next_eval += a.eval_min * 60

    last = hist[-2:] if len(hist) > 2 else hist[-1:]
    ok_score = all(r["score"] >= ref + 0.02 for r in last)
    ok_msg = (not net.use_msg) or hist[-1]["wshare"] <= 0.05
    print(f"\n=== RESULT ({meth} mostly watermarked, n={len(RI)}) ===")
    print(f"  start {hist[0]['score']:.3f} -> end {hist[-1]['score']:.3f}   (7b: {ref:.3f}; rule: last two evaluations >= {ref + 0.02:.3f})")
    print(f"  score rule: {'PASS' if ok_score else 'FAIL'}" + ("" if not net.use_msg else
          f"   wrong-message rule (share <= 0.05): {'PASS' if ok_msg else 'FAIL'}"))
    print(f"  {'Worth submitting if the test share below is about 0.9.' if ok_score and ok_msg else 'Keep 7b.'} Paste me this log.", flush=True)
    try:
        write_submission(net, a, accs, kthr, hist[-1]["thr"], dev)
    except Exception as e:                                                                        # the checkpoint is already saved
        print(f"  [!] could not write the submission file ({type(e).__name__}: {e}). Checkpoint {tag}.pt is saved; "
              f"paste me this line and step7b_submit.py.", flush=True)


if __name__ == "__main__":
    main()