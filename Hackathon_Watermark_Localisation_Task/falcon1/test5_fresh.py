"""
Test 5: test 4 with the three faults of that run removed.

  1. Starts from the DECODER weights, where bit reading is intact (test 4 started from 7b,
     whose encoder reads bits at chance).
  2. Fresh messages all the time: while training, pool images are re-embedded with a new
     random message through the same kit call as build_pool. A photo no longer always
     carries the same message, so the message cannot be memorised from the photo.
  3. Decoy holes: some holes are filled with the same photo carrying ANOTHER message.
     The target is still "hole", so the network has to read the message locally.

Needs test4_msg.py in the same folder. Run inside falcon1/ :
  PYTHONPATH=.. python -u test5_fresh.py --data ../release --minutes 40
  PYTHONPATH=.. python -u test5_fresh.py --data ../release --predict_only

Writes test5_<method>_<tag>.pt and submission_test5_<method>_<tag>.pt. It never submits.
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128, bits_of, to_tensor
from step3_train import loss_fn
from test3_synth import build_pool, u8, to_input, embed
from test3b_synth import sample, best_score
from test3c_dec import measure_preproc, raw_logits
from test4_msg import MsgNet, Gen, infer, pixel_auc, as_bits, ac, pool_to_device, write_submission

REF = {"trustmark": 0.578}


@torch.no_grad()
def kit_acc(wm, imgs, msgs, dev):
    """Bit accuracy of the untouched kit decoder on whole images (uint8 HWC numpy)."""
    out = []
    for img, m in zip(imgs, msgs):
        z = raw_logits(wm, to_tensor(np.ascontiguousarray(img))[None].to(dev).float()).float().flatten().cpu()
        m = torch.as_tensor(np.asarray(m)).float().flatten()
        L = min(len(z), len(m))
        out.append(((z[:L] > 0).float() == m[:L]).float().mean().item())
    return float(np.mean(out))


def bit_loss(z, m_cur, m_prev, y, decoy):
    """Per-position bits: the queried message where it is, the OTHER message inside decoy holes, undecided in clean areas."""
    w = F.adaptive_avg_pool2d(y[:, None], z.shape[-2:])
    on, off = (w > 0.75).float(), (w < 0.25).float()
    t_on = m_cur[:, :, None, None].expand_as(z)
    t_off = torch.where(decoy[:, None, None, None], m_prev[:, :, None, None].expand_as(z), torch.full_like(z, 0.5))
    l_on = F.binary_cross_entropy_with_logits(z, t_on, reduction="none").mean(1, keepdim=True)
    l_off = F.binary_cross_entropy_with_logits(z, t_off, reduction="none").mean(1, keepdim=True)
    return (l_on * on).sum() / on.sum().clamp(min=1) + 0.5 * (l_off * off).sum() / off.sum().clamp(min=1)


class Gen5(Gen):
    """Pool on the device with a current and a previous embedding per photo; one slot is re-embedded per call to refresh()."""

    def __init__(self, Cg, Wg, Mg, holes128, a, dev):
        super().__init__(Cg, Wg, Mg, holes128, a, dev)
        self.Wp, self.Mp = Wg.clone(), Mg.clone()
        self.has_prev = torch.zeros(len(Wg), dtype=torch.bool, device=dev)
        self.order, self.ptr, self.n_fresh = torch.randperm(len(Wg)).tolist(), 0, 0
        self.rng = np.random.default_rng(123)

    @torch.no_grad()
    def refresh(self, wm):
        s = self.order[self.ptr % len(self.order)]
        self.ptr += 1
        m = self.rng.integers(0, 2, self.msgs.shape[1]).astype(np.uint8)
        c = np.ascontiguousarray(self.C[s].cpu().numpy())
        w = embed(wm, c, m)                                          # same call as build_pool; uint8 HWC
        self.Wp[s], self.Mp[s] = self.W[s], self.msgs[s]
        self.has_prev[s] = True
        self.W[s] = torch.from_numpy(w).to(self.dev)
        self.msgs[s] = torch.from_numpy(m).float().to(self.dev)
        self.n_fresh += 1
        return w, m

    def __call__(self, B):
        d, a = self.dev, self.a
        i = torch.randint(len(self.W), (B,), device=d)
        h = self.holes[torch.randint(len(self.holes), (B,), device=d)]           # True = hole
        r = torch.rand(B, 3, device=d) < 0.5                                     # flips of the hole SHAPE only
        h = torch.where(r[:, 0, None, None], h.flip(2), h)
        h = torch.where(r[:, 1, None, None], h.flip(1), h)
        h = torch.where(r[:, 2, None, None], h.transpose(1, 2), h)
        kind = torch.rand(B, device=d)
        is_win = kind < a.win_frac
        decoy = (kind >= a.win_frac) & (kind < a.win_frac + a.decoy_frac) & self.has_prev[i]
        wm = torch.where(is_win[:, None, None], self.windows(B), ~h)             # True = carries the QUERIED message
        m512 = wm.repeat_interleave(4, 1).repeat_interleave(4, 2)[..., None]
        other = torch.where(decoy[:, None, None, None], self.Wp[i], self.C[i])   # what fills the rest
        img = torch.where(m512, self.W[i], other).permute(0, 3, 1, 2).float() / 255
        m_true, m_prev = self.msgs[i], self.Mp[i]
        neg = torch.rand(B, device=d) < a.neg_frac
        m_in = torch.where(neg[:, None], torch.randint(0, 2, m_true.shape, device=d).float(), m_true)
        return img, wm.float(), m_true, m_prev, m_in, neg, decoy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bb_mult", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--minutes", type=float, default=40, help="training + embedding time, evaluations not counted")
    ap.add_argument("--eval_min", type=float, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--embed_share", type=float, default=0.5, help="share of the time spent embedding fresh messages")
    ap.add_argument("--neg_frac", type=float, default=0.10, help="whole-image wrong message, empty target")
    ap.add_argument("--win_frac", type=float, default=0.15, help="watermark only inside a window")
    ap.add_argument("--decoy_frac", type=float, default=0.15, help="holes filled with the same photo carrying another message")
    ap.add_argument("--bit_w", type=float, default=0.5)
    ap.add_argument("--init", default="", help="optional checkpoint; default is the decoder weights")
    ap.add_argument("--tag", default="a", help="suffix for output files, so two runs do not overwrite each other")
    ap.add_argument("--base", default="submission_step7b.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--predict_only", action="store_true")
    a = ap.parse_args()
    dev, meth = a.device, a.method
    tag = f"test5_{meth}_{a.tag}"
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
    RMw = torch.randint(0, 2, RM.shape, generator=torch.Generator().manual_seed(7)).float()
    nbits = RM.shape[1]
    print(f"real group: {meth} mostly watermarked, n={len(real)}, {nbits} bits", flush=True)
    hole_m = [~mask128(it["mask128"]).astype(bool) for it in all_train
              if str(it["method"]).lower() != meth and mostly_wm(it)]
    holes = [np.repeat(np.repeat(h, 4, 0), 4, 1) for h in hole_m]

    wm = W._CLASSES[meth](Path("models").resolve(), dev)
    pre = measure_preproc(wm, R_img[0], dev)
    net = MsgNet(wm.net, pre, nbits).to(dev)
    if not net.bit_from_decoder:
        raise SystemExit("the decoder does not end in a plain Linear layer on layer4; this script needs that. Paste me this line.")

    init = f"{tag}.pt" if a.predict_only else a.init
    if init:
        if not Path(init).exists():
            raise SystemExit(f"{init} not found")
        missing, unexpected = net.load_state_dict(torch.load(init, map_location="cpu", weights_only=False), strict=False)
        print(f"start from {init}", flush=True)
    else:
        print("start from the decoder weights", flush=True)

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

    rng = np.random.default_rng(11)
    vi = rng.choice(val, 300, replace=len(val) < 300)
    vs = [sample(int(i), C, Wm, rng, holes) for i in vi]                          # test 3b's generator, unseen covers
    VI = torch.stack([to_input(s[0], "rgb") for s in vs])
    VY = torch.from_numpy(np.stack([s[1] for s in vs])).float()
    VM = torch.stack([as_bits(msgs[int(i)]) for i in vi])
    del vs

    t0 = time.time()
    Cg, Wg = pool_to_device(C, tr, dev), pool_to_device(Wm, tr, dev)
    Mg = torch.stack([as_bits(msgs[int(i)]) for i in tr]).to(dev)
    gen = Gen5(Cg, Wg, Mg, torch.from_numpy(np.stack(hole_m)).to(dev), a, dev)
    print(f"pool on {dev}: {len(tr)} photos, {3 * Cg.numel() / 2**30:.1f} GB (clean, current, previous), loaded in {time.time() - t0:.0f}s", flush=True)

    # --- check: my fresh embeddings are read by the kit decoder as well as the cached pool is
    k = [int(i) for i in tr[:8]]
    acc_pool = kit_acc(wm, [np.asarray(Wm[i]) for i in k], [msgs[i] for i in k], dev)
    t0 = time.time()
    fresh = [gen.refresh(wm) for _ in range(8)]
    rate = 8 / max(time.time() - t0, 1e-6)
    acc_fresh = kit_acc(wm, [f[0] for f in fresh], [f[1] for f in fresh], dev)
    print(f"embedding check: kit decoder reads the cached pool at {acc_pool:.3f} and my fresh embeddings at {acc_fresh:.3f} "
          f"(both about 0.97+); embedder speed {rate:.0f} images/s", flush=True)
    if acc_pool > 0.9 and acc_fresh < 0.9:
        raise SystemExit("fresh embeddings are not read back correctly. Paste me this line.")

    def report(label, loss, extra):
        r = real_eval()
        P, _, sacc = infer(net, VI, VM, dev)
        ss = best_score(P, VY)[0]
        print(f"[{label}] {extra}loss {loss:.3f} | synthetic {ss:.3f} | REAL {r['score']:.3f} (thr {r['thr']:.2f}, share {r['share']:.2f}) | "
              f"wrong-msg share {r['wshare']:.2f} | msg-map AUC {r['auc']:.3f} (wrong msg {r['wauc']:.3f}) | "
              f"bit acc real {r['acc']:.3f} synthetic {sacc.mean().item():.3f}", flush=True)
        return r

    ref = REF.get(meth, 0)
    print(f"\n=== training {a.minutes:g} min: batch {a.batch}, {a.embed_share:.0%} of the time embedding fresh messages, "
          f"windows {a.win_frac:.0%}, decoy holes {a.decoy_frac:.0%}, wrong message {a.neg_frac:.0%}, bit loss {a.bit_w} ===")
    print(f"  score to beat on REAL: {ref:.3f}. At start: bit acc about 0.97, REAL low (the mask head is new).", flush=True)
    print("  Healthy run: REAL rises past the reference, share stays near 0.9, bit acc stays high, msg-map AUC rises.", flush=True)
    hist = [report("start ", float("nan"), "")]

    bbp = [p for n, p in net.named_parameters() if n.startswith(("bb.", "bit."))]
    head = [p for n, p in net.named_parameters() if not n.startswith(("bb.", "bit."))]
    base_lr = [a.lr * a.bb_mult, a.lr]
    opt = torch.optim.AdamW([{"params": bbp, "lr": base_lr[0]}, {"params": head, "lr": base_lr[1]}], weight_decay=1e-4)
    T, t_tr, t_em, views, tot, n = a.minutes * 60, 0.0, 0.0, 0, 0.0, 0
    ratio = a.embed_share / max(1e-6, 1 - a.embed_share)
    next_eval = a.eval_min * 60
    while t_tr + t_em < T:
        t0 = time.time()
        frac = (t_tr + t_em) / T
        sc = min(1.0, frac / 0.03 + 0.05) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * sc
        net.train()
        img, y, m_true, m_prev, m_in, neg, decoy = gen(a.batch)
        with ac(dev):
            logit, z = net(img, m_in, return_bits=True)
        logit, z, pos = logit.float(), z.float(), ~neg
        loss = logit.sum() * 0
        if pos.any():
            loss = loss + loss_fn(logit[pos], y[pos]) * pos.float().mean()
        if neg.any():
            loss = loss + F.binary_cross_entropy_with_logits(logit[neg], torch.zeros_like(logit[neg])) * neg.float().mean()
        if a.bit_w > 0:
            loss = loss + a.bit_w * bit_loss(z, m_true, m_prev, y, decoy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
        views += len(img)
        t_tr += time.time() - t0
        while a.embed_share > 0 and t_em < ratio * t_tr:
            t0 = time.time()
            gen.refresh(wm)
            t_em += time.time() - t0
        if t_tr + t_em >= next_eval or t_tr + t_em >= T:
            extra = (f"{views / 1000:.0f}k views ({views / max(t_tr, 1e-6):.0f} img/s while training) | "
                     f"{gen.n_fresh / 1000:.1f}k fresh embeddings ({gen.n_fresh / max(t_em, 1e-6):.0f}/s, "
                     f"{views / max(gen.n_fresh, 1):.1f} views each) | ")
            hist.append(report(f"{(t_tr + t_em) / 60:4.0f} min", tot / max(n, 1), extra))
            torch.save(net.state_dict(), f"{tag}.pt")
            tot, n = 0.0, 0
            next_eval += a.eval_min * 60

    last = hist[-2:] if len(hist) > 2 else hist[-1:]
    ok_score = all(r["score"] >= ref + 0.02 for r in last)
    ok_msg = hist[-1]["wshare"] <= 0.05
    ok_share = hist[-1]["share"] >= 0.8
    print(f"\n=== RESULT ({meth} mostly watermarked, n={len(RI)}) ===")
    print(f"  REAL at the end {hist[-1]['score']:.3f}, best during the run {max(r['score'] for r in hist):.3f}   "
          f"(7b: {ref:.3f}; rule: last two evaluations >= {ref + 0.02:.3f})")
    print(f"  score rule: {'PASS' if ok_score else 'FAIL'}   wrong-message rule (share <= 0.05): {'PASS' if ok_msg else 'FAIL'}   "
          f"share rule (true message >= 0.8): {'PASS' if ok_share else 'FAIL'}")
    print(f"  {'Worth submitting if the test share below is about 0.9.' if ok_score and ok_msg and ok_share else 'Keep 7b.'} Paste me this log.", flush=True)
    try:
        write_submission(net, a, accs, kthr, hist[-1]["thr"], dev)
    except Exception as e:
        print(f"  [!] could not write the submission file ({type(e).__name__}: {e}). Checkpoint {tag}.pt is saved; "
              f"paste me this line and step7b_submit.py.", flush=True)


if __name__ == "__main__":
    main()