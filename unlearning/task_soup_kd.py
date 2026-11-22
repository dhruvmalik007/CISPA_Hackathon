"""Consistency distillation of the best soup (1 GPU, ~10 min on A100).

The best leaderboard checkpoint (runs/soup_4way_tb10_s42.pt, 0.1956) forgets the
altered copies much more than the exact images (local: exact CE 2.26, altered CE 4.61,
original on altered 0.61). Gold never saw either, so it should treat both alike.

Target for every forget image x (fixed, computed once on the EXACT image, T=2):
    q(x) = (1 - hedge) * softmax(soup(x) / 2) + hedge * softmax(orig(x) / 2)
The student (initialised from the soup) is trained with forward KL(q || p_student),
the direction the server uses for prediction_score, on the exact image AND on altered
copies of it. Retain: KL(p_orig || p_student) at T=2 on mosaics, as in the retain score.

  python task_soup_kd.py --hedge 0.1 --out runs/skd_h01
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import alter, make_mosaics, retain_proxy_score, save_submission
from task_template import DEVICE, load_forget_images, load_model, predict, preprocess

T = 2.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("model.pt"))
    p.add_argument("--teacher", type=Path, default=Path("runs/soup_4way_tb10_s42.pt"))
    p.add_argument("--start", type=Path, default=None, help="Student init (default: --teacher)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hedge", type=float, default=0.1, help="Weight of the original's distribution in the target")
    p.add_argument("--alt-hedge", type=float, default=None, help="Hedge used for altered copies (default: --hedge)")
    p.add_argument("--same-input-target", action="store_true",
                   help="Target on an altered copy = teacher's own output on that copy (forget behaviour held fixed)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--aug-prob", type=float, default=0.5)
    p.add_argument("--proxy-weight", type=float, default=2.0)
    p.add_argument("--proxy-batch", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--save-steps", type=int, nargs="*", default=[150, 300])
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.start = args.start or args.teacher
    args.alt_hedge = args.hedge if args.alt_hedge is None else args.alt_hedge
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    images = load_forget_images()
    orig = load_model(args.model).requires_grad_(False)
    teacher = load_model(args.teacher).requires_grad_(False)
    orig_logits = predict(orig, images)
    labels = orig_logits.argmax(1)
    p_orig = F.softmax(orig_logits / T, 1)
    p_teach = F.softmax(predict(teacher, images) / T, 1)
    if not args.same_input_target:
        teacher = None
    q_exact = (1 - args.hedge) * p_teach + args.hedge * p_orig
    q_alt = (1 - args.alt_hedge) * p_teach + args.alt_hedge * p_orig

    student = load_model(args.start).train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    ev = torch.randperm(len(images), generator=torch.Generator().manual_seed(1234))[:args.eval_n]
    ev_ex, ev_mx = images[ev], alter(images[ev], "mix", seed=98765)
    ev_mos = make_mosaics(images, 512, torch.Generator().manual_seed(4321))
    o_mos = predict(orig, ev_mos)

    def evaluate(step):
        res = {"step": step}
        for name, x, q in (("exact", ev_ex, q_exact[ev]), ("mix", ev_mx, q_alt[ev])):
            z = predict(student, x)
            ce = F.cross_entropy(z, labels[ev], reduction="none")
            ps = F.softmax(z / T, 1)
            res[name] = {"acc": (z.argmax(1) == labels[ev]).float().mean().item(),
                         "ce_mean": ce.mean().item(), "ce_std": ce.std().item(),
                         "p_label_T2": ps.gather(1, labels[ev, None]).mean().item(),
                         "kl_to_target": (q * (q.clamp_min(1e-12).log() - ps.clamp_min(1e-12).log())).sum(1).mean().item()}
        res["retain_proxy"] = retain_proxy_score(predict(student, ev_mos), o_mos)
        with open(args.out / "eval.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        e, m = res["exact"], res["mix"]
        print(f"[eval {step:04d}] exact acc={e['acc']:.3f} CE={e['ce_mean']:.3f}/{e['ce_std']:.3f} pL={e['p_label_T2']:.3f} "
              f"kl={e['kl_to_target']:.3f} | mix acc={m['acc']:.3f} CE={m['ce_mean']:.3f}/{m['ce_std']:.3f} "
              f"pL={m['p_label_T2']:.3f} kl={m['kl_to_target']:.3f} | retain={res['retain_proxy']:.4f}", flush=True)
        student.train()

    evaluate(0)
    step = 0
    while step < args.max_steps:
        for idx in torch.randperm(len(images), generator=rng).split(args.batch):
            x = images[idx].clone()
            aug = torch.rand(len(idx), generator=rng) < args.aug_prob
            if aug.any():
                x[aug] = alter(x[aug], "mix", seed=args.seed * 100003 + step)
            aug = aug.to(DEVICE)
            q = torch.where(aug[:, None], q_alt[idx].to(DEVICE), q_exact[idx].to(DEVICE))
            mos = preprocess(make_mosaics(images, args.proxy_batch, rng))
            with torch.no_grad(), amp():
                p_mos = F.softmax(orig(mos).float() / T, 1)
                if teacher is not None:
                    q = F.softmax(teacher(preprocess(x)).float() / T, 1)
            with amp():
                z = student(torch.cat([preprocess(x), mos])).float()
            log_s = F.log_softmax(z[:len(idx)] / T, 1)
            loss_f = F.kl_div(log_s, q, reduction="batchmean")
            loss_r = F.kl_div(F.log_softmax(z[len(idx):] / T, 1), p_mos, reduction="batchmean")
            loss = loss_f + args.proxy_weight * loss_r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{args.max_steps} forget_kl={loss_f.item():.4f} retain_kl={loss_r.item():.4f}", flush=True)
            if step in args.save_steps or step == args.max_steps:
                save_submission(student, args.out / f"step{step:04d}.pt")
                evaluate(step)
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
