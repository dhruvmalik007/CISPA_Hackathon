#!/usr/bin/env python3
"""Train a contrastive verifier on the proxy corpus.

The task's hard part is invariance: same checkpoint served with a different system prompt or
temperature must embed to the same point, while a different checkpoint must embed elsewhere. Plain
cross-entropy on text cannot express that; Supervised Contrastive (SupCon) loss can, because it pulls
all same-label items together while pushing every other label away, including the hard ones.

Positives come from `build_training_set.py`'s output: same (model, prompt), different disguise. That is
exactly the invariance the competition rewards, and the proxy zoo measured an AUROC 0.887 for it with a
trivial metric, so there is something to learn.

HONEST LIMITATION, stated here so it is not lost: this corpus has **zero hard negatives** - no
same-family different-quantization pairs, because Doubleword's catalog has no such pairs. So this
trainer can only learn "ignore deployment-level variation" and "separate unrelated models". It cannot
learn the fine-grained separation that carries 90% of the competition score. That is what the local GPU
rows (KGW watermark, int8/int4, LoRA) are for.

Usage:
    python training/train_verifier.py --pairs training/proxy_pairs.jsonl --smoke
    python training/train_verifier.py --pairs training/proxy_pairs.jsonl \
        --model microsoft/deberta-v3-base --epochs 3 --out training/verifier.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_batches(rows: list[dict], batch_size: int, rng: random.Random) -> list[list[dict]]:
    """Build batches where each label appears more than once, as SupCon requires.

    A batch of all-distinct labels has no positives and the loss degenerates to zero. So batching is
    done by LABEL: pick a few (model, prompt) groups and draw several disguises from each. That
    guarantees every anchor has at least one positive, and the negatives in the batch are the other
    models on the SAME prompt — the hardest negatives available in this corpus.
    """
    by_label: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_label[(row["model"], row["prompt_id"])].append(row)

    groups = [(label, members) for label, members in by_label.items() if len(members) >= 2]
    if not groups:
        return []
    rng.shuffle(groups)

    # Aim for ~16 groups per batch with ~3 members each, then trim/pad to batch_size.
    batches: list[list[dict]] = []
    current: list[dict] = []
    for _label, members in groups:
        take = members[: max(2, min(len(members), 3))]
        current.extend(take)
        if len(current) >= batch_size:
            batches.append(current[:batch_size])
            current = []
    if len(current) >= 4:
        batches.append(current)
    return batches


def supcon_loss(embeddings, labels, temperature: float = 0.07):
    """Supervised contrastive loss (Khosla et al.), computed without a scipy dependency."""
    import torch
    import torch.nn.functional as F

    device = embeddings.device
    n = embeddings.shape[0]
    if n < 2:
        return torch.zeros((), device=device, requires_grad=True)
    norms = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-9)
    z = embeddings / norms
    sim = z @ z.t() / temperature
    # For numerical stability, mask self-similarity before log-softmax.
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    log_prob = log_prob.masked_fill(self_mask, 0.0)
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    labels = labels.view(-1, 1)
    positive = (labels == labels.t()) & ~self_mask
    counts = positive.sum(dim=1)
    valid = counts > 0
    if not bool(valid.any()):
        return torch.zeros((), device=device, requires_grad=True)
    mean_log_prob = (log_prob * positive).sum(dim=1) / counts.clamp_min(1)
    return -mean_log_prob[valid].mean()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default="training/proxy_pairs.jsonl")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out", default="training/verifier.pt")
    parser.add_argument("--smoke", action="store_true", help="20 steps on a small slice, to prove the loop runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = ROOT / pairs_path
    if not pairs_path.is_file():
        print(f"missing pairs file: {pairs_path}", file=sys.stderr)
        return 2

    rows = load_pairs(pairs_path)
    if not rows:
        print(f"no rows in {pairs_path}", file=sys.stderr)
        return 2
    if args.smoke:
        rows = rows[:200]

    labels_seen = {(r["model"], r["prompt_id"]) for r in rows}
    print(f"rows: {len(rows)}  distinct (model,prompt) labels: {len(labels_seen)}")
    print(f"models: {sorted({r['model'] for r in rows})}")
    print(f"disguises: {sorted({r['disguise'] for r in rows})}")

    rng = random.Random(args.seed)
    batches = build_batches(rows, args.batch_size, rng)
    if not batches:
        print("no usable batches: need at least one (model,prompt) with 2+ disguises", file=sys.stderr)
        return 2
    print(f"batches: {len(batches)}  (avg {np.mean([len(b) for b in batches]):.0f} rows)")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        print(f"training needs torch + transformers: {exc}", file=sys.stderr)
        return 2

    device = "cuda"
    print(f"device: {device}  base model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = None

    steps_per_epoch = 20 if args.smoke else len(batches)
    history: list[float] = []
    for epoch in range(args.epochs):
        losses: list[float] = []
        for step, batch in enumerate(batches[:steps_per_epoch]):
            enc = tokenizer(
                [r["text"] for r in batch],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Label = (model, prompt): same model on the same prompt is a positive regardless of disguise.
            index: dict[tuple[str, str], int] = {}
            ids = []
            for row in batch:
                key = (row["model"], row["prompt_id"])
                if key not in index:
                    index[key] = len(index)
                ids.append(index[key])
            label_tensor = torch.tensor(ids, device=device, dtype=torch.long)

            def forward():
                hidden = model(**enc).last_hidden_state[:, 0]
                return supcon_loss(hidden, label_tensor, args.temperature)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = forward()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = forward()
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            value = float(loss.detach().cpu())
            losses.append(value)
            history.append(value)
            if args.smoke or (step + 1) % 20 == 0:
                print(f"  epoch {epoch + 1}/{args.epochs} step {step + 1} loss {value:.4f}")
            if args.smoke and step + 1 >= 20:
                break
        if losses:
            print(f"epoch {epoch + 1}: mean loss {np.mean(losses):.4f}")

    # A decreasing loss is the minimum evidence the loop learned anything.
    if len(history) >= 10:
        first, last = np.mean(history[:5]), np.mean(history[-5:])
        print(f"\nloss trend: first5 {first:.4f} -> last5 {last:.4f}  ({last - first:+.4f})")
        if last > first:
            print("WARNING: loss did not decrease; the model is not learning from this corpus")
    else:
        print("\nWARNING: too few steps to show a loss trend")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"base_model": args.model, "state_dict": model.state_dict()}, out)
    print(f"\nwrote {out}")
    print("next:  python training/eval_transfer.py --verifier " + str(out) + " --answers data/ref_power.jsonl")
    print("       that is THE GATE - if it fails, do not spend ep-* budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
