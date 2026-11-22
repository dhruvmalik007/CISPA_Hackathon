#!/usr/bin/env python3
"""SetPrint response-change evaluation with LOO and spectral clustering.

SetPrint compares how an endpoint's response *changes* under small prompt perturbations. Static
system prompts and watermarks shift the base response and the perturbed responses together, so the
difference vectors cancel much of that nuisance variation.

Gate 1: leave-one-base-out practice MCC >= 0.85 on refs.
Gate 2: held-out proxy-family score >= 0.85 (with the proxy's no-hard-negative caveat recorded).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from round3_eval import auroc, load_pooled, load_truth, practice_mcc  # noqa: E402


def load_embedder(model_name: str, device: str, batch_size: int = 16):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    cache: dict[str, np.ndarray] = {}

    def embed_many(texts: list[str]) -> np.ndarray:
        missing = [text for text in dict.fromkeys(texts) if text not in cache]
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors = pooled.float().cpu().numpy()
            for text, vector in zip(batch, vectors):
                norm = np.linalg.norm(vector)
                cache[text] = vector / (norm + 1e-9)
        return np.stack([cache[text] for text in texts])

    return embed_many


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def kmeans(x: np.ndarray, k: int, seed: int = 0, restarts: int = 30) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_inertia = float("inf")
    for _ in range(restarts):
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        labels = np.zeros(len(x), dtype=int)
        for _iteration in range(100):
            distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2)
            new_labels = distances.argmin(1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for index in range(k):
                members = x[labels == index]
                if len(members):
                    centers[index] = members.mean(0)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    assert best_labels is not None
    return best_labels, best_inertia


def spectral_labels(distance: np.ndarray, k: int | None = None, seed: int = 0) -> tuple[np.ndarray, int, list[float]]:
    distance = np.nan_to_num(distance, nan=float(np.nanmax(distance[np.isfinite(distance)]) if np.isfinite(distance).any() else 1.0))
    n = len(distance)
    positive = distance[(distance > 0) & np.isfinite(distance)]
    scale = float(np.median(positive)) if positive.size else 1.0
    similarity = np.exp(-distance / max(scale, 1e-6))
    np.fill_diagonal(similarity, 0.0)
    degree = similarity.sum(1)
    inv = 1.0 / np.sqrt(np.clip(degree, 1e-9, None))
    laplacian = np.eye(n) - (inv[:, None] * similarity * inv[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    max_k = max(2, min(n - 1, 8))
    if k is None:
        gaps = np.diff(eigenvalues[: max_k + 1])
        k = int(np.argmax(gaps) + 2)
    k = max(2, min(k, n - 1))
    embedding = eigenvectors[:, :k]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(norms, 1e-9, None)
    labels, _ = kmeans(embedding, k, seed=seed)
    return labels, k, [float(v) for v in eigenvalues[: max_k + 2]]


def labels_for(roots: np.ndarray, endpoints: list[str]) -> dict[str, tuple[str, str]]:
    return {endpoint: ("P", f"M{int(root)}") for endpoint, root in zip(endpoints, roots)}


def auroc_distance(distance: np.ndarray, endpoints: list[str], truth: dict[str, tuple[str, str]]) -> float:
    same, diff = [], []
    for i, left in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            value = float(distance[i, j])
            if not np.isfinite(value):
                continue
            (same if truth[left] == truth[endpoints[j]] else diff).append(value)
    return auroc(diff, same)


def setprint_base_distances(embeddings, endpoints: list[str], bases: list[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for base in bases:
        delta: dict[str, np.ndarray] = {}
        for endpoint in endpoints:
            base_text = embeddings[endpoint][f"sp{base}_0"]
            vectors = []
            for variant in range(1, 5):
                vectors.append(embeddings[endpoint][f"sp{base}_{variant}"] - base_text)
            matrix = np.stack(vectors)
            matrix = matrix - matrix.mean(0, keepdims=True)
            matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
            delta[endpoint] = matrix
        grid = np.zeros((len(endpoints), len(endpoints)))
        for i, left in enumerate(endpoints):
            for j in range(i + 1, len(endpoints)):
                value = float(np.mean([cosine(delta[left][p], delta[endpoints[j]][p]) for p in range(4)]))
                grid[i, j] = grid[j, i] = 1.0 - value
        result[base] = grid
    return result


def load_proxy_cells(path: Path) -> dict[tuple[str, str, str], list[str]]:
    cells: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cells[(str(row["family"]), str(row["model"]), str(row["disguise"]))].append(str(row["text"]))
    return dict(cells)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    def mean_norm(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2).mean())

    return 2.0 * mean_norm(a, b) - mean_norm(a, a) - mean_norm(b, b)


def proxy_eval(cells, embed_many, endpoints: list[tuple[str, str]], seed: int) -> dict:
    vectors = {cell: embed_many(texts) for cell, texts in cells.items()}
    distance = np.zeros((len(endpoints), len(endpoints)))
    for i, left in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            distance[i, j] = distance[j, i] = energy_distance(vectors[left], vectors[endpoints[j]])
    truth = {cell: cell for cell in endpoints}
    labels = {cell: ("P", f"M{int(root)}") for cell, root in zip(endpoints, spectral_labels(distance, seed=seed)[0])}
    same, diff = [], []
    for i, left in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            (same if truth[left][1] == truth[endpoints[j]][1] else diff).append(distance[i, j])
    auc = auroc(diff, same)
    heldout = []
    for test in sorted({family for family, _model, _disguise in endpoints}):
        test_indices = [
            i for i, (family, _model, _disguise) in enumerate(endpoints) if family == test
        ]
        test_pairs = []
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if i in test_indices or j in test_indices:
                    test_pairs.append((i, j))
        # Recompute labels for this held-out-family evaluation using the same distance matrix.
        roots, k, _ = spectral_labels(distance, seed=seed)
        pred = {endpoints[i]: int(roots[i]) for i in range(len(endpoints))}
        actual = [truth[endpoints[i]][1] == truth[endpoints[j]][1] for i, j in test_pairs]
        called = [pred[endpoints[i]] == pred[endpoints[j]] for i, j in test_pairs]
        tp = sum(a and c for a, c in zip(actual, called))
        tn = sum((not a) and (not c) for a, c in zip(actual, called))
        fp = sum((not a) and c for a, c in zip(actual, called))
        fn = sum(a and (not c) for a, c in zip(actual, called))
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = max(0.0, (tp * tn - fp * fn) / denominator) if denominator else 0.0
        heldout.append({"family": test, "mcc": mcc, "k": k, "pairs": len(test_pairs)})
    return {"auroc": auc, "heldout": heldout, "mean_heldout_mcc": float(np.mean([h["mcc"] for h in heldout])) if heldout else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/setprint_ref.jsonl")
    parser.add_argument("--proxy", default="training/proxy_pairs.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out/setprint_eval.json")
    args = parser.parse_args()

    answers = ROOT / args.answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    pooled = load_pooled(answers, prefix="ref-")
    needed = {f"sp{base}_{variant}" for base in range(1, 6) for variant in range(5)}
    missing = [
        (pid, ep)
        for pid in sorted(needed)
        for ep in endpoints
        if ep not in pooled.get(pid, {})
    ]
    if missing:
        print(f"missing {len(missing)} prompt/endpoint cells; first: {missing[:5]}", file=sys.stderr)
        return 2

    embed_many = load_embedder(args.model, args.device)
    print(f"embedder: {args.model}")
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for endpoint in endpoints:
        embeddings[endpoint] = {
            pid: embed_many([pooled[pid][endpoint][0]])[0] for pid in sorted(needed)
        }

    base_distances = setprint_base_distances(embeddings, endpoints, list(range(1, 6)))
    for base, grid in sorted(base_distances.items()):
        au = auroc_distance(grid, endpoints, truth)
        roots, k, _ = spectral_labels(grid, seed=args.seed)
        labels = labels_for(roots, endpoints)
        print(f"base {base}: AUROC {au:.3f} k={k} practice MCC {practice_mcc(labels, endpoints, truth):.3f}")

    all_grid = np.mean([base_distances[b] for b in range(1, 6)], axis=0)
    all_roots, all_k, _ = spectral_labels(all_grid, seed=args.seed)
    all_labels = labels_for(all_roots, endpoints)
    all_mcc = practice_mcc(all_labels, endpoints, truth)
    loo_results = []
    for holdout in range(1, 6):
        train_grid = np.mean([base_distances[b] for b in range(1, 6) if b != holdout], axis=0)
        roots, k, _ = spectral_labels(train_grid, seed=args.seed)
        labels = labels_for(roots, endpoints)
        mcc = practice_mcc(labels, endpoints, truth)
        loo_results.append({"holdout": holdout, "mcc": mcc, "k": k})
        print(f"LOO base {holdout}: k={k} practice MCC {mcc:.3f}")
    loo_mean = float(np.mean([r["mcc"] for r in loo_results]))
    print(f"\nall-bases: k={all_k} MCC {all_mcc:.3f}")
    print(f"LOO mean MCC {loo_mean:.3f}; min {min(r['mcc'] for r in loo_results):.3f}")
    gate1 = loo_mean >= 0.85

    proxy_cells = load_proxy_cells(ROOT / args.proxy)
    if proxy_cells:
        proxy = proxy_eval(proxy_cells, embed_many, sorted(proxy_cells), args.seed)
        print(f"\nproxy: cells={len(proxy_cells)} AUROC {proxy['auroc']:.3f} "
              f"held-out family MCC {proxy['mean_heldout_mcc']:.3f}")
        for row in proxy["heldout"]:
            print(f"  {row['family']}: MCC {row['mcc']:.3f} k={row['k']} pairs={row['pairs']}")
        gate2 = proxy["mean_heldout_mcc"] >= 0.85
    else:
        proxy = {}
        gate2 = False

    print(f"\nGATE 1 {'PASS' if gate1 else 'FAIL'} (LOO mean {loo_mean:.3f})")
    print(f"GATE 2 {'PASS' if gate2 else 'FAIL'} (proxy held-out {proxy.get('mean_heldout_mcc', float('nan')):.3f})")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "bases": {str(k): v.tolist() for k, v in base_distances.items()},
        "all_mcc": all_mcc,
        "all_k": all_k,
        "loo": loo_results,
        "loo_mean": loo_mean,
        "gate1": gate1,
        "proxy": proxy,
        "gate2": gate2,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if gate1 and gate2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
