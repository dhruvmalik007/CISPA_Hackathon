"""Optional in-distribution probe set (ImageNet-100 validation) for retain checks.

This set is used for **evaluation only**. It is never passed to the unlearning
objective; the challenge says the method must work from the model and the forget
set alone. A high mapped accuracy means the original model was trained on this
distribution, which makes the set a usable proxy for the hidden control set.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

AUX_REPO = "clane9/imagenet-100"
AUX_FILE = "data/validation-00000-of-00001.parquet"
IMAGE_SIZE = 224


def download_aux_npz(target: str | Path, limit: int = 5000, token: str | None = None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageOps
    import pyarrow.parquet as pq

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = hf_hub_download(repo_id=AUX_REPO, repo_type="dataset", filename=AUX_FILE, token=token)
    table = pq.read_table(source)
    images_column = table.column("image").to_pylist()
    labels_column = table.column("label").to_pylist()
    count = min(limit, len(images_column)) if limit else len(images_column)

    arrays = np.empty((count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty((count,), dtype=np.int64)
    written = 0
    for index in range(count):
        raw = images_column[index]
        if isinstance(raw, dict):
            raw = raw.get("bytes") or raw.get("path")
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = raw.encode("latin-1")
        try:
            picture = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        picture = ImageOps.fit(picture, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BICUBIC)
        arrays[written] = np.asarray(picture, dtype=np.uint8)
        labels[written] = int(labels_column[index])
        written += 1

    arrays = arrays[:written]
    labels = labels[:written]
    np.savez(target, images=arrays, labels=labels)
    return {"source": AUX_REPO, "rows": int(written), "path": str(target)}


def probe_class_map(
    aux_npz: str | Path,
    model_path: str | Path,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Map ImageNet-100 labels to the custom classifier and measure mapped accuracy."""
    from scipy.optimize import linear_sum_assignment

    from unlearning.task import load_model, predict_logits

    with np.load(aux_npz, allow_pickle=False) as data:
        images = torch.from_numpy(data["images"].copy())
        labels = torch.from_numpy(data["labels"].copy())

    model = load_model(model_path, device=device)
    logits = predict_logits(model, images, batch_size=batch_size, device=device)
    predictions = logits.argmax(dim=-1)
    counts = torch.zeros((100, 100), dtype=torch.float64)
    for aux_label, prediction in zip(labels.tolist(), predictions.tolist()):
        counts[aux_label, prediction] += 1

    rows, columns = linear_sum_assignment(-counts.numpy())
    mapping = {int(row): int(column) for row, column in zip(rows, columns)}
    class_map = [mapping[index] for index in range(100)]
    correct = sum(int(counts[index, class_map[index]]) for index in range(100))
    accuracy = correct / max(1, len(labels))
    totals = counts.sum(dim=1).numpy().clip(1)
    per_class = (counts[torch.arange(100), torch.tensor(class_map)].numpy() / totals).tolist()

    return {
        "aux_repo": AUX_REPO,
        "rows": int(len(labels)),
        "mapped_accuracy": float(accuracy),
        "in_distribution": bool(accuracy >= 0.5),
        "per_class_accuracy": per_class,
        "class_map": class_map,
        "note": "Evaluation-only. A high mapped accuracy means this set is a usable retain proxy.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download ImageNet-100 validation into an npz")
    prepare.add_argument("--output", default="aux/aux_images.npz")
    prepare.add_argument("--limit", type=int, default=5000)

    probe = subparsers.add_parser("probe", help="map classes and measure mapped accuracy")
    probe.add_argument("--aux", default="aux/aux_images.npz")
    probe.add_argument("--model", default="model.pt")
    probe.add_argument("--output-dir", default="aux")
    probe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(download_aux_npz(args.output, args.limit), indent=2))
        return 0

    report = probe_class_map(args.aux, args.model, device=args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe.json").write_text(json.dumps(report, indent=2))
    (output_dir / "class_map.json").write_text(json.dumps({"class_map": report["class_map"]}, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "per_class_accuracy"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
