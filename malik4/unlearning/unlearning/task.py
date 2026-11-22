"""Model, data, and submission primitives matching the official task template.

Everything here mirrors `task_template.py` from the competition dataset because the
scoring server uses exactly that preprocessing, architecture, and state-dict contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

NUM_CLASSES = 100
IMAGE_SIZE = 224
EXPECTED_FORGET_IMAGES = 3500
MAX_SUBMISSION_BYTES = 450 * 1024 * 1024

MODEL_FILENAME = "model.pt"
FORGET_FILENAME = "forget_images.npz"
SUBMISSION_FILENAME = "submission.pt"


def build_model() -> torch.nn.Module:
    """ViT-B/16 at 224 with LayerScale init 1e-4, exactly as the organizers built it."""
    import timm

    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
        init_values=1e-4,
    )


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a plain state dict (got {type(state).__name__})")
    return state


def load_model(path: str | Path = MODEL_FILENAME, device: str | torch.device = "cpu") -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(load_state_dict(path), strict=True)
    return model.to(device).eval()


def load_forget_images(path: str | Path = FORGET_FILENAME) -> torch.Tensor:
    """uint8 [N, 224, 224, 3], channels last, raw pixels in 0-255."""
    with np.load(path, allow_pickle=False) as data:
        if "images" not in data.files:
            raise KeyError(f"{path} has no 'images' array; available: {data.files}")
        images = torch.from_numpy(data["images"].copy())

    if images.dtype != torch.uint8:
        raise ValueError(f"expected uint8 pixels, got {images.dtype}")
    if images.ndim != 4:
        raise ValueError(f"expected [N, H, W, 3], got {tuple(images.shape)}")
    if images.shape[-1] != 3 and images.shape[1] == 3:
        images = images.permute(0, 2, 3, 1).contiguous()
    if tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"expected [N, {IMAGE_SIZE}, {IMAGE_SIZE}, 3], got {tuple(images.shape)}")
    return images


def preprocess(images_uint8: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
    """uint8 [B, H, W, 3] -> float32 [B, 3, H, W] in [-1, 1]; the server's exact transform."""
    x = images_uint8.to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a parameterless model is not a ViT
        return torch.device("cpu")


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    images_uint8: torch.Tensor,
    batch_size: int = 128,
    device: str | torch.device | None = None,
    autocast: bool = True,
) -> torch.Tensor:
    """Logits [N, 100] on CPU, computed like the scoring server (eval, fp16 autocast on GPU)."""
    target = torch.device(device) if device is not None else model_device(model)
    model.eval()
    chunks = []
    for start in range(0, len(images_uint8), batch_size):
        batch = images_uint8[start : start + batch_size]
        with torch.autocast(
            device_type=target.type,
            dtype=torch.float16,
            enabled=autocast and target.type == "cuda",
        ):
            chunks.append(model(preprocess(batch, target)).float().cpu())
    if not chunks:
        return torch.empty((0, NUM_CLASSES), dtype=torch.float32)
    return torch.cat(chunks)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_state_dict(state: dict[str, torch.Tensor], expected: dict[str, torch.Size] | None = None) -> dict[str, Any]:
    """Validate dtype, finiteness, keys, and shapes without needing timm."""
    arrays = {name: tensor for name, tensor in state.items() if isinstance(tensor, torch.Tensor)}
    non_tensors = sorted(set(state) - set(arrays))
    dtypes = sorted({str(tensor.dtype) for tensor in arrays.values()})
    float_tensors = [tensor for tensor in arrays.values() if tensor.is_floating_point()]
    non_float32 = sorted({str(t.dtype) for t in float_tensors if t.dtype != torch.float32})
    non_finite = [name for name, tensor in arrays.items() if tensor.is_floating_point() and not torch.isfinite(tensor).all()]
    bytes_total = sum(tensor.numel() * tensor.element_size() for tensor in arrays.values())

    missing: list[str] = []
    unexpected: list[str] = []
    mismatched: list[str] = []
    if expected is not None:
        missing = sorted(set(expected) - set(arrays))
        unexpected = sorted(set(arrays) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(arrays) if tuple(arrays[name].shape) != tuple(expected[name])
        )

    return {
        "tensors": len(arrays),
        "non_tensors": non_tensors,
        "dtypes": dtypes,
        "non_float32": non_float32,
        "non_finite": non_finite,
        "bytes": bytes_total,
        "megabytes": round(bytes_total / 1024 / 1024, 1),
        "within_size_limit": bytes_total <= MAX_SUBMISSION_BYTES,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": mismatched,
    }


def validate_submission(
    path: str | Path,
    reference: dict[str, torch.Tensor] | None = None,
    check_architecture: bool = True,
) -> dict[str, Any]:
    """Full server-side contract check, plus an optional identity distance to the baseline."""
    path = Path(path)
    state = load_state_dict(path)
    expected = None
    if check_architecture:
        try:
            expected = {name: tuple(tensor.shape) for name, tensor in build_model().state_dict().items()}
        except ImportError:
            expected = None

    report = check_state_dict(state, expected)
    report["path"] = str(path)
    report["file_bytes"] = path.stat().st_size
    report["strict_load"] = not (report["missing_keys"] or report["unexpected_keys"] or report["shape_mismatches"])
    report["ok"] = (
        report["strict_load"]
        and not report["non_float32"]
        and not report["non_finite"]
        and report["within_size_limit"]
        and not report["non_tensors"]
    )

    if reference is not None:
        shared = sorted(set(state) & set(reference))
        max_abs = 0.0
        for name in shared:
            max_abs = max(max_abs, float((state[name].float() - reference[name].float()).abs().max()))
        report["max_abs_diff_vs_reference"] = max_abs
        report["identical_to_reference"] = max_abs == 0.0
    return report


def save_submission(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    """Save float32 CPU weights and validate them before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().float() for name, tensor in model.state_dict().items()}
    torch.save(state, path)
    return validate_submission(path)


def baseline_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().float().clone() for name, tensor in model.state_dict().items()}
