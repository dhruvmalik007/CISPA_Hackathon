"""Deterministic altered-copy suite used only for local evaluation.

The scoring server keeps hidden altered copies of every forget image. We cannot see
them, so this module builds a fixed, reproducible set of plausible alterations and
uses the original model's predictions on different alterations as a stand-in for the
unseen gold model's predictions. Nothing here is used to train on external data.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from unlearning.task import IMAGE_SIZE

VARIANTS = ("crop", "flip", "color", "blur_jpeg", "occlude", "rotate")
DEFAULT_VARIANTS = ("crop", "flip", "color", "blur_jpeg")


def _resample() -> Image.Resampling:
    return Image.Resampling.BICUBIC


def _variant_rng(seed: int, index: int, variant: str) -> np.random.Generator:
    return np.random.default_rng([seed, index, VARIANTS.index(variant)])


def _augment_one(array: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    image = Image.fromarray(array)
    width, height = image.size

    if variant == "crop":
        scale = float(rng.uniform(0.60, 0.95))
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))
        left = int(rng.integers(0, width - crop_w + 1))
        top = int(rng.integers(0, height - crop_h + 1))
        image = image.crop((left, top, left + crop_w, top + crop_h))
    elif variant == "flip":
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.rotate(float(rng.uniform(-12.0, 12.0)), resample=_resample(), expand=False)
    elif variant == "color":
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.75, 1.25)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.75, 1.25)))
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.60, 1.40)))
    elif variant == "blur_jpeg":
        if rng.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.8, 2.0))))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(rng.integers(45, 80)))
        image = Image.open(buffer).convert("RGB")
    elif variant == "occlude":
        array_copy = np.asarray(image).copy()
        size = int(rng.integers(40, 80))
        top = int(rng.integers(0, IMAGE_SIZE - size + 1))
        left = int(rng.integers(0, IMAGE_SIZE - size + 1))
        array_copy[top : top + size, left : left + size] = 0
        image = Image.fromarray(array_copy)
    elif variant == "rotate":
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.rotate(float(rng.uniform(-15.0, 15.0)), resample=_resample(), expand=False)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown variant: {variant}")

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), _resample())
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def build_altered(
    images: torch.Tensor,
    variants: tuple[str, ...] | list[str] = DEFAULT_VARIANTS,
    seed: int = 0,
    progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `len(variants)` altered copies per image.

    Returns ``(altered, sources)`` where ``altered`` is uint8
    ``[N * len(variants), 224, 224, 3]`` and ``sources[i]`` is the index of the
    original image that produced row ``i``.
    """
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {VARIANTS}")
    if images.dtype != torch.uint8 or images.ndim != 4:
        raise ValueError("images must be uint8 [N, H, W, 3]")

    pixels = images.cpu().numpy()
    count = len(pixels)
    outputs: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    for index in range(count):
        for variant in variants:
            outputs.append(_augment_one(pixels[index], variant, _variant_rng(seed, index, variant)))
            sources.append(np.int64(index))
        if progress and (index + 1) % 500 == 0:
            print(f"altered copies: {index + 1}/{count}", flush=True)

    altered = torch.from_numpy(np.stack(outputs, axis=0))
    return altered, torch.from_numpy(np.stack(sources, axis=0))


def build_altered_by_variant(
    images: torch.Tensor,
    variants: tuple[str, ...] | list[str] = DEFAULT_VARIANTS,
    seed: int = 0,
    progress: bool = False,
) -> list[torch.Tensor]:
    """Same alterations, grouped so ``result[k]`` has shape ``[N, 224, 224, 3]``."""
    altered, sources = build_altered(images, variants, seed, progress)
    grouped: list[torch.Tensor] = []
    positions = torch.arange(len(sources)) % len(variants)
    for offset in range(len(variants)):
        grouped.append(altered[positions == offset])
    return grouped


def build_target_copy(images: torch.Tensor, seed: int = 0, progress: bool = False) -> torch.Tensor:
    """A deliberately strong alteration used only as the "unseen image" target.

    Mild alterations are still partly recognised by a model that memorised the clean
    image. A heavy crop + blur + JPEG removes enough evidence that the original
    model's prediction is a better stand-in for what a retrained gold model would
    output on the clean image.
    """
    pixels = images.cpu().numpy()
    outputs: list[np.ndarray] = []
    for index in range(len(pixels)):
        rng = np.random.default_rng([seed, index, 10_007])
        image = Image.fromarray(pixels[index])
        width, height = image.size

        scale = float(rng.uniform(0.35, 0.60))
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))
        left = int(rng.integers(0, width - crop_w + 1))
        top = int(rng.integers(0, height - crop_h + 1))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), _resample())
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(1.2, 2.5))))
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.7, 1.3)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.7, 1.3)))

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(rng.integers(30, 60)))
        image = Image.open(buffer).convert("RGB")
        outputs.append(np.asarray(image, dtype=np.uint8))
        if progress and (index + 1) % 500 == 0:
            print(f"target copies: {index + 1}/{len(pixels)}", flush=True)

    return torch.from_numpy(np.stack(outputs, axis=0))
