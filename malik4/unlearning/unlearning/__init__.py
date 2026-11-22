"""Unlearning task 3: ViT-B/16, 100 classes, SCRUB-style forgetting.

The same package runs inside Modal and on JURECA. The official
`task_template.py` / `submission.py` contract is reproduced in `task.py`;
nothing here changes the architecture or the preprocessing.
"""

from unlearning.task import (
    EXPECTED_FORGET_IMAGES,
    FORGET_FILENAME,
    IMAGE_SIZE,
    MODEL_FILENAME,
    NUM_CLASSES,
    build_model,
    load_forget_images,
    load_model,
    predict_logits,
    preprocess,
    save_submission,
    validate_submission,
)

__all__ = [
    "EXPECTED_FORGET_IMAGES",
    "FORGET_FILENAME",
    "IMAGE_SIZE",
    "MODEL_FILENAME",
    "NUM_CLASSES",
    "build_model",
    "load_forget_images",
    "load_model",
    "predict_logits",
    "preprocess",
    "save_submission",
    "validate_submission",
]
