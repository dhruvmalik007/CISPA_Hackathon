import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from unlearning.task import load_model, load_forget_images, predict_logits

M = Path("/p/scratch/training2666/falcon1/FrenchyFighters/malik4/unlearning")
sub = load_model(M / "output/stage6-match-final/submission.pt", device="cuda")
orig = load_model(M / "model.pt", device="cuda")
images = load_forget_images(M / "forget_images.npz")
orig_logits = predict_logits(orig, images, batch_size=64, device="cuda")
sub_logits = predict_logits(sub, images, batch_size=64, device="cuda")
labels = orig_logits.argmax(dim=-1)

print("alpha exact_mean exact_std")
for alpha in np.arange(0.30, 1.001, 0.05):
  ce = F.cross_entropy(sub_logits * alpha, labels, reduction="none")
  print(f"{alpha:.2f} {ce.mean().item():.4f} {ce.std(unbiased=False).item():.4f}")

