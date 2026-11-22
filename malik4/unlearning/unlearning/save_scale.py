
import sys
from pathlib import Path
import torch
from unlearning.task import load_model

M = Path("/p/scratch/training2666/falcon1/FrenchyFighters/malik4/unlearning")
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
submission = M / "output/stage6-match-final/submission.pt"

state = torch.load(submission, map_location="cpu", weights_only=True)
scaled = {
  key: (value.detach().float().clone() if torch.is_tensor(value) else value)
  for key, value in state.items()
}
scaled["head.weight"] = scaled["head.weight"] * alpha
scaled["head.bias"] = scaled["head.bias"] * alpha

target = M / "output/autosearch_scale" / f"scale-{alpha:.2f}.pt"
target.parent.mkdir(parents=True, exist_ok=True)
torch.save(scaled, target)
load_model(target, device="cpu")
print("saved", target)
