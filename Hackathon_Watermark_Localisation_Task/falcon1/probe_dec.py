from pathlib import Path
import torch
import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import to_tensor
dev = "cuda" if torch.cuda.is_available() else "cpu"
item = next(it for it in LocalisationDataset("../release", "train") if str(it["method"]).lower() == "trustmark")
wm = W._CLASSES["trustmark"](Path("models").resolve(), dev)
print("object:", type(wm).__name__)
mods = {k: v for k, v in vars(wm).items() if isinstance(v, torch.nn.Module)}
print("modules:", {k: type(v).__name__ for k, v in mods.items()})
for k, v in mods.items():
    print(f"  {k}: {sum(p.numel() for p in v.parameters())/1e6:.1f}M params")
    for n, c in list(v.named_children())[:12]:
        print(f"     {n}: {type(c).__name__}")
seen = []
for top, mod in mods.items():
    for name, sub in mod.named_modules():
        sub.register_forward_hook(lambda m, i, o, n=f"{top}.{name}": seen.append((n, tuple(o.shape))) if torch.is_tensor(o) and o.dim() == 4 else None)
x = to_tensor(item["image"])[None].to(dev)
fn = getattr(type(wm).logits, "__wrapped__", None)
z = fn(wm, x) if fn is not None else wm.logits(x)
print("logits:", tuple(z.shape))
last = {}
for n, s in seen:
    last[n] = s
for n, s in list(last.items())[-30:]:
    print(f"  {n:55s} {s}")
