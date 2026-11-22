import argparse
import os
import torch


def load_sd(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    assert len(args.ckpts) == len(args.weights), "ckpts and weights must match in length"
    total_w = sum(args.weights)
    weights = [w / total_w for w in args.weights]

    merged = None
    for path, w in zip(args.ckpts, weights):
        print(f"Loading {path} (weight={w:.4f})")
        sd = load_sd(path)
        if merged is None:
            merged = {}
            for k, v in sd.items():
                merged[k] = v.float() * w if v.is_floating_point() else v.clone()
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    merged[k] += v.float() * w

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save(merged, tmp)
    os.replace(tmp, args.out)
    print(f"Saved merged checkpoint to {args.out}")


if __name__ == "__main__":
    main()