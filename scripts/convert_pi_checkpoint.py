import argparse
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Convert legacy PI checkpoint into tensor-only actor state dict.")
    parser.add_argument("--src", type=Path, required=True, help="Path to legacy checkpoint (.pt)")
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output path (default: overwrite src and write *_legacy backup)",
    )
    args = parser.parse_args()

    src = args.src
    dst = args.dst or src
    backup = src.with_name(f"{src.stem}_legacy{src.suffix}")

    try:
        import rsl_rl.utils.utils as rsl_utils

        class Normalizer:
            pass

        rsl_utils.Normalizer = Normalizer
    except Exception as e:
        raise RuntimeError("rsl_rl is required for one-time legacy checkpoint loading") from e

    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported legacy checkpoint format: {src}")

    model_state = checkpoint["model_state_dict"]
    safe = {
        "format": "pi_actor_state_v1",
        "dims": [345, 512, 256, 128, 20],
        "activation": "ELU",
        "state_dict": {
            "0.weight": model_state["actor.0.weight"].detach().cpu(),
            "0.bias": model_state["actor.0.bias"].detach().cpu(),
            "2.weight": model_state["actor.2.weight"].detach().cpu(),
            "2.bias": model_state["actor.2.bias"].detach().cpu(),
            "4.weight": model_state["actor.4.weight"].detach().cpu(),
            "4.bias": model_state["actor.4.bias"].detach().cpu(),
            "6.weight": model_state["actor.6.weight"].detach().cpu(),
            "6.bias": model_state["actor.6.bias"].detach().cpu(),
        },
    }

    if dst == src:
        shutil.copy2(src, backup)
        print(f"Backup written: {backup}")

    torch.save(safe, dst)
    print(f"Secure checkpoint written: {dst}")


if __name__ == "__main__":
    main()
