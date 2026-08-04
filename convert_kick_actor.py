"""Convert an exported kick ONNX actor into a ROCm-native Torch checkpoint."""

import argparse
from pathlib import Path

from algorithm.kick_teacher_torch import convert_onnx_kick_actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.onnx.with_suffix(".pt")
    print(convert_onnx_kick_actor(args.onnx, output))


if __name__ == "__main__":
    main()
