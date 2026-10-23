"""Convert named IsaacLab FK references to the SuperMimic MOS9 schema."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def convert(source, output):
    with np.load(source, allow_pickle=False) as d:
        fps = float(d["fps"].item())
        if not np.isclose(fps, 50):
            raise ValueError("resample before converting; expected 50 Hz")
        if "root_pos" in d and "root_quat" in d:
            data = dict(d)
        else:
            names = d["body_names"].tolist()
            idx = names.index("base_link")
            data = {
                "robot": "MOS9",
                "fps": fps,
                "joint_names": d["joint_names"],
                "joint_pos": d["joint_pos"],
                "joint_vel": d["joint_vel"],
                "root_pos": d["body_pos_w"][:, idx],
                "root_quat": d["body_quat_w"][:, idx],
                "root_vel": d["body_lin_vel_w"][:, idx],
                "root_ang_vel": d["body_ang_vel_w"][:, idx],
            }
        data["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        data["source_file"] = str(source)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    convert(a.input, a.output)
    print(json.dumps({"source": str(a.input), "output": str(a.output)}))


if __name__ == "__main__":
    main()
