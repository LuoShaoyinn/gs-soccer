"""Fetch public sources at immutable revisions, without executing upstream code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

MOS9_REVISION = "22aa35fd80b65d2064cbc790163e59f9667266bd"
G1_REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"


def read(url):
    with urlopen(Request(url, headers={"User-Agent": "gs-soccer-motion-audit"}), timeout=60) as response:
        return response.read()


def fetch_mos9(output):
    repo = "THMOS2025/MOS9-AMP"
    tree = json.loads(read(f"https://api.github.com/repos/{repo}/git/trees/{MOS9_REVISION}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("incomplete upstream tree")
    selected = []
    for entry in tree["tree"]:
        path = entry["path"]
        data = path.endswith(".npz") and any(path.startswith(f"data/motions/{d}/") for d in
            ("mos9_fk_motion", "mos9_fk_motion_clipped_simple", "mos9_fk_rotate_motion"))
        source = path in ("data/motions/README.md", "scripts/mos9_fk_npz.py", "README.en.md",
                           "source/amp_tasks/amp_tasks/robots/MOS9.py")
        if data or source:
            selected.append(entry)

    def download(entry):
        path = entry["path"]
        raw = read(f"https://raw.githubusercontent.com/{repo}/{MOS9_REVISION}/{path}")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != entry["sha"]:
            raise ValueError(f"Git blob mismatch: {path}")
        target = output/"raw"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {"path": path, "bytes": len(raw), "git_blob_sha1": blob,
                "sha256": hashlib.sha256(raw).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(download, selected))
    manifest = {"repo": repo, "commit": MOS9_REVISION, "files": files}
    (output/"upstream_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def fetch_g1(output):
    repo = "lvhaidong/LAFAN1_Retargeting_Dataset"
    files = []
    for path in ("g1/walk1_subject1.csv", "g1/walk2_subject1.csv", "README.md"):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{G1_REVISION}/{path}"
        raw = read(url)
        target = output/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files.append({"path": path, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    (output/"source_manifest.json").write_text(json.dumps({"repo": repo, "revision": G1_REVISION,
        "source_fps": 30, "quaternion": "xyzw", "files": files}, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=("mos9", "g1"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    (fetch_mos9 if a.source == "mos9" else fetch_g1)(a.output)


if __name__ == "__main__":
    main()
