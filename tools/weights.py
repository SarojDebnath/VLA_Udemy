"""Pretrained course weights: package them (course author) or download them (students).

Students:
    python tools/weights.py get                 # all sections (~750 MB)
    python tools/weights.py get S08 S17         # only some sections
    python tools/weights.py list                # what exists, sizes, what you already have

Course author:
    python tools/weights.py pack                # -> dist/weights/<section>.zip + manifest.json (sha256, sizes)
    huggingface-cli upload <user>/vla-course-weights dist/weights . --repo-type model

Every notebook that uses weights also trains them itself when they are missing (see TRAINING.md), so downloading is
optional: it saves the training time, nothing else.
Source repo: the environment variable VLA_WEIGHTS_REPO, else DEFAULT_REPO below.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

DEFAULT_REPO = "SarojDebnath/vla-course-weights"       # Hugging Face model repo holding <section>.zip + manifest.json
ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "weights"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sections_with_weights():
    return sorted(p.parent.name for p in ROOT.glob("S*/checkpoints") if any(f.is_file() for f in p.rglob("*")))


def pack():
    DIST.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for sec in sections_with_weights():
        z = DIST / f"{sec}.zip"
        files = [f for f in (ROOT / sec / "checkpoints").rglob("*") if f.is_file()]
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:      # weights don't compress; STORED is fast
            for f in files:
                zf.write(f, f.relative_to(ROOT).as_posix())
        manifest[sec] = dict(file=z.name, size=z.stat().st_size, sha256=sha256(z), n_files=len(files))
        print(f"{sec:40s} {len(files):3d} files  {z.stat().st_size / 2**20:7.1f} MB")
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(m["size"] for m in manifest.values())
    print(f"\n{len(manifest)} sections, {total / 2**20:.0f} MB -> {DIST}")
    print(f"upload:  huggingface-cli upload <user>/vla-course-weights {DIST} . --repo-type model")


def _url(repo, name):
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


def _download(url, dest):
    import requests
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 2**20:6.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()


def manifest(repo):
    import requests
    r = requests.get(_url(repo, "manifest.json"), timeout=60)
    if r.status_code != 200:
        sys.exit(f"Could not read {_url(repo, 'manifest.json')} (HTTP {r.status_code}). Check VLA_WEIGHTS_REPO, or train the "
                 f"models yourself: every notebook trains its own weights when they are missing (see TRAINING.md).")
    return r.json()


def get(sections=(), repo=None):
    repo = repo or os.environ.get("VLA_WEIGHTS_REPO", DEFAULT_REPO)
    man = manifest(repo)
    want = [s for s in man if not sections or any(s.startswith(x) for x in sections)]
    tmp = ROOT / "dist" / "download"
    tmp.mkdir(parents=True, exist_ok=True)
    for sec in want:
        m = man[sec]
        z = tmp / m["file"]
        if not (z.exists() and z.stat().st_size == m["size"] and sha256(z) == m["sha256"]):
            _download(_url(repo, m["file"]), z)
            if sha256(z) != m["sha256"]:
                sys.exit(f"checksum mismatch for {z.name}: delete it and try again")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ROOT)
        print(f"{sec}: {m['n_files']} files installed")
    print("done: notebooks will now load these weights instead of training.")


def list_(repo=None):
    repo = repo or os.environ.get("VLA_WEIGHTS_REPO", DEFAULT_REPO)
    man = manifest(repo)
    have = set(sections_with_weights())
    for sec, m in man.items():
        print(f"{sec:40s} {m['size'] / 2**20:7.1f} MB  {'installed' if sec in have else '-'}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "help"), sys.argv[2:]
    if cmd == "pack":
        pack()
    elif cmd == "get":
        get(args)
    elif cmd == "list":
        list_()
    else:
        print(__doc__)
