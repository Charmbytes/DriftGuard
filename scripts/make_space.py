"""
Assemble (and optionally upload) the Hugging Face Space for the live demo.

A Space wants its own README.md (with Space settings at the top) and a file
named exactly `Dockerfile` at its root. Neither belongs at the root of this
repository, so this script copies the app plus the files in deploy/huggingface/
into a separate folder, `space/`, which is what gets uploaded.

    python scripts/make_space.py                       # build space/ only
    python scripts/make_space.py --push USER/SPACE     # build and upload

Uploading needs a one-time login with a Hugging Face *write* token:

    hf auth login

`space/` is gitignored -- it is build output, regenerated on every run.
"""

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPACE_FILES = ROOT / "deploy" / "huggingface"
OUT = ROOT / "space"

# What the container needs to run, and nothing else (no tests, no data).
COPY_DIRS = ["driftguard", "dashboard"]
COPY_FILES = ["requirements.txt", "LICENSE"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "data", "*.db*", ".jwt_secret")


def build() -> pathlib.Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for d in COPY_DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=IGNORE)
    for f in COPY_FILES:
        shutil.copy2(ROOT / f, OUT / f)

    for f in SPACE_FILES.iterdir():
        # Write with LF endings: a Windows CRLF in start.sh would stop bash
        # from running it inside the Linux container.
        text = f.read_text(encoding="utf-8").replace("\r\n", "\n")
        (OUT / f.name).write_text(text, encoding="utf-8", newline="\n")

    print(f"Space assembled in {OUT}")
    return OUT


def push(repo_id: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=str(OUT),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Update DriftGuard demo",
        # Remove files that no longer exist locally, so the Space mirrors space/.
        delete_patterns=["*"],
    )
    print(f"Uploaded. The Space builds in a few minutes: "
          f"https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--push", metavar="USER/SPACE",
                        help="Upload to this Space after building, e.g. yourname/DriftGuard")
    args = parser.parse_args()

    build()
    if args.push:
        push(args.push)


if __name__ == "__main__":
    main()
