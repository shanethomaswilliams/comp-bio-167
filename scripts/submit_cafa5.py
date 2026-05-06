#!/usr/bin/env python3
"""
Automated CAFA-5 Kaggle Submission Script
Uploads a TSV as a dataset, waits, then pushes a notebook linked to the competition.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COMPETITION_SLUG = "cafa-5-protein-function-prediction"


def slugify(name):
    s = re.sub(r"[_\s]+", "-", name.lower().strip())
    s = re.sub(r"[^a-z0-9\-]", "", s)
    return re.sub(r"-+", "-", s).strip("-")


def titleify(name):
    return name.replace("_", " ").replace("-", " ").title()


def get_kaggle_username(override=None):
    if override:
        return override
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.exists():
        u = json.loads(p.read_text()).get("username")
        if u:
            return u
    sys.exit("ERROR: Could not determine Kaggle username. Pass --username.")


def run(cmd):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(f"  STDERR: {r.stderr.strip()}", file=sys.stderr)
    return r


def dataset_exists(slug):
    name = slug.split("/")[1]
    r = run(f'kaggle datasets list --mine -s "{name}"')
    return r.returncode == 0 and name in r.stdout.lower()


def upload_dataset(tsv_path, username, name_slug, name_title, description):
    slug = f"{username}/cafa5-{name_slug}"

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(tsv_path, Path(tmp) / "submission.tsv")
        meta = {"id": slug, "title": f"CAFA5 {name_title}", "licenses": [{"name": "CC0-1.0"}]}
        (Path(tmp) / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        if dataset_exists(slug):
            print(f"\n→ Versioning existing dataset: {slug}")
            r = run(f'kaggle datasets version -p "{tmp}" -m "{description}" --dir-mode zip')
        else:
            print(f"\n→ Creating new dataset: {slug}")
            r = run(f'kaggle datasets create -p "{tmp}"')

        if r.returncode != 0:
            print("  ⚠ Upload command returned an error, but the file may have landed.")
            print(f"  Check: https://www.kaggle.com/datasets/{slug}")
            print("  Continuing — notebook push will retry if dataset isn't ready.")

    print(f"  ✓ Dataset uploaded: {slug}")
    return slug


def _try_push_notebook(username, dataset_slug, name_slug, name_title, description):
    """Attempt a single notebook push. Returns True on success, False if dataset not ready yet."""
    raw_kernel_slug = f"cafa5-submit-{name_slug}"
    kernel_slug = f"{username}/{raw_kernel_slug[:50]}"  # Kaggle hard limit: 50 chars

    with tempfile.TemporaryDirectory() as tmp:
        script = f"""\
# CAFA-5 Submission: {name_title}
# {description}
import shutil, os

dataset_slug = "{dataset_slug}"
dataset_folder = "{dataset_slug.split('/')[1]}"

candidates = [
    f"/kaggle/input/{{dataset_folder}}/submission.tsv",
    f"/kaggle/input/{{dataset_slug}}/submission.tsv",
]

src = None
for c in candidates:
    if os.path.exists(c):
        src = c
        break

if not src:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "submission.tsv" in files:
            src = os.path.join(root, "submission.tsv")
            break

if not src:
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.replace("/kaggle/input", "").count(os.sep)
        if depth < 4:
            print(f"  {{'  ' * depth}}{{os.path.basename(root)}}/  {{files[:5]}}")
    raise FileNotFoundError("submission.tsv not found in /kaggle/input/")

dst = "/kaggle/working/submission.tsv"
shutil.copy2(src, dst)
print(f"Copied {{src}} -> {{dst}}")
print(f"Size: {{os.path.getsize(dst):,}} bytes")
with open(dst) as f:
    lines = f.readlines()
print(f"Rows: {{len(lines):,}}")
for line in lines[:5]:
    print(line.rstrip())
"""
        (Path(tmp) / "script.py").write_text(script)

        meta = {
            "id": kernel_slug,
            "title": f"CAFA5 Submit: {name_title}",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [dataset_slug],
            "competition_sources": [COMPETITION_SLUG],
            "kernel_data_sources": [],
            "keywords": [],
        }
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        r = run(f'kaggle kernels push -p "{tmp}"')

        output = (r.stdout + r.stderr).lower()

        if r.returncode != 0 or "not valid dataset sources" in output:
            return False  # Dataset not ready yet, caller should retry

    print(f"  ✓ Notebook pushed: {kernel_slug}")
    print(
        f"\n{'='*60}\n"
        f"  {name_title}\n"
        f"  \"{description}\"\n"
        f"{'='*60}\n"
        f"  1. Wait ~1-2 min, then check:\n"
        f"     kaggle kernels status {kernel_slug}\n"
        f"  2. Submit at:\n"
        f"     https://www.kaggle.com/code/{kernel_slug}\n"
        f"     ⋮ menu → 'Submit to Competition'\n"
        f"{'='*60}"
    )
    return True


def push_notebook_with_retry(username, dataset_slug, name_slug, name_title, description,
                              retries=20, interval=30):
    print(f"\n→ Pushing notebook (will retry up to {retries}x every {interval}s)...")
    for attempt in range(1, retries + 1):
        print(f"\n  Attempt {attempt}/{retries}")
        if _try_push_notebook(username, dataset_slug, name_slug, name_title, description):
            return
        if attempt < retries:
            print(f"  Dataset not ready yet, waiting {interval}s...")
            time.sleep(interval)

    sys.exit("ERROR: Notebook push failed after all retries. Check dataset at "
             f"https://www.kaggle.com/datasets/{dataset_slug}")


def main():
    parser = argparse.ArgumentParser(description="Submit TSV to CAFA-5 on Kaggle")
    parser.add_argument("tsv", type=Path, help="Path to submission TSV")
    parser.add_argument("--name", "-n", required=True, help="Short name (e.g. protgoat_test)")
    parser.add_argument("--description", "-d", default="Auto submission", help="Notes")
    parser.add_argument("--username", "-u", default=None, help="Kaggle username")
    parser.add_argument("--retries", type=int, default=20,
                        help="Max push attempts while waiting for dataset (default: 20)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between push retries (default: 30)")
    args = parser.parse_args()

    if not args.tsv.exists():
        sys.exit(f"ERROR: File not found: {args.tsv}")

    username = get_kaggle_username(args.username)
    name_slug = slugify(args.name)
    name_title = titleify(args.name)
    file_mb = args.tsv.stat().st_size / (1024 * 1024)

    print(f"Kaggle user:  {username}")
    print(f"Submission:   {name_title} ({name_slug})")
    print(f"TSV file:     {args.tsv}  ({file_mb:.0f}MB)")
    print(f"Description:  {args.description}")

    dataset_slug = upload_dataset(args.tsv, username, name_slug, name_title, args.description)

    push_notebook_with_retry(
        username, dataset_slug, name_slug, name_title, args.description,
        retries=args.retries,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()