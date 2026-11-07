"""Downloads and summarizes small files from the published SGLang-Omni benchmark runs.

Only small metadata files are fetched. The revision is pinned, so results stay
reproducible even if the dataset changes. Dataset license: CC BY-NC 4.0.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


def download(dataset: str, revision: str, files: dict[str, str], out_dir: str | Path) -> None:
    """Fetch each `local name: path in dataset` entry unless it already exists."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_name, remote_path in files.items():
        target = out_dir / local_name
        if target.exists():
            continue
        url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{remote_path}"
        request = urllib.request.Request(url, headers={"User-Agent": "mip-capstone"})
        with urllib.request.urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())


def summarize(raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    capabilities = json.loads((raw_dir / "cascade_capabilities.json").read_text())
    replay = json.loads((raw_dir / "empty_reply_replay.json").read_text())
    return {
        "cascade_capability_fields": len(capabilities),
        "cascade_capability_names": sorted(capabilities),
        "empty_replies_by_history_form": {
            form: {"empty": result["empty"], "samples": result["samples"]}
            for form, result in replay.items()
        },
    }


def write_summary(summary: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path
