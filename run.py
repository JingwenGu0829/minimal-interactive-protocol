#!/usr/bin/env python
"""Build script: `python run.py <target> [<target> ...]`.

Targets
  test      Unit tests, then ablation and baseline on unit test data. Offline.
  ablation  Remove each protocol primitive in simulation and measure what breaks.
  data      Download small metadata files from the published benchmark runs.
  baseline  Summarize the downloaded files.
  live      Run config/live.json against OpenAI-compatible endpoints.
  probe     Ask a real chat model what it last said, with primitives removed.
  all       data, baseline, ablation.
  clean     Delete everything this script builds.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mip import baseline  # noqa: E402
from mip.figures import write_timeline  # noqa: E402
from mip.simulate import SimConfig, format_rows, run_ablation, write_rows  # noqa: E402


def load_config(name: str) -> dict:
    return json.loads((ROOT / "config" / f"{name}.json").read_text())


def build_ablation(config: dict) -> None:
    sim_config = SimConfig(**config["sim"])
    rows = run_ablation(ROOT / config["scenario"], sim_config)
    table = write_rows(rows, ROOT / config["out_dir"])
    figure = write_timeline(ROOT / config["scenario"], sim_config, ROOT / config["out_dir"])
    print(format_rows(rows))
    print(f"Wrote {table.relative_to(ROOT)} and {figure.relative_to(ROOT)}")


def build_data(config: dict) -> None:
    data = config["data"]
    baseline.download(data["dataset"], data["revision"], data["files"], ROOT / data["out_dir"])
    print(f"Downloaded {len(data['files'])} files to {data['out_dir']}")


def build_baseline(config: dict) -> None:
    summary = baseline.summarize(ROOT / config["data"]["out_dir"])
    path = baseline.write_summary(summary, ROOT / config["out_dir"])
    print(f"Wrote {path.relative_to(ROOT)}")


def load_live_config() -> dict:
    path = ROOT / "config" / "live.json"
    if not path.exists():
        sys.exit("config/live.json not found. Copy config/live.example.json and fill it in.")
    return json.loads(path.read_text())


def build_live() -> None:
    import asyncio

    from mip.live import run_dialog

    out_dir = ROOT / "outputs"
    result = asyncio.run(run_dialog(load_live_config(), out_dir))
    (out_dir / "live.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Wrote outputs/live.json and outputs/live.wav")


def build_probe() -> None:
    import asyncio

    from mip.probe import run_probe, write_summary

    summary = asyncio.run(run_probe(load_live_config()))
    path = write_summary(summary, ROOT / "outputs")
    print(json.dumps({key: summary[key] for key in ("counts", "first_text_ms")}, indent=2))
    print(f"Wrote {path.relative_to(ROOT)}")


def build_test() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "test"), top_level_dir=str(ROOT / "test"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    config = load_config("test")
    build_ablation(config)
    build_baseline(config)


def main(targets: list[str]) -> None:
    if not targets:
        sys.exit(__doc__)
    known = {"test", "ablation", "data", "baseline", "live", "probe", "all", "clean"}
    unknown = set(targets) - known
    if unknown:
        sys.exit(f"Unknown targets: {sorted(unknown)}\n{__doc__}")

    config = load_config("default")
    if "clean" in targets:
        for name in ("outputs", "data"):
            shutil.rmtree(ROOT / name, ignore_errors=True)
    if "test" in targets:
        build_test()
    if "data" in targets or "all" in targets:
        build_data(config)
    if "baseline" in targets or "all" in targets:
        build_baseline(config)
    if "ablation" in targets or "all" in targets:
        build_ablation(config)
    if "live" in targets:
        build_live()
    if "probe" in targets:
        build_probe()


if __name__ == "__main__":
    main(sys.argv[1:])
