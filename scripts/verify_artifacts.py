#!/usr/bin/env python3
"""Verify frozen HOTC 2026 release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifact-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository-shaped root containing generated artifacts and weights.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Repository root containing pinned external source; defaults to artifact-root.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=("full", "runtime"),
        help="Artifact profile to verify; may be repeated. Defaults to runtime.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    source_root = (args.source_root or root).resolve()
    profiles = set(args.profile or ["runtime"])
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results: list[dict] = []
    failures: list[str] = []

    for artifact in manifest["artifacts"]:
        if profiles.isdisjoint(artifact["profiles"]):
            continue
        artifact_base = root if artifact.get("bundle") == "model-weights" else source_root
        path = artifact_base / artifact["path"]
        result = {
            "path": artifact["path"],
            "role": artifact["role"],
            "exists": path.exists(),
        }
        if not path.exists():
            failures.append(f"missing: {artifact['path']}")
            results.append(result)
            continue
        if "sha256" in artifact:
            actual = sha256(path)
            result["sha256"] = actual
            result["sha256_ok"] = actual == artifact["sha256"]
            if not result["sha256_ok"]:
                failures.append(f"SHA-256 mismatch: {artifact['path']}")
        if "size" in artifact:
            actual_size = path.stat().st_size
            result["size"] = actual_size
            result["size_ok"] = actual_size == artifact["size"]
            if not result["size_ok"]:
                failures.append(f"size mismatch: {artifact['path']}")
        if "glob" in artifact:
            actual_count = sum(1 for candidate in path.glob(artifact["glob"]) if candidate.is_file())
            result["count"] = actual_count
            result["count_ok"] = actual_count == artifact["count"]
            if not result["count_ok"]:
                failures.append(
                    f"file count mismatch: {artifact['path']} "
                    f"({actual_count} != {artifact['count']})"
                )
        results.append(result)

    external = manifest["external_source"]
    if not profiles.isdisjoint(external["profiles"]):
        source = source_root / external["path"]
        runtime_file = source / external["runtime_file"]
        result = {
            "path": external["path"],
            "role": "pinned official SAM 3 source",
            "exists": source.is_dir() and runtime_file.is_file(),
        }
        if result["exists"]:
            actual_sha256 = sha256(runtime_file)
            result["runtime_file"] = external["runtime_file"]
            result["runtime_sha256"] = actual_sha256
            result["runtime_sha256_ok"] = actual_sha256 == external["runtime_sha256"]
            if not result["runtime_sha256_ok"]:
                failures.append(f"SHA-256 mismatch: {runtime_file}")
        if (source / ".git").exists():
            try:
                actual_commit = subprocess.check_output(
                    ["git", "-C", str(source), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                actual_commit = ""
            result["commit"] = actual_commit
            result["commit_ok"] = actual_commit == external["commit"]
            if not result["commit_ok"]:
                failures.append(f"commit mismatch: {external['path']}")
        elif not result["exists"]:
            failures.append(f"missing: {external['path']}")
        results.append(result)

    report = {
        "format": "hotc2026-hyperdam-artifact-verification-v1",
        "artifact_root": str(root),
        "profiles": sorted(profiles),
        "passed": not failures,
        "failures": failures,
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
