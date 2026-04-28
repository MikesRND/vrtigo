#!/usr/bin/env bash
# Check that release-managed version files agree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 - <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path.cwd()
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def extract(pattern: str, text: str, label: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        print(f"Error: could not find version in {label}", file=sys.stderr)
        return None
    return match.group(1)


cmake_text = read_text("CMakeLists.txt")
pyproject_text = read_text("bindings/python/pyproject.toml")

versions: dict[str, str | None] = {
    "CMakeLists.txt": extract(
        r"project\s*\(\s*vrtigo\b.*?\bVERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
        cmake_text,
        "CMakeLists.txt",
    ),
    "bindings/python/pyproject.toml": extract(
        r'^version\s*=\s*"([^"]+)"',
        pyproject_text,
        "bindings/python/pyproject.toml",
    ),
}

try:
    manifest = json.loads(read_text(".release-please-manifest.json"))
except json.JSONDecodeError as exc:
    print(f"Error: invalid .release-please-manifest.json: {exc}", file=sys.stderr)
    sys.exit(1)

manifest_version = manifest.get(".")
if not isinstance(manifest_version, str):
    print('Error: .release-please-manifest.json must contain string key "."', file=sys.stderr)
    manifest_version = None
versions[".release-please-manifest.json"] = manifest_version

errors: list[str] = []
for path, version in versions.items():
    display = version if version is not None else "<missing>"
    print(f"{path}: {display}")
    if version is None:
        errors.append(f"{path}: missing version")
    elif VERSION_RE.fullmatch(version) is None:
        errors.append(f"{path}: expected MAJOR.MINOR.PATCH, found {version!r}")

for path, text in {
    "CMakeLists.txt": cmake_text,
    "bindings/python/pyproject.toml": pyproject_text,
}.items():
    if "x-release-please-version" not in text:
        errors.append(f"{path}: missing x-release-please-version marker")

present_versions = {version for version in versions.values() if version is not None}
if len(present_versions) > 1:
    errors.append("version mismatch: release-managed files must agree")

if errors:
    print("\nVersion check failed:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    sys.exit(1)

print("Version check passed")
PY
