#!/usr/bin/env python3
"""Lightweight markdown link checker for local repository links."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LINK_RE = re.compile(r"(?<!\!)\[[^\]]+\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
EXCLUDE_DIRS = {".venv", ".git", "node_modules"}


def iter_markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.md")
        if not any(
            part in EXCLUDE_DIRS or part.startswith("build")
            for part in path.relative_to(REPO_ROOT).parts
        )
    )


def strip_code(text: str) -> str:
    text = FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    return text


def normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target:
        target = target.split(" ", 1)[0]
    return target


def should_skip(target: str) -> bool:
    return (
        not target
        or target.startswith("#")
        or "://" in target
        or target.startswith("mailto:")
        or target.startswith("data:")
    )


def check_link(doc_path: Path, target: str) -> str | None:
    target = normalize_target(target)
    if should_skip(target):
        return None

    target_path = target.split("#", 1)[0]
    if not target_path:
        return None

    resolved = (doc_path.parent / target_path).resolve()

    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return f"{doc_path.relative_to(REPO_ROOT)} -> {target}: escapes repository root"

    if resolved.exists():
        return None

    return f"{doc_path.relative_to(REPO_ROOT)} -> {target}: target does not exist"


def main() -> int:
    errors: list[str] = []

    for doc_path in iter_markdown_files():
        text = strip_code(doc_path.read_text(encoding="utf-8"))
        for raw_target in LINK_RE.findall(text):
            error = check_link(doc_path, raw_target)
            if error:
                errors.append(error)

    if errors:
        print("Markdown link check failed:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Markdown link check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
