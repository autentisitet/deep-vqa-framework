"""Synchronize the project version from pyproject.toml to runtime/docs files."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def replace(path: Path, pattern: str, replacement: str, *, write: bool = True) -> bool:
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one version marker in {path}, found {count}")
    if write:
        path.write_text(updated, encoding="utf-8")
    return content == updated


def sync(version: str, *, write: bool = True) -> bool:
    changed = not replace(ROOT / "src/__init__.py", r'__version__ = "[^"]+"', f'__version__ = "{version}"', write=write)
    for name in ("README.md", "README_zh.md"):
        changed = not replace(ROOT / name, r"version-[0-9]+\.[0-9]+\.[0-9]+-blue", f"version-{version}-blue", write=write) or changed
        changed = not replace(ROOT / name, r"\*\*(Version|版本)\*\*: [0-9]+\.[0-9]+\.[0-9]+", rf"**\1**: {version}", write=write) or changed
    changed = not replace(
        ROOT / "knowledge/architecture.html",
        r"Deep-VQA-Framework Platform v[0-9]+\.[0-9]+\.[0-9]+",
        f"Deep-VQA-Framework Platform v{version}",
        write=write,
    ) or changed
    changed = not replace(
        ROOT / "docs/openapi.json",
        r'("version":\s*")[0-9]+\.[0-9]+\.[0-9]+(\")',
        rf'\g<1>{version}\g<2>',
        write=write,
    ) or changed
    changed = not replace(
        ROOT / "requirements.txt",
        r"deep-vqa-framework==[0-9]+\.[0-9]+\.[0-9]+",
        f"deep-vqa-framework=={version}",
        write=write,
    ) or changed
    changed = not replace(
        ROOT / "uv.lock",
        r'(name = "deep-vqa-framework"\nversion = ")[0-9]+\.[0-9]+\.[0-9]+(")',
        rf'\g<1>{version}\g<2>',
        write=write,
    ) or changed
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    version = project_version()
    if args.check:
        raise SystemExit(1 if sync(version, write=False) else 0)
    sync(version)
    print(f"Synchronized project version: {version}")


if __name__ == "__main__":
    main()
