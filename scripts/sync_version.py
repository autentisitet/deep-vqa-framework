"""Synchronize the project version from pyproject.toml to runtime/docs files."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    """Return the release version declared in pyproject.toml."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version: object = tomllib.load(handle)["project"]["version"]
    if not isinstance(version, str):
        raise ValueError("project.version must be a string")
    return version


def replace(path: Path, pattern: str, replacement: str, *, write: bool = True) -> bool:
    """Replace exactly one version marker in ``path`` and report whether it changed."""
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Expected one version marker in {path}, found {count}")
    if write:
        path.write_text(updated, encoding="utf-8")
    return content == updated


def sync(version: str, *, write: bool = True) -> bool:
    """Synchronize ``version`` into every runtime and documentation file."""
    changed = not replace(ROOT / "src/__init__.py", r'__version__ = "[^"]+"', f'__version__ = "{version}"', write=write)
    for name in ("README.md", "README_zh.md"):
        changed = (
            not replace(ROOT / name, r"version-[0-9]+\.[0-9]+\.[0-9]+-blue", f"version-{version}-blue", write=write)
            or changed
        )
        # Badge image text and its alt text are separate version markers.
        changed = (
            not replace(
                ROOT / name,
                r'alt="(Version|版本) [0-9]+\.[0-9]+\.[0-9]+"',
                rf'alt="\1 {version}"',
                write=write,
            )
            or changed
        )
        changed = (
            not replace(
                ROOT / name, r"\*\*(Version|版本)\*\*: [0-9]+\.[0-9]+\.[0-9]+", rf"**\1**: {version}", write=write
            )
            or changed
        )
    changed = (
        not replace(
            ROOT / "DISCLAIMER.md",
            r"\*\*Applies to:\*\* deep-vqa-framework v[0-9]+\.[0-9]+\.[0-9]+ and later",
            f"**Applies to:** deep-vqa-framework v{version} and later",
            write=write,
        )
        or changed
    )
    changed = (
        not replace(
            ROOT / "docs/architecture.html",
            r"Deep-VQA-Framework Platform v[0-9]+\.[0-9]+\.[0-9]+",
            f"Deep-VQA-Framework Platform v{version}",
            write=write,
        )
        or changed
    )
    changed = (
        not replace(
            ROOT / "docs/openapi.json",
            r'("version":\s*")[0-9]+\.[0-9]+\.[0-9]+(\")',
            rf"\g<1>{version}\g<2>",
            write=write,
        )
        or changed
    )
    # requirements.txt no longer embeds a "deep-vqa-framework==x.y.z" marker:
    # uv >= 0.11 exports the editable install as "-e ." without a project
    # version comment. The release version is sourced from pyproject.toml and
    # still checked against uv.lock and docs/openapi.json below.
    changed = (
        not replace(
            ROOT / "uv.lock",
            r'(name = "deep-vqa-framework"\nversion = ")[0-9]+\.[0-9]+\.[0-9]+(")',
            rf"\g<1>{version}\g<2>",
            write=write,
        )
        or changed
    )
    return changed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without executing the command."""
    parser = argparse.ArgumentParser(
        description="Synchronize project release version markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python scripts/sync_version.py --check\n  python scripts/sync_version.py",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--version", action="version", version=f"Deep-VQA {project_version()}")
    return parser


def parse_args(argv: list[str] | None = None) -> bool:
    """Produce validated options from argv (or the process command line)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return bool(args.check)


def run(check: bool = False) -> int:
    """Consume parsed options and execute the command."""
    version = project_version()
    if check:
        return 1 if sync(version, write=False) else 0
    sync(version)
    print(f"Synchronized project version: {version}")
    return 0


def main(argv: list[str] | None = None) -> None:
    """Parse command-line options and execute the command."""
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
