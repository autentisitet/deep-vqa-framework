"""Documentation links must resolve to real files in this repository."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS = [
    *(ROOT / name for name in ("README.md", "README_zh.md", "AGENTS.md", "SECURITY.md", "DISCLAIMER.md")),
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / ".github").rglob("*.md")),
]


def _relative_links(document: Path) -> list[str]:
    text = document.read_text(encoding="utf-8")
    links = []
    for raw in re.findall(r"\]\(([^)]+)\)", text):
        target = raw.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        links.append(target)
    return links


def test_every_relative_document_link_resolves() -> None:
    missing = [
        f"{document.relative_to(ROOT)} -> {target}"
        for document in DOCUMENTS
        for target in _relative_links(document)
        if not (document.parent / target).exists()
    ]
    assert missing == []


def test_no_obsolete_knowledge_path_remains() -> None:
    offenders = [
        str(document.relative_to(ROOT))
        for document in DOCUMENTS
        if "knowledge/" in document.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_frontend_does_not_link_to_paths_nginx_cannot_serve() -> None:
    """Nginx serves only frontend/, so repository docs must be linked out."""
    index = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert "/knowledge/" not in index
    assert "docs/deployment.md" in index
