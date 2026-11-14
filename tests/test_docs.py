"""Keep the documentation honest: links resolve, anchors exist, nothing is left undocumented."""

import re
from pathlib import Path

from curio import cli

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DOCUMENTATION.md"
README = ROOT / "README.md"
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub-style anchor for a heading."""
    text = re.sub(r"[^a-z0-9 _-]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    return {slug(m.group(1)) for m in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", path.read_text())}


def test_relative_links_and_anchors_resolve():
    problems = []
    for md in (README, DOC):
        for target in LINK_RE.findall(md.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, anchor = target.partition("#")
            dest = md if not file_part else (md.parent / file_part).resolve()
            if not dest.exists():
                problems.append(f"{md.name}: missing file {target}")
            elif anchor and anchor not in anchors(dest):
                problems.append(f"{md.name}: missing anchor {target}")
    assert not problems, problems


def test_every_cli_command_and_module_is_documented():
    text = DOC.read_text()
    commands = cli.build_parser()._subparsers._group_actions[0].choices.keys()
    missing = [c for c in commands if f"`{c}`" not in text]
    modules = [p.stem for p in (ROOT / "src" / "curio").glob("*.py") if p.stem != "__init__"]
    missing += [f"curio.{m}" for m in modules if f"`curio.{m}`" not in text]
    assert not missing, f"not documented: {missing}"


def test_readme_points_to_the_documentation():
    assert "docs/DOCUMENTATION.md" in README.read_text()
