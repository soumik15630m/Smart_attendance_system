"""Fail if pyproject.toml's [project.dependencies] and requirements.txt
list different package names. Catches drift like alembic being declared
in one file but missing from the other (the actual cause of a CI break).
"""

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def _package_name(requirement: str) -> str:
    """Strip version specifiers/extras, return the bare package name, lowercased."""
    name = re.split(r"[<>=!~\[;]", requirement, maxsplit=1)[0]
    return name.strip().lower()


def _requirements_txt_names() -> set[str]:
    text = (ROOT / "requirements.txt").read_text()
    names = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-r "):
            names.add(_package_name(line))
    return names


def _pyproject_names() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    return {_package_name(d) for d in deps}


def main() -> int:
    req_names = _requirements_txt_names()
    proj_names = _pyproject_names()

    missing_from_requirements = proj_names - req_names
    missing_from_pyproject = req_names - proj_names

    if missing_from_requirements:
        print(
            "In pyproject.toml but missing from requirements.txt: "
            f"{sorted(missing_from_requirements)}"
        )
    if missing_from_pyproject:
        print(
            "In requirements.txt but missing from pyproject.toml: "
            f"{sorted(missing_from_pyproject)}"
        )

    if missing_from_requirements or missing_from_pyproject:
        return 1

    print("pyproject.toml and requirements.txt are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
