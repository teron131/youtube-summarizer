#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys

# Use tomllib from standard library (Python 3.11+)
try:
    import tomllib
except ImportError:
    print("Error: This script requires Python 3.11+ (missing tomllib).")
    sys.exit(1)


def run_uv(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a uv command with consistent error handling."""
    cmd = ["uv", *args]
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=False)  # noqa: S603
        if check and result.returncode != 0:
            print(f"Error running command: {result.stderr or result.stdout}")
            sys.exit(result.returncode)
        return result
    except Exception as e:
        print(f"Failed to execute command: {e}")
        sys.exit(1)


def parse_package_name(dep_string: str) -> str:
    """Extract package name and extras from dependency string.

    Handles: 'requests', 'requests>=2.0', 'requests[security]>=2.0', etc.
    """
    # Remove environment markers
    base = dep_string.split(";")[0].strip()
    # Match name and optional [extras]
    match = re.match(r"^([A-Za-z0-9\.\_\-]+)(\[[^\]]*\])?", base)
    return match.group(0) if match else base


def get_dependencies(pyproject_path: Path) -> dict[str, list[str]]:
    """Parse dependencies from pyproject.toml into groups."""
    try:
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)

        groups = {
            "standard": data.get("project", {}).get("dependencies", []),
            "dev": [],
        }

        # Check dependency-groups (PEP 735 / modern uv)
        if "dependency-groups" in data:
            for group in data["dependency-groups"].values():
                groups["dev"].extend(group)
        # Check tool.uv.dev-dependencies (legacy uv)
        elif "tool" in data and "uv" in data.get("tool", {}):
            groups["dev"].extend(data["tool"]["uv"].get("dev-dependencies", []))

        return {k: [parse_package_name(d) for d in v] for k, v in groups.items() if v}
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        sys.exit(1)


def update_group(pkg_names: list[str], is_dev: bool = False) -> None:
    """Update a group of dependencies by removing and re-adding them."""
    label = "dev" if is_dev else "standard"
    print(f"\nUpdating {label} dependencies: {', '.join(pkg_names)}")

    # Remove packages (strip extras for removal)
    remove_names = [p.split("[")[0] for p in pkg_names]
    remove_args = ["remove", *remove_names]
    if is_dev:
        remove_args.insert(1, "--dev")
    run_uv(remove_args)

    # Re-add packages to get latest versions
    add_args = ["add", *pkg_names]
    if is_dev:
        add_args.insert(1, "--dev")
    run_uv(add_args)


def main() -> None:
    """Main entry point for dependency updates."""
    # 1. Verify environment
    if not shutil.which("uv"):
        print("Error: 'uv' is not installed or not in PATH.")
        sys.exit(1)

    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print("No pyproject.toml found in current directory.")
        sys.exit(1)

    # 2. Extract and process dependencies
    print("Reading dependencies from pyproject.toml...")
    groups = get_dependencies(pyproject)

    if not any(groups.values()):
        print("No dependencies found to update.")
        return

    # 3. Execute updates
    if "standard" in groups:
        update_group(groups["standard"], is_dev=False)

    if "dev" in groups:
        update_group(groups["dev"], is_dev=True)

    print("\nAll dependencies updated to latest versions in pyproject.toml!")


if __name__ == "__main__":
    main()
