"""Stage tracked runtime sources into a fresh Docker source directory."""

import argparse
import shutil
import subprocess
from pathlib import Path


def stage(destination):
    repository = Path(__file__).resolve().parents[2]
    destination = Path(destination).resolve()
    if destination == repository or repository.is_relative_to(destination):
        raise ValueError("Destination must not contain the source repository")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty destination to avoid stale source files")
    files = subprocess.check_output(["git", "-C", str(repository), "ls-files", "-z"]).decode().split("\0")
    base = {"quick_install.sh", "setup.py", "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"}
    for name in files:
        if name not in base and not name.startswith(
            ("verl/", "plugins/phimm/", "plugins/qwen35_audio/", "docker/remax/")
        ):
            continue
        if "__pycache__" in Path(name).parts or name.endswith((".pyc", ".ipynb")):
            continue
        source = repository / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    stage(parser.parse_args().destination)
