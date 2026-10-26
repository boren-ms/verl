"""Copy only tracked ReMax runtime sources, excluding unrelated examples/docs."""
import pathlib
import shutil
import subprocess
import sys

repo = pathlib.Path(__file__).resolve().parents[2]
dest = pathlib.Path(sys.argv[1]).resolve()
dest.mkdir(parents=True, exist_ok=True)
base = {"quick_install.sh", "quick_install_linear_attention.sh", "requirements_vllm.txt", "setup.py", "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"}
files = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0")
for name in files:
    if name not in base and not name.startswith(("verl/", "plugins/qwen35_audio/", "recipe/phimm/")):
        continue
    if name.endswith((".md", ".ipynb")) and name not in base and name != "plugins/qwen35_audio/README.md":
        continue
    source = repo / name
    if source.is_file():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
shutil.copytree(repo / "docker/remax", dest / "docker/remax", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
