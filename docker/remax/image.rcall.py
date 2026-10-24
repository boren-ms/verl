# Pass this file with skip_rcall_user_config=true: Orange's normal user config
# otherwise loads after project configs and overwrites DOCKER_IMAGE.
import os
from pathlib import Path

# Load the normal Orange network/storage/infra settings before selecting the
# job image. Do not modify the user's shared configuration.
_user_config = Path.home() / ".rcall_config.py"
exec(compile(_user_config.read_text(), str(_user_config), "exec"), globals())
DOCKER_IMAGE = os.environ["VERL_DOCKER_IMAGE"]
if not DOCKER_IMAGE.startswith("iridiumsdc.azurecr.io/boren/verl-remax"):
    raise ValueError("Select the dedicated verl image tag or digest")

# Build-time /tmp markers do not survive this startup path. Create the guard
# immediately before TorchFlow's startup hook. The curated image already supplies the validated dependencies.
EXTRA_SETUP = (
    "set -e\n"
    "test -f /opt/verl-env/base-ray-version.txt\n"
    "touch /tmp/DID_TORCHFLOW_SETUP\n"
    "python -m pip install --no-deps --no-build-isolation -e /root/code/openai/torchflow\n"
) + EXTRA_SETUP
ENVIRONMENT["SKIP_TORCHFLOW_SETUP_CHECK"] = "1"
