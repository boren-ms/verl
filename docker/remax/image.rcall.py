# Pass as extra_rcall_configs=/absolute/path/to/image.rcall.py for this launch.
# Keep the Orange infrastructure image selected by the existing configuration.
import os

DOCKER_IMAGE = os.environ["VERL_DOCKER_IMAGE"]
if not DOCKER_IMAGE.startswith("iridiumsdc.azurecr.io/boren/verl-remax"):
    raise ValueError("Select the dedicated verl image tag or digest")
