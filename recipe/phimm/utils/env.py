import os
from pathlib import Path
import socket
from recipe.phimm.utils.ray_node import RayNode


def get_region():
    """Get the region of the Kubernetes cluster from the environment variable."""
    rcall_kube_cluster = os.environ.get("RCALL_KUBE_CLUSTER", "")
    cluster_region = rcall_kube_cluster.split("-")[1] if "-" in rcall_kube_cluster else None
    return cluster_region


REGION_STORAGES = {
    "southcentralus": "orngscuscresco",
    "westus2": "orngwus2cresco",
    "uksouth": "orngcresco",
}


class EnvBase:
    """Base class for environment settings."""

    def envs(self):
        """Get the environment variables."""
        raise NotImplementedError

    def prepare(self, forced=False):
        raise NotImplementedError


class OrngEnv(EnvBase):
    """Class to manage orange settings."""

    def __init__(self, region=None):
        """Initialize the OrngSetting with the specified region."""
        self.region = region or get_region()

        if not self.region:
            self.region = "westus2"
            print("Warning: RCALL_KUBE_CLUSTER not set, defaulting region to westus2")
        self._region_storage = REGION_STORAGES.get(self.region, "orngscuscresco")
        self._user = os.environ.get("OPENAI_USER", "boren")

    def envs(self):
        """Get the environment variables for the OrngSetting."""
        return {
            "DATA_STORAGE": self._region_storage,
            "DATA_PATH": self.data_path,
            "USER_HOME_PATH": self.user_home_path,
            "USER_DATA_PATH": self.user_data_path,
            "USER_OUTPUT_PATH": self.user_output_path,
        }

    def prepare(self, forced=False):
        """Prepare the environment by creating necessary directories."""
        RayNode().prepare(forced=forced)

    @property
    def data_storage(self):
        """Get the data storage account based on the region."""
        return self._region_storage

    @property
    def data_path(self):
        """Get the data storage account based on the region."""
        return f"az://{self.data_storage}/data"

    @property
    def user_home_path(self):
        """Get the storage path based on the region."""
        return f"{self.data_path}/{self._user}"

    @property
    def user_data_path(self):
        """Get the user data storage path based on the region."""
        return f"{self.user_home_path}/data"

    @property
    def user_output_path(self):
        """Get the user output storage path based on the region."""
        return f"{self.user_home_path}/outputs"


class LocalEnv(EnvBase):
    """Class to manage local environment variables."""

    def __init__(self):
        """Initialize the LocalEnv class."""
        self.hostname = socket.gethostname()
        self.user = os.environ.get("USER", "boren")
        self.data_path = Path("/home/")
        self.user_home_path = self.data_path / self.user
        self.user_data_path = self.user_home_path / "data"
        self.user_output_path = self.user_home_path / "outputs"

    def envs(self):
        """Get the environment variables."""
        return {
            "DATA_PATH": str(self.data_path),
            "USER_HOME_PATH": str(self.user_home_path),
            "USER_DATA_PATH": str(self.user_data_path),
            "USER_OUTPUT_PATH": str(self.user_output_path),
        }

    def prepare(self, forced=False):
        """Prepare the local environment by creating necessary directories."""
        print("Skipping local env preparation.")


class EnvMgr:
    """Class to manage environment variables."""

    def __init__(self):
        """Initialize the EnvMgr class."""

        self.env = OrngEnv() if "RCALL_KUBE_CLUSTER" in os.environ else LocalEnv()
        print(f"Using Env: {self.env.__class__.__name__}")

    def envs(self):
        """Get the environment variables."""
        return self.env.envs()

    def prepare(self, forced=False):
        """Prepare the environment by creating necessary directories."""
        self.env.prepare(forced=forced)
