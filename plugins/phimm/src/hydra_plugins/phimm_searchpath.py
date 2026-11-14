"""Make ASR and verl configs available to inheritable recipe YAML files."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class PhimmSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Register package roots independently of the selected primary config."""
        search_path.append(provider="phimm", path="pkg://recipe.phimm.config")
        search_path.append(provider="phimm", path="pkg://verl.trainer.config")
