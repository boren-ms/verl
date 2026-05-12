# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree vLLM plugin for Qwen3.5-Audio."""

from .plugin import register, register_transformers_config

register_transformers_config()

__all__ = ["register", "register_transformers_config"]
