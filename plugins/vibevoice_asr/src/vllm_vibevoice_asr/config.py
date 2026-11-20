"""Import-safe config for vLLM spawn workers."""

from vibevoice_asr.compat import prepare_upstream

prepare_upstream()

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig as UpstreamConfig  # noqa: E402


class VibeVoiceConfig(UpstreamConfig):
    """Import compatibility before unpickling upstream config children.

    Spawn unpickles VllmConfig before loading general plugins. Referencing this
    module ensures upstream registrations are prepared in that fresh process.
    """

    def __init__(self, **kwargs):
        # Transformers 5 generates an initializer for subclasses that do not
        # define one, leaving the legacy nested configs as plain dictionaries.
        super().__init__(**kwargs)
