# Importing this recipe package registers the self-contained Qwen3.5-Audio HF
# classes (model_type "qwen3_5_audio") with the Transformers Auto* registries.
#
# This runs in EVERY process that touches the recipe — the driver (which runs
# ``recipe.phimm.*`` entrypoints) and every Ray worker (which imports this
# package transitively when loading ``pkg://recipe.phimm.*`` custom dataset /
# reward classes). It lets ``from_pretrained(..., trust_remote_code=False)``
# resolve to the installed package instead of the checkpoint's bundled remote
# code. Best-effort: never break ``recipe.phimm`` imports if the plugin is
# absent (e.g. on a non-audio dev box that didn't run quick_install).
try:
    from hf_qwen35_audio import register_hf_audio_model

    register_hf_audio_model()
except Exception:
    pass
