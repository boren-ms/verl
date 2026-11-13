"""Check real model assets, audio, data mix and current dataclass schemas."""

import json
from collections import Counter
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from recipe.phimm.agentic.config import adapt_config
from recipe.phimm.agentic.dataset import ASRDataset
from recipe.phimm.main_asr_remax import main as _register_resolvers  # noqa: F401
from recipe.phimm.reward.openasr_expansion import measure_expansion
from recipe.phimm.utils.audio import load_audio

from verl.utils.config import omega_conf_to_dataclass


@hydra.main(config_path="config/v2611_dev/main", config_name="remax_main_base3a_2611r2", version_base=None)
def main(source):
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from vllm_qwen35_audio.plugin import register

    config = adapt_config(source)
    # These lazy normalizers are required by validation/full evaluation and
    # must work before allocating the training/inference workers.
    scorer_checks = {}
    for language, text in {"zh": "繁體中文一百二十三", "vi": "tiếng Việt", "fr": "bonjour le monde"}.items():
        result = measure_expansion(text, text, language)
        assert result["n_err"] == 0 and result["n_ref"] > 0, (language, result)
        scorer_checks[language] = result
    model = omega_conf_to_dataclass(config.actor_rollout_ref.model)
    # engine_workers translates the reference's legacy log_prob_* fields before
    # instantiation. Check the actor, engine and rollout schemas directly here.
    for section in (
        config.actor_rollout_ref.actor,
        config.actor_rollout_ref.ref.fsdp_config,
        config.actor_rollout_ref.rollout,
    ):
        omega_conf_to_dataclass(section)
    assert model.processor is not None, "ASR requires the registered audio processor"
    register()
    inference_model = ModelConfig(model=model.local_path, dtype="bfloat16", max_model_len=config.data.max_model_length)
    inference_processor = MULTIMODAL_REGISTRY.create_processor(inference_model, tokenizer=model.tokenizer)
    train = ASRDataset(config.data.train_files, model.tokenizer, config.data, model.processor)
    val = ASRDataset(config.data.val_files, model.tokenizer, config.data, model.processor)
    counts = dict(Counter(train.ds["data_source"]))
    expected = {item.name: item.sample for group in config.data.mix for item in group.datasource}
    assert counts == expected, (counts, expected)
    assert len(train) == 55664
    assert len(train) // config.data.train_batch_size * config.trainer.total_epochs == 868
    prompt_lengths = {}
    inference_prompt_checks = {}
    for source_name in counts:
        index = next(i for i, name in enumerate(train.ds["data_source"]) if name == source_name)
        row = train.ds[index]
        audio = load_audio(row, config.data.max_audio_dur)
        chat = model.processor if getattr(model.processor, "chat_template", None) else model.tokenizer
        text = chat.apply_chat_template(row[config.data.prompt_key], add_generation_prompt=True, tokenize=False)
        text += (row.get("extra_info") or {}).get("prefix", "") or ""
        features = model.processor(text=[text], audios=[audio], return_tensors="pt")
        raw_ids = model.tokenizer.encode(text, add_special_tokens=False)
        processed = inference_processor.apply(
            ProcessorInputs(raw_ids, inference_processor.info.parse_mm_data({"audio": [audio]}), {}),
            TimingContext(enabled=False),
        )
        assert processed["prompt_token_ids"] == features["input_ids"][0].tolist(), source_name
        inference_prompt_checks[source_name] = {
            "raw_length": len(raw_ids),
            "expanded_length": len(processed["prompt_token_ids"]),
        }
        prompt_lengths[source_name] = features["input_ids"].shape[-1]
        assert prompt_lengths[source_name] <= config.data.max_prompt_length
    report = {
        "training_rows": len(train),
        "training_counts": counts,
        "validation_segments": len(val.ds),
        "validation_parents": len(val),
        "target": 868,
        "expansion_scorers": scorer_checks,
        "prompt_lengths": prompt_lengths,
        "inference_prompt_equivalence": inference_prompt_checks,
        "model": config.actor_rollout_ref.model.path,
        "local_model": model.local_path,
    }
    destination = Path(config.trainer.default_local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preflight.json").write_text(json.dumps(report, indent=2))
    OmegaConf.save(config, destination / "effective_config.yaml", resolve=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
