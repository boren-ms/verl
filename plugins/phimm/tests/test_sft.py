"""Check Base3a data parity, Qwen audio supervision, and SFT batch contracts."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from recipe.phimm.agentic.dataset import _prepare_data_confs
from recipe.phimm.agentic.scoring import aggregate_error_rates, score_response
from recipe.phimm.data.prompts import get_task_output
from recipe.phimm.reward.asr_response import get_hyp_text, parse_jsonl_response
from recipe.phimm.sft.dataset import ASRSFTDataset


def load_config(kind="sft", overrides=()):
    root = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev"
    name = "sft_base3a_2611r2" if kind == "sft" else "remax_main_base3a_2611r2"
    OmegaConf.register_new_resolver("eval", lambda text: eval(text, {"__builtins__": {}}, {}), replace=True)
    with initialize_config_dir(config_dir=str(root / kind), version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


class AudioProcessor:
    chat_template = "test"

    def __init__(self):
        self.audios = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        assert not tokenize
        text = "".join(f"<{m['role']}>\n{m['content']}<eot>\n" for m in messages)
        return text + ("<assistant>\n" if add_generation_prompt else "")

    def __call__(self, text, audios, return_tensors):
        assert return_tensors == "pt"
        waveform, sample_rate = audios[0]
        assert sample_rate == 16000
        self.audios.append(waveform)
        # Mimic the audio processor expanding the prompt's audio marker.
        ids = Tokenizer().encode(text[0].replace("<audio>", "AAA"))
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "input_audio_embeds": torch.ones(1, len(waveform) // 160, 4),
            "audio_attention_mask": torch.ones(1, len(waveform) // 160),
        }


@pytest.fixture
def local_config(tmp_path):
    cfg = load_config()
    audio = tmp_path / "sample.wav"
    sf.write(audio, np.ones(3200, dtype=np.float32) * 0.1, 16000)
    row = {
        "id": "example",
        "audio_path": str(audio),
        "url": str(audio),
        "text": "Hello <NE>world</NE>",
        "transcript": "Hello world",
        "display": "Hello world",
        "display_ne_tagged": "Hello <NE>world</NE>",
        "language": "English",
        "parent_audio_path": "parent.wav",
        "seg_start": 0.0,
        "seg_end": 0.2,
        "seg_index": 0,
        "n_segments": 1,
    }
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    for split in ("train_data", "val_data"):
        for group in cfg.data[split]:
            for source in group if OmegaConf.is_list(group) else [group]:
                source.jsonl_paths = str(manifest)
    for entry in cfg.data.mix[0].datasource:
        entry.sample = 2
    return cfg


def test_sft_reuses_exact_base3a_data_and_model():
    sft, rl = load_config(), load_config("main")
    for key in ("train_data", "val_data", "mix", "shuffle", "train_batch_size"):
        assert sft.data[key] == rl.data[key]
    assert sft.data.get("random_segment") is rl.data.get("random_segment") is None
    assert sum(entry.sample for entry in sft.data.mix[0].datasource) == 55664
    cohorts = [source for group in sft.data.val_data for source in group]
    assert len(cohorts) == 9
    for key in ("path", "external_lib", "lora_rank", "lora_alpha", "target_modules", "exclude_modules"):
        assert sft.model[key] == rl.actor_rollout_ref.model[key]
    assert sft.optim.lr == rl.actor_rollout_ref.actor.optim.lr
    assert sft.trainer.total_epochs == rl.trainer.total_epochs == 2
    assert "actor_rollout_ref" not in sft
    assert sft.engine.strategy == "fsdp2"
    assert sft.data.custom_cls.name == "ASRSFTDataset"


def test_sft_model_and_lora_overrides():
    cfg = load_config(overrides=["model.path=/models/new-qwen", "model.lora_rank=0", "trainer.nnodes=1"])
    assert cfg.model.path == "/models/new-qwen"
    assert cfg.model.lora_rank == 0
    assert cfg.trainer.nnodes == 1


def test_sft_generation_validation_matches_remax_cohorts_and_scoring():
    directory = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/sft"
    baseline = load_config("main")
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(
            config_name="eval_sft_base3a_2611r2",
            overrides=["actor_rollout_ref.model.path=/models/sft-export", "trainer.experiment_name=sft-val"],
        )
    OmegaConf.resolve(cfg)
    assert cfg.data.val_data == baseline.data.val_data
    assert cfg.reward_map == baseline.reward_map
    assert cfg.rewards == baseline.rewards
    assert cfg.actor_rollout_ref.rollout.custom == baseline.actor_rollout_ref.rollout.custom
    assert cfg.trainer.val_only
    assert cfg.trainer.resume_mode == "disable"
    assert cfg.actor_rollout_ref.model.lora_rank == 0
    assert cfg.data.mix is None


def test_real_source_loading_keeps_train_budgets_and_all_val_rows(local_config):
    cfg = local_config
    train = ASRSFTDataset(cfg.data.train_files, Tokenizer(), cfg.data, AudioProcessor())
    val = ASRSFTDataset(cfg.data.val_files, Tokenizer(), cfg.data, AudioProcessor())
    assert len(train) == 12
    assert len(val) == 9
    assert {row["data_source"] for row in train.ds} == {entry.name for entry in cfg.data.mix[0].datasource}
    for dataset in (train, val):
        for index in range(len(dataset)):
            sample = dataset[index]
            supervised = sample["input_ids"][sample["loss_mask"].bool()].tolist()
            assert bytes(supervised).decode() == dataset.ds[index]["reward_model"]["gt_output"] + "<eot>\n"
            assert sample["loss_mask"][0] == 0
            assert sample["loss_mask"][-1] == 1
            assert torch.equal(sample["position_ids"], torch.arange(len(sample["input_ids"])))
            assert "input_audio_embeds" in sample["multi_modal_inputs"]
            assert "input_ids" not in sample["multi_modal_inputs"]


def test_jsonl_sft_supervises_one_object_with_original_text(local_config):
    train = ASRSFTDataset(local_config.data.train_files, Tokenizer(), local_config.data, AudioProcessor())
    assert len(train) == 12
    labels = set()
    for index, row in enumerate(train.ds):
        sample = train[index]
        assert "Return JSONL only" in row["prompt"][0]["content"]
        supervised = bytes(sample["input_ids"][sample["loss_mask"].bool()].tolist()).decode()
        line = supervised.removesuffix("<eot>\n")
        [parsed] = parse_jsonl_response(line)
        assert parsed["text"] == row["reward_model"]["ground_truth"]
        labels.add(parsed["language"])
        assert "\n" not in line
    assert labels == {"Chinese", "English"}


def test_sft_can_mix_asr_jsonl_and_plain_diarization_targets(local_config, tmp_path):
    cfg = local_config
    wav = str(tmp_path / "sample.wav")
    target = "speaker1 Hello. speaker2 Hi."
    diary = tmp_path / "diarization.jsonl"
    diary.write_text(
        json.dumps(
            {
                "audio_path": wav,
                "data_source": "moe_ami_train",
                "prompt": "Identify each speaker in the transcript.<audio>",
                "text": "Hello. Hi.",
                "gt_output": target,
            }
        )
        + "\n"
    )
    formatter = {"verl_format": {"prompt_key": "prompt"}}
    cfg.data.mix = None
    cfg.data.train_data = [
        {
            "dataset_name": "jsonl",
            "jsonl_paths": str(tmp_path / "data.jsonl"),
            "add_task_info": {"style": "verb"},
            "post_process": formatter,
        },
        {
            "dataset_name": "jsonl",
            "jsonl_schema": "diarization",
            "jsonl_paths": str(diary),
            "sft_target_format": "text",
            "post_process": formatter,
        },
    ]
    ds = ASRSFTDataset(cfg.data.train_files, Tokenizer(), cfg.data, AudioProcessor())
    assert len(ds) == 2
    outputs = []
    for sample in ds:
        supervised = sample["input_ids"][sample["loss_mask"].bool()].tolist()
        outputs.append(bytes(supervised).decode().removesuffix("<eot>\n"))
    assert parse_jsonl_response(outputs[0])[0]["language"] == "English"
    assert outputs[1] == target


def test_ami_speaker_jsonl_preset_uses_only_ami_and_supervises_one_line_per_speaker(tmp_path):
    root = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/sft"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(config_name="sft_ami_diarization_2611r2")
    assert cfg.data.mix is None
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(3200, dtype=np.float32), 16000)
    for is_train, split in [(True, "train"), (False, "dev")]:
        [source] = _prepare_data_confs(cfg.data, is_train=is_train)
        assert f"/manifests/moe_ami_{split}/" in source["jsonl_paths"]
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "audio_path": str(wav),
                    "prompt": "Old prompt.<audio>",
                    "text": "Good morning today.",
                    "gt_output": "speaker1 Good speaker2 morning speaker1 today.",
                    "reference_words": [
                        {"speaker": "speaker1", "start": 0.1, "end": 0.2, "text": "today."},
                        {"speaker": "speaker1", "start": 0.0, "end": 0.04, "text": "Good"},
                        {"speaker": "speaker2", "start": 0.04, "end": 0.1, "text": "morning"},
                    ],
                }
            )
            + "\n"
        )
        key = "train_data" if is_train else "val_data"
        cfg.data[key][0].jsonl_paths = str(manifest)
        path = cfg.data.train_files if is_train else cfg.data.val_files
        ds = ASRSFTDataset(path, Tokenizer(), cfg.data, AudioProcessor())
        assert len(ds) == 1
        assert "one object per speaker" in ds.ds[0]["prompt"][0]["content"]
        assert ds.ds[0]["reward_model"]["ground_truth"] == "Good morning today."
        sample = ds[0]
        tokens = sample["input_ids"][sample["loss_mask"].bool()].tolist()
        lines = bytes(tokens).decode().removesuffix("<eot>\n").splitlines()
        assert list(map(json.loads, lines)) == [
            {"spk": 1, "text": "Good today."},
            {"spk": 2, "text": "morning"},
        ]


def test_jsonl_full_eval_extracts_transcript_and_counts_invalid_responses():
    root = Path(__file__).resolve().parents[1] / "src/recipe/phimm/config/v2611_dev/eval"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(
            config_name="eval_openfull_zip",
            overrides=["actor_rollout_ref.model.path=/models/jsonl", "trainer.experiment_name=jsonl-eval"],
        )
    sources = _prepare_data_confs(cfg.data, is_train=False)
    assert len(sources) == 35 and all(source.get("add_task_info") for source in sources)
    line = json.dumps({"language": "English", "text": 'He said "hi".\nThen left.'}, ensure_ascii=False)
    assert parse_jsonl_response(line)[0]["text"] == 'He said "hi".\nThen left.'
    other = json.dumps({"language": "English", "text": "Again."})
    assert get_hyp_text([line + "\n" + other, "invalid"]) == 'He said "hi".\nThen left. Again. invalid'
    duplicate = '{"language":"English","text":"one","text":"two"}'
    assert parse_jsonl_response(duplicate) == [{"language": None, "text": duplicate}]
    assert parse_jsonl_response('{"language":"English","text":""}')[0]["text"] == ""
    row = {
        "data_source": "ls_clean",
        "reward_model": {"ground_truth": "hello"},
        "extra_info": {"language": "English"},
    }
    valid = score_response(cfg, row, ['{"language":"English","text":"hello"}\n{"language":"English","text":""}'])
    invalid = score_response(cfg, row, ['{"language":"English","text":"hello"}\nlegacy output'])
    assert valid["n_err"] == 0
    assert invalid["n_err"] == 2
    assert score_response(cfg, row, ["hello"])["n_err"] == 0
    results = {key: [valid[key], invalid[key]] for key in ("n_json_valid", "n_json_total")}
    assert aggregate_error_rates([row["data_source"]] * 2, results) == {"val-aux/ls_clean/json_valid_rate/mean@1": 0.5}


def test_sft_keeps_list_audio_and_validation_segments(local_config):
    cfg = local_config
    ds = ASRSFTDataset(cfg.data.val_files, Tokenizer(), cfg.data, AudioProcessor())
    row = dict(ds.ds[0])
    row["audio_path"] = [row["audio_path"], row["audio_path"]]
    segments = [{"language": "English", "text": "Hi."}, {"language": "French", "text": "Salut."}]
    row["reward_model"]["gt_output"] = get_task_output(components=segments)
    ds.ds = [row, row]
    assert len(ds) == 2  # Parent metadata never collapses supervised rows.
    sample = ds[0]
    target = bytes(sample["input_ids"][sample["loss_mask"].bool()].tolist()).decode().removesuffix("<eot>\n")
    assert list(map(json.loads, target.splitlines())) == segments
    assert len(ds.processor.audios[0]) == 6400


@pytest.mark.parametrize("problem", ["target", "format", "length", "duration", "template"])
def test_sft_rejects_misaligned_or_truncated_targets(local_config, problem):
    cfg = local_config
    ds = ASRSFTDataset(cfg.data.train_files, Tokenizer(), cfg.data, AudioProcessor())
    ds.ds = [dict(ds.ds[0])]
    if problem == "target":
        ds.ds[0]["reward_model"]["gt_output"] = None
    elif problem == "format":
        ds.ds[0]["reward_model"]["gt_output"] = "<src=English><tgt=English>\n<VERBATIM>\nHello"
    elif problem == "length":
        cfg.data.max_length = 1
    elif problem == "duration":
        cfg.data.max_audio_dur = 0.1
    else:
        original = ds.processor.apply_chat_template

        def template(*args, **kwargs):
            result = original(*args, **kwargs)
            return result if kwargs["add_generation_prompt"] else "changed" + result

        ds.processor.apply_chat_template = template
    with pytest.raises(ValueError):
        ds[0]


def test_sft_requires_audio_processor_and_safe_padding():
    cfg = load_config()
    with pytest.raises(ValueError, match="audio processor"):
        ASRSFTDataset(cfg.data.train_files, Tokenizer(), cfg.data)
    cfg.data.truncation = "right"
    with pytest.raises(ValueError, match="truncation=error"):
        ASRSFTDataset(cfg.data.train_files, Tokenizer(), cfg.data, AudioProcessor())


def test_sft_collator_preserves_features_and_response_mask(local_config):
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from verl.utils.model import extract_multi_modal_inputs

    cfg = local_config
    ds = ASRSFTDataset(cfg.data.train_files, Tokenizer(), cfg.data, AudioProcessor())
    samples = [ds[0], ds[1]]
    samples[1]["multi_modal_inputs"]["input_audio_embeds"] = torch.ones(1, 30, 4)
    samples[1]["multi_modal_inputs"]["audio_attention_mask"] = torch.ones(1, 30)
    batch = SFTTensorCollator("no_padding")(samples)
    features = extract_multi_modal_inputs(batch["multi_modal_inputs"])
    assert features["input_audio_embeds"].shape == (2, 30, 4)
    assert features["audio_attention_mask"].shape == (2, 30)
    assert batch["input_ids"].is_nested
    assert batch["loss_mask"].values().sum() == sum(item["loss_mask"].sum() for item in samples)


def test_supervised_loss_only_differentiates_assistant_predictions():
    from verl.workers.utils.losses import sft_loss

    values = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0], requires_grad=True)
    log_probs = torch.nested.nested_tensor_from_jagged(values, torch.tensor([0, 4, 7]))
    mask = torch.nested.as_nested_tensor([torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1])], layout=torch.jagged)
    data = {"pad_mode": "no_padding", "loss_mask": mask, "dp_size": 1, "batch_num_tokens": 4}
    loss, _ = sft_loss(None, {"log_probs": log_probs}, data)
    loss.backward()
    assert loss.item() == pytest.approx((2 + 3 + 5 + 6) / 4)
    torch.testing.assert_close(values.grad, torch.tensor([0, -0.25, -0.25, 0, -0.25, -0.25, 0]))
