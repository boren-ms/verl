"""Official EER failures must not appear as perfect recognition."""

import pytest

from recipe.phimm.reward import asr_inhouse_measure as measure


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"Metrics": []},
        {"Metrics": [{"EntityInfo": {}}]},
        {"Metrics": [{"EntityInfo": {"NumTransEnts": 0, "NumTransEntsMatched": 0}}]},
        {"Metrics": [{"EntityInfo": {"NumTransEnts": 2, "NumTransEntsMatched": 3}}]},
    ],
)
def test_strict_eer_rejects_failed_or_invalid_metrics(monkeypatch, record):
    monkeypatch.setattr(measure, "get_metrics", lambda *a, **k: record)
    with pytest.raises(RuntimeError):
        measure._compute_eer("hello <NE> World </NE>", "hello world", strict=True)


def test_official_counts_used(monkeypatch):
    monkeypatch.setattr(
        measure,
        "get_metrics",
        lambda *a, **k: {"Metrics": [{"EntityInfo": {"NumTransEnts": 3, "NumTransEntsMatched": 2}}]},
    )
    assert measure._compute_eer("<NE> a </NE>", "b", strict=True) == (1, 3, 1 / 3)


def test_dter_and_eer_keep_their_own_reference(monkeypatch):
    monkeypatch.setattr(measure, "ensure_pack_dir", lambda *a, **k: None)
    refs = {}

    def dter(ref, hyp, **kwargs):
        refs["display"] = ref
        return 0, 3, 0.0, {}

    def eer(ref, hyp, **kwargs):
        refs["entity"] = ref
        assert kwargs["strict"]
        return 1, 2, 0.5

    monkeypatch.setattr(measure, "_compute_dter", dter)
    monkeypatch.setattr(measure, "_compute_eer", eer)
    result = measure.eval_score(
        "<VERBAT>hello world</VERBAT>",
        "[0 1] display reference",
        extra_info={"entity_transcription": "<NE> original </NE> reference"},
        compute_eer=True,
        strict_eer=True,
    )
    assert refs == {"display": "display reference", "entity": "<NE> original </NE> reference"}
    assert result["eer_n_ref"] == 2
    assert result["eer_n_err"] == 1


def test_dtest_uses_display_nondisfluency_and_locale(monkeypatch):
    observed = {}
    monkeypatch.setattr(measure, "ensure_pack_dir", lambda pack, root: observed.update(pack=(pack, root)))
    monkeypatch.setattr(measure, "_compute_dter", lambda *a, **k: (0, 3, 0.0, {}))

    def metrics(ref, hyp, *args, **kwargs):
        observed.update(ref=ref, args=args, locale=kwargs["locale"])
        return {"Metrics": [{"EntityInfo": {"NumTransEnts": 2, "NumTransEntsMatched": 1}}]}

    monkeypatch.setattr(measure, "get_metrics", metrics)
    result = measure.eval_score(
        "recognized",
        "[0 1] display <NE> 2016 </NE>",
        extra_info={"entity_transcription": "different <NE> twenty sixteen </NE>"},
        compute_eer=True,
        strict_eer=True,
        eer_reference="display",
        eer_mode="nondisfluency",
        locale="en-us",
    )
    assert observed == {
        "pack": (measure.DEFAULT_PACK_DIR, measure.DEFAULT_REMOTE_PACK_ROOT),
        "ref": "display <NE> 2016 </NE>",
        "args": ("--nondisfluency",),
        "locale": "en-us",
    }
    assert result["eer"] == 0.5


def test_ter_does_not_invoke_entity_scoring(monkeypatch):
    monkeypatch.setattr(measure, "ensure_pack_dir", lambda *a, **k: None)
    monkeypatch.setattr(measure, "_compute_dter", lambda *a, **k: (1, 200, 0.005, {}))

    def unexpected(*a, **k):
        pytest.fail("TER-only suite invoked GetMetrics entity scoring")

    monkeypatch.setattr(measure, "_compute_eer", unexpected)
    result = measure.eval_score("recognized", "reference", compute_eer=False)
    assert result["dter_p_err"] == 0.005
    assert "eer" not in result


def test_low_nonzero_ter_is_normalized_from_counts(monkeypatch):
    class Backend:
        def compute_ter_from_strings(self, **kwargs):
            return {"summary": {"ter_info": {"number_of_edits": 1, "number_of_tokens": 200, "display_ter": 0.5}}}

    monkeypatch.setattr(measure, "_get_ter_backend", lambda *a, **k: Backend())
    assert measure._compute_dter("reference", "recognition")[:3] == (1, 200, 0.005)


@pytest.mark.parametrize("version", ["2.18.0", "2.28.4"])
def test_setup_checks_actual_speechinsight_version_before_installing_ter(tmp_path, monkeypatch, version):
    import json

    binary = measure._metric_bin(tmp_path)
    binary.parent.mkdir(parents=True)
    binary.write_text("binary")
    (tmp_path / "speechinsight_tools_latest" / measure.TER_WHEEL_NAME).write_text("wheel")
    (binary.parent / "GetMetrics.deps.json").write_text(
        json.dumps({"libraries": {f"Microsoft.Speech.ASR.Metrics.Metrics.NetCore.AnyCPU/{version}.0": {}}})
    )
    monkeypatch.setattr(measure.shutil, "which", lambda _: None)
    error = "Expected SpeechInsight 2.28.4" if version == "2.18.0" else "TER wheel SHA-256 mismatch"
    with pytest.raises(RuntimeError, match=error):
        measure.ensure_pack_dir(tmp_path)
    assert str(tmp_path) not in measure._SETUP_DONE


def test_setup_does_not_fall_back_to_legacy_tools(tmp_path, monkeypatch):
    old_binary = tmp_path / "speechinsight_tools/linux-x64/framework-dependent/GetMetrics"
    old_binary.parent.mkdir(parents=True)
    old_binary.write_text("legacy binary")
    monkeypatch.setattr(measure.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="bbb is unavailable"):
        measure.ensure_pack_dir(tmp_path)
    assert str(tmp_path) not in measure._SETUP_DONE


@pytest.mark.parametrize("corrupt", ["wheel", "binary", None])
def test_setup_verifies_both_manifest_hashes_before_install(tmp_path, monkeypatch, corrupt):
    import copy
    import hashlib
    import json

    binary = measure._metric_bin(tmp_path)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    wheel = tmp_path / "speechinsight_tools_latest" / measure.TER_WHEEL_NAME
    wheel.write_bytes(b"wheel")
    (binary.parent / "GetMetrics.deps.json").write_text(
        json.dumps({"libraries": {"Microsoft.Speech.ASR.Metrics.Metrics.NetCore.AnyCPU/2.28.4.0": {}}})
    )
    manifest = copy.deepcopy(measure.TOOLS_MANIFEST)
    manifest["ter"]["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest["get_metrics"]["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(measure, "TOOLS_MANIFEST", manifest)
    monkeypatch.setattr(measure.shutil, "which", lambda _: None)
    if corrupt == "wheel":
        wheel.write_bytes(b"changed wheel")
    elif corrupt == "binary":
        binary.write_bytes(b"changed executable")
    error = {
        "wheel": "TER wheel SHA-256 mismatch",
        "binary": "GetMetrics executable SHA-256 mismatch",
        None: "bbb is unavailable",
    }[corrupt]
    with pytest.raises(RuntimeError, match=error):
        measure.ensure_pack_dir(tmp_path)
    assert str(tmp_path) not in measure._SETUP_DONE


def test_bundle_pins_current_moe_wheel_and_executable():
    assert measure.BUNDLE_REVISION == "2026-11-10"
    assert measure.TER_VERSION == "2.22.1"
    assert measure.TER_WHEEL_NAME == "ter-2.22.1-py3-none-any.whl"
    assert measure.TOOLS_MANIFEST["ter"]["sha256"] == "af2a5c37985beaf9df11baf86791d14210d6d4f46bca3c89fe86ea34f8314b84"
    assert (
        measure.TOOLS_MANIFEST["get_metrics"]["sha256"]
        == "329cce87466c14b8d97a5bc05d665ebb36080601ac6057617b65aa260d3ad920"
    )
