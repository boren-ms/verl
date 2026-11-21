from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from recipe.phimm.reward.asr_response import get_hyp_text
from recipe.phimm.utils.languages import get_language_code

# Strip "[start end]" timing tokens emitted anywhere in DisplayTranscription.
_TIME_MARKER_RE = re.compile(r"\[\s*-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s*\]")

# Default TER locale per ISO-639 language code. The dfmetrics TER backend keys
# on a lowercase ``lang-region`` locale (e.g. ``en-us``, ``zh-cn``); pick the
# canonical region used by the in-house 2605 eval set for each language.
_LANG_CODE_TO_LOCALE = {
    "ar": "ar-sa",
    "cs": "cs-cz",
    "da": "da-dk",
    "de": "de-de",
    "en": "en-us",
    "es": "es-es",
    "fi": "fi-fi",
    "fr": "fr-fr",
    "hi": "hi-in",
    "hu": "hu-hu",
    "id": "id-id",
    "it": "it-it",
    "ja": "ja-jp",
    "ko": "ko-kr",
    "nb": "nb-no",
    "no": "nb-no",
    "nl": "nl-nl",
    "pl": "pl-pl",
    "pt": "pt-br",
    "ru": "ru-ru",
    "sv": "sv-se",
    "tr": "tr-tr",
    "zh": "zh-cn",
}


def _resolve_locale(explicit_locale, language) -> str:
    """Resolve a TER locale (``lang-region``) from an explicit override or language.

    An explicit ``locale`` kwarg wins (``zh_cn`` / ``zh-CN`` are normalized to
    ``zh-cn``). Otherwise the locale is derived from the ``language`` name/code
    (e.g. ``Chinese`` -> ``zh-cn``). Falls back to ``en-us`` for unknown or
    ``Unknown`` languages.
    """
    if explicit_locale:
        return str(explicit_locale).replace("_", "-").lower()
    if language and str(language).strip().lower() not in ("", "unknown"):
        code = get_language_code(str(language))
        # Code-switch identifiers like "en_zh" resolve to their first component.
        code = code.split("_")[0]
        locale = _LANG_CODE_TO_LOCALE.get(code)
        if locale:
            return locale
    return "en-us"


def _clean_ref(text: str) -> str:
    if not text:
        return ""
    cleaned = _TIME_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_hyp(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


logger = logging.getLogger(__name__)

SPEECHINSIGHT_VERSION = "2.28.4"
TOOLS_MANIFEST = json.loads(Path(__file__).with_name("speechinsight_tools_manifest.json").read_text())
BUNDLE_REVISION = TOOLS_MANIFEST["bundle_revision"]
TER_VERSION = TOOLS_MANIFEST["ter"]["version"]
DEFAULT_PACK_DIR = "/root/data/packages/SpeechInsight_moe_20261110_ter_2_22_1"
DEFAULT_REMOTE_PACK_ROOT = "az://orngwus2cresco/data/speech/users/boren/eval_packages/inhouse_moe_20261110"
DOTNET_REMOTE_ROOT = "az://orngwus2cresco/data/speech/users/ruchaofan/packages"
TER_WHEEL_NAME = TOOLS_MANIFEST["ter"]["wheel"]
DOTNET_TAR_NAME = "dotnet-runtime-8.0.0-linux-x64.tar.gz"
_SETUP_DONE: set[str] = set()


def _metric_bin(pack_dir: Path) -> Path:
    return pack_dir / "speechinsight_tools_latest" / TOOLS_MANIFEST["get_metrics"]["path"]


def _lib_path(pack_dir: Path) -> Path:
    return _metric_bin(pack_dir).parent


def _dotnet_dir(pack_dir: Path) -> Path:
    return pack_dir / "dotnet"


def ensure_pack_dir(pack_dir: str | Path | None, remote_pack_root: str | None = None) -> Path:
    """Install the pinned MoE 2026-11-10 bundle (SpeechInsight 2.28.4 / TER 2.22.1).

    Node-local setup is locked across reward workers. TER is installed into an
    isolated target so the selected wheel, rather than a system installation,
    supplies dfmetrics. There is only one supported scoring-tool version.
    """
    pack = Path(pack_dir or DEFAULT_PACK_DIR).resolve()
    key = str(pack)
    if key in _SETUP_DONE:
        return pack
    remote_root = (remote_pack_root or DEFAULT_REMOTE_PACK_ROOT).rstrip("/")
    pack.mkdir(parents=True, exist_ok=True)
    si_dir = pack / "speechinsight_tools_latest"
    metric_bin = _metric_bin(pack)
    dotnet_dir = _dotnet_dir(pack)
    target = pack / ("ter_python_" + TER_VERSION.replace(".", "_"))

    def download(*args):
        if not shutil.which("bbb"):
            raise RuntimeError(f"Cannot install SpeechInsight {SPEECHINSIGHT_VERSION}: bbb is unavailable")
        subprocess.run(["bbb", *args], check=True, capture_output=True, text=True)

    with (pack / ".setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ter_whl = si_dir / TER_WHEEL_NAME
        if not metric_bin.is_file() or not ter_whl.is_file():
            download("sync", "-q", "--concurrency", "64", f"{remote_root}/speechinsight_tools_latest/", str(si_dir))
        if not metric_bin.is_file() or not ter_whl.is_file():
            raise RuntimeError(
                f"Missing SpeechInsight {SPEECHINSIGHT_VERSION} binary or TER {TER_VERSION} wheel: {pack}"
            )
        metadata = _lib_path(pack) / "GetMetrics.deps.json"
        libraries = json.loads(metadata.read_text())["libraries"]
        expected = f"Microsoft.Speech.ASR.Metrics.Metrics.NetCore.AnyCPU/{SPEECHINSIGHT_VERSION}.0"
        if expected not in libraries:
            raise RuntimeError(f"Expected SpeechInsight {SPEECHINSIGHT_VERSION} in {metadata}")
        for path, entry, label in (
            (ter_whl, TOOLS_MANIFEST["ter"], "TER wheel"),
            (metric_bin, TOOLS_MANIFEST["get_metrics"], "GetMetrics executable"),
        ):
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != entry["sha256"]:
                raise RuntimeError(f"{label} SHA-256 mismatch: {path} ({actual})")
        if not (dotnet_dir / "dotnet").is_file():
            dotnet_tar = pack / DOTNET_TAR_NAME
            if not dotnet_tar.is_file():
                download("cp", f"{DOTNET_REMOTE_ROOT}/{DOTNET_TAR_NAME}", str(dotnet_tar))
            dotnet_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["tar", "-xzf", str(dotnet_tar), "-C", str(dotnet_dir)], check=True, capture_output=True, text=True
            )
        if not (dotnet_dir / "dotnet").is_file():
            raise RuntimeError(f"Missing dotnet runtime: {dotnet_dir}")
        installed = target / ".installed"
        if not installed.is_file() or installed.read_text().strip() != TOOLS_MANIFEST["ter"]["sha256"]:
            subprocess.run(
                ["uv", "pip", "install", "--no-deps", "--target", str(target), str(ter_whl)],
                check=True,
                capture_output=True,
                text=True,
            )
            installed.write_text(TOOLS_MANIFEST["ter"]["sha256"] + "\n")
        metric_bin.chmod(0o755)

    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    # Reject an older dfmetrics module already loaded by this process.
    import dfmetrics.ter as ter

    if importlib.metadata.version("ter") != TER_VERSION or not Path(ter.__file__).is_relative_to(target):
        raise RuntimeError(f"Expected isolated TER {TER_VERSION} in {target}; loaded {ter.__file__}")
    dotnet_path = str(dotnet_dir)
    if dotnet_path not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{dotnet_path}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["DOTNET_ROOT"] = dotnet_path
    os.environ["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "false"
    lib_path = str(_lib_path(pack))
    if lib_path not in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}"
    _SETUP_DONE.add(key)
    return pack


_TER_BACKENDS: dict[str, object] = {}


def _get_ter_backend(locale: str = "en-us"):
    """Return a cached ``dfmetrics.ter.TER`` configured for DisfluencyTolerant TER.

    Backends are cached per ``locale`` so a single worker can score multiple
    languages. The ``ter`` wheel is installed by ``ensure_pack_dir``; importing
    here lets us bail out cleanly if it's not on the worker.
    """
    loc = (locale or "en-us").lower()
    backend = _TER_BACKENDS.get(loc)
    if backend is None:
        from dfmetrics.ter import TER  # type: ignore

        backend = TER(locale=loc, ter_type="disfluencytolerant")
        _TER_BACKENDS[loc] = backend
    return backend


def _category_edits(ter_category_info: dict | None) -> dict[str, int]:
    """Extract per-category edit counts from ``ter_category_info``.

    Returns a dict with keys ``dter_n_punc``, ``dter_n_cap``, ``dter_n_itn``,
    ``dter_n_lex`` (zero when the category is missing).
    """
    cats = (ter_category_info or {}).get("ter_categories") or {}

    def _n(k: str) -> int:
        return int(((cats.get(k) or {}).get("number_of_edits")) or 0)

    return {
        "dter_n_punc": _n("punc"),
        "dter_n_cap": _n("cap"),
        "dter_n_itn": _n("itn"),
        "dter_n_lex": _n("lexical"),
    }


def _compute_dter(ref: str, hyp: str, locale: str = "en-us") -> tuple[int, int, float, dict | None]:
    """Return ``(n_err, n_ref, dter_fraction, detail)`` for DTER.

    ``detail`` is a single ``UtteranceTERMetrics``-style entry (the same shape
    consumed by the ``inhouse-asr-compare`` skill) carrying the word-level
    alignment, per-word TER classes, and category breakdown.

    GetMetrics CLI emits an empty ``UtteranceTERMetrics`` for single-utterance
    runs, so go through the Python ``dfmetrics`` TER backend directly. The
    backend returns ``{"summary": {...}, "sent_details": [{...}]}`` where the
    per-utterance ``word_align`` / ``word_ter_class`` / ``ter_category_info`` /
    ``display_form_*`` fields live under ``sent_details[0]`` (NOT the top level).
    """
    backend = _get_ter_backend(locale)
    result = backend.compute_ter_from_strings(transcription=ref, recognition=hyp)
    if not isinstance(result, dict):
        raise RuntimeError(f"DTER backend returned invalid result type: {type(result).__name__}")

    summary = result.get("summary") or {}
    info = summary.get("ter_info") or {}
    n_err = int(info.get("number_of_edits") or 0)
    n_ref = int(info.get("number_of_tokens") or 0)
    if ref and n_ref <= 0:
        raise RuntimeError("DTER backend returned zero reference tokens for a nonempty reference")
    # Counts are authoritative, including low nonzero percentages (<1.5%).
    # Inferring units from display_ter's magnitude turns 1% into 100%.
    dter = n_err / n_ref if n_ref else 0.0

    sent_details = result.get("sent_details") or []
    sd = sent_details[0] if sent_details else {}
    detail = {
        "word_align": list(sd.get("word_align") or []),
        "word_ter_class": list(sd.get("word_ter_class") or []),
        "ter_category_info": sd.get("ter_category_info") or summary.get("ter_category_info") or {},
    }
    return n_err, n_ref, dter, detail


def get_metrics(trans: str, reco: str, *args: str, pack_dir: str | Path | None = None, locale=None):
    output_dir = Path(tempfile.gettempdir()) / "get_metrics" / str(uuid.uuid4())
    pack = ensure_pack_dir(pack_dir)
    metric_bin = _metric_bin(pack)
    dotnet_dir = _dotnet_dir(pack)
    lib_path = _lib_path(pack)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DOTNET_ROOT"] = str(dotnet_dir)
    env["PATH"] = f"{dotnet_dir}:{env.get('PATH', '')}"
    env["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "false"

    # Use MoE's TSV contract; full recordings can exceed the OS argv limit.
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv = output_dir / "input.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerow(
            ["recording", _clean_hyp(reco), _clean_ref(trans)]
        )
    cmd = [
        str(metric_bin),
        "-t",
        str(tsv),
        "-o",
        str(output_dir),
        "--idcol",
        "0",
        "--recocol",
        "1",
        "--transcol",
        "2",
    ]
    if locale:
        cmd += ["-l", locale]
    cmd += list(args)

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except Exception as e:
        logger.warning("GetMetrics launch failed: %s", e)
        return None

    result_json = output_dir / "UtteranceResults.json"
    if result.returncode != 0 or not result_json.exists():
        logger.warning(
            "GetMetrics failed (code=%s): %s",
            result.returncode,
            (result.stderr or result.stdout or "")[:400],
        )
        return None

    try:
        results = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("GetMetrics output parse failed: %s", e)
        return None

    if not results:
        return None
    record = results[0] if isinstance(results[0], dict) else None
    if not record:
        return None
    return record


def _compute_eer(
    ref: str, hyp: str, pack_dir: str | Path | None = None, *, strict: bool = False, mode="verbatim", locale=None
) -> tuple[int, int, float]:
    """Return (n_err, n_ref, eer) by calling SpeechInsight GetMetrics.

    EER = (NumTransEnts - NumTransEntsMatched) / NumTransEnts, taken from the
    first ``EntityInfo`` block in ``UtteranceResults.json``. Returns zeros when
    GetMetrics fails or the utterance has no annotated entities.
    """
    if mode not in {"verbatim", "nondisfluency"}:
        raise ValueError(f"Unknown EER normalization: {mode}")
    record = get_metrics(ref, hyp, "--" + mode, pack_dir=pack_dir, locale=locale)
    if not record:
        if strict:
            raise RuntimeError("SpeechInsight EER scoring failed")
        return 0, 0, 0.0
    metric_list = record.get("Metrics") or []
    if not metric_list:
        if strict:
            raise RuntimeError("SpeechInsight returned no EER metrics")
        return 0, 0, 0.0
    entity_info = metric_list[0].get("EntityInfo") or {}
    n_ref = int(entity_info.get("NumTransEnts") or 0)
    if strict and (
        "NumTransEnts" not in entity_info
        or "NumTransEntsMatched" not in entity_info
        or n_ref < 0
        or (n_ref == 0 and re.search(r"<NE(?::[^>]+)?>", ref, re.IGNORECASE))
    ):
        raise RuntimeError("SpeechInsight returned invalid entity denominators")
    if n_ref == 0:
        return 0, 0, 0.0
    n_matched = int(entity_info.get("NumTransEntsMatched") or 0)
    if strict and not 0 <= n_matched <= n_ref:
        raise RuntimeError("SpeechInsight returned invalid matched entity counts")
    n_err = max(n_ref - n_matched, 0)
    return n_err, n_ref, n_err / n_ref


def eval_score(
    solution_str: str | list[str],
    ground_truth: str,
    *,
    pack_dir: str | Path | None = DEFAULT_PACK_DIR,
    speechinsight_package_root: str | None = DEFAULT_REMOTE_PACK_ROOT,
    **kwargs,
):
    """Inhouse DTER/EER scorer for eval_asr.py.

    Reports only DisfluencyTolerant TER (DTER, via dfmetrics Python backend)
    and entity recognition error rate (EER, from SpeechInsight EntityInfo).
    ``dter_p_err`` is the reporting alias used by the 2609 benchmark pipeline.
    ``score = 1 - dter`` is exposed for the trainer's reward aggregation.
    """
    compute_eer = kwargs.get("compute_eer", False)
    extra_info = kwargs.get("extra_info") or {}
    locale = _resolve_locale(kwargs.get("locale"), extra_info.get("language", kwargs.get("language")))
    hyp_text = _clean_hyp(get_hyp_text(solution_str))
    ref_text = _clean_ref(ground_truth)
    # The dfmetrics TER backend used by `_compute_dter` shells out to `dotnet`
    # (via fstalign), so the SpeechInsight pack must be installed and its dotnet
    # runtime exported onto PATH/DOTNET_ROOT *before* DTER runs. `_compute_eer`
    # also relies on this, but it runs after DTER, so set it up up-front here.
    ensure_pack_dir(pack_dir, speechinsight_package_root)

    dter_n_err, dter_n_ref, dter, dter_detail = _compute_dter(ref_text, hyp_text, locale=locale)

    result = {
        "score": 1.0 - dter,
        "dter": dter,
        "dter_p_err": dter,
        "dter_n_err": dter_n_err,
        "dter_n_ref": dter_n_ref,
        "dter_detail": dter_detail,
    }
    result.update(_category_edits((dter_detail or {}).get("ter_category_info")))
    if compute_eer:
        # Display scoring uses ground_truth; transcription scoring uses the
        # original tagged Transcription carried through extra_info.
        reference = kwargs.get("eer_reference", "transcription")
        if reference not in {"transcription", "display"}:
            raise ValueError(f"Unknown EER reference: {reference}")
        entity_ref = _clean_ref(
            ground_truth if reference == "display" else extra_info.get("entity_transcription", ground_truth)
        )
        eer_n_err, eer_n_ref, eer = _compute_eer(
            entity_ref,
            hyp_text,
            pack_dir=pack_dir,
            strict=kwargs.get("strict_eer", False),
            mode=kwargs.get("eer_mode", "verbatim"),
            locale=locale,
        )
        result["eer"] = eer
        result["eer_n_err"] = eer_n_err
        result["eer_n_ref"] = eer_n_ref
    return result
