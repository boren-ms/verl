from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

from recipe.phimm.utils.languages import get_language_code
from recipe.phimm.utils.shared import parse_asr_response

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

logger = logging.getLogger(__name__)

DEFAULT_PACK_DIR = "/root/data/packages/SpeechInsight"
DEFAULT_REMOTE_PACK_ROOT = "az://orngwus2cresco/data/speech/users/ruchaofan/packages"
TER_WHEEL_NAME = "ter-2.10.0-py3-none-any.whl"
DOTNET_TAR_NAME = "dotnet-runtime-8.0.0-linux-x64.tar.gz"
_SETUP_DONE: set[str] = set()


def _metric_bin(pack_dir: Path) -> Path:
    return pack_dir / "speechinsight_tools" / "linux-x64" / "framework-dependent" / "GetMetrics"


def _lib_path(pack_dir: Path) -> Path:
    return pack_dir / "speechinsight_tools" / "linux-x64" / "framework-dependent"


def _dotnet_dir(pack_dir: Path) -> Path:
    return pack_dir / "dotnet"


def ensure_pack_dir(pack_dir: str | Path | None, remote_pack_root: str | None = None) -> Path:
    """Ensure SpeechInsight runtime + dotnet + ter wheel are installed.

    Pulls from ruchaofan's package mirror on blob (the same files the MoE
    eval script uses). Best effort: on failure we log and let scoring fall
    back to zero metrics rather than crashing the eval.
    """
    pack = Path(pack_dir or DEFAULT_PACK_DIR)
    key = str(pack)
    if key in _SETUP_DONE:
        return pack

    remote_root = (remote_pack_root or DEFAULT_REMOTE_PACK_ROOT).rstrip("/")
    has_bbb = bool(shutil.which("bbb"))
    pack.mkdir(parents=True, exist_ok=True)

    si_dir = pack / "speechinsight_tools"
    metric_bin = _metric_bin(pack)
    if not metric_bin.exists():
        if has_bbb:
            try:
                logger.info("Syncing SpeechInsight tools from %s/speechinsight_tools/", remote_root)
                subprocess.run(
                    [
                        "bbb", "sync", "-q", "--concurrency", "64",
                        f"{remote_root}/speechinsight_tools/", str(si_dir),
                    ],
                    check=False, capture_output=True, text=True,
                )
            except Exception as e:
                logger.warning("SpeechInsight sync failed: %s", e)
        else:
            logger.warning("Cannot sync SpeechInsight: `bbb` not found in PATH")

    dotnet_tar = pack / DOTNET_TAR_NAME
    dotnet_dir = _dotnet_dir(pack)
    dotnet_marker = dotnet_dir / "dotnet"
    if not dotnet_marker.exists():
        if not dotnet_tar.exists() and has_bbb:
            try:
                subprocess.run(
                    [
                        "bbb", "cp",
                        f"{remote_root}/{DOTNET_TAR_NAME}", str(dotnet_tar),
                    ],
                    check=False, capture_output=True, text=True,
                )
            except Exception as e:
                logger.warning("dotnet runtime download failed: %s", e)
        if dotnet_tar.exists():
            try:
                dotnet_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["tar", "-xzf", str(dotnet_tar), "-C", str(dotnet_dir)],
                    check=False, capture_output=True, text=True,
                )
            except Exception as e:
                logger.warning("dotnet runtime extract failed: %s", e)

    ter_whl = pack / TER_WHEEL_NAME
    if not ter_whl.exists() and has_bbb:
        try:
            subprocess.run(
                [
                    "bbb", "cp",
                    f"{remote_root}/{TER_WHEEL_NAME}", str(ter_whl),
                ],
                check=False, capture_output=True, text=True,
            )
        except Exception as e:
            logger.warning("ter wheel download failed: %s", e)
    marker = pack / ".ter_installed"
    if ter_whl.exists() and not marker.exists():
        try:
            subprocess.run(
                ["pip", "install", "--no-deps", str(ter_whl)],
                check=False, capture_output=True, text=True,
            )
            marker.write_text("ok\n", encoding="utf-8")
        except Exception as e:
            logger.warning("ter wheel install failed: %s", e)

    try:
        metric_bin.chmod(0o755)
    except Exception:
        pass

    # Export to process env so the dfmetrics TER python API (which internally
    # shells out to fstalign and needs `dotnet` on PATH) works without each
    # caller having to set it.
    dotnet_path = str(_dotnet_dir(pack))
    if dotnet_path not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{dotnet_path}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("DOTNET_ROOT", dotnet_path)
    os.environ.setdefault("DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "false")
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
    cats = ((ter_category_info or {}).get("ter_categories") or {})
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
    alignment, per-word TER classes, and category breakdown. It is ``None`` when
    the backend fails.

    GetMetrics CLI emits an empty ``UtteranceTERMetrics`` for single-utterance
    runs, so go through the Python ``dfmetrics`` TER backend directly. The
    backend returns ``{"summary": {...}, "sent_details": [{...}]}`` where the
    per-utterance ``word_align`` / ``word_ter_class`` / ``ter_category_info`` /
    ``display_form_*`` fields live under ``sent_details[0]`` (NOT the top level).
    """
    try:
        backend = _get_ter_backend(locale)
        result = backend.compute_ter_from_strings(transcription=ref, recognition=hyp) or {}
    except Exception as e:
        logger.warning("DTER computation failed: %s", e)
        return 0, 0, 0.0, None

    summary = result.get("summary") or {}
    info = summary.get("ter_info") or {}
    n_err = int(info.get("number_of_edits") or 0)
    n_ref = int(info.get("number_of_tokens") or 0)
    raw = float(info.get("display_ter") or 0.0)
    # dfmetrics returns display_ter as a percentage (e.g. 42.85), normalize to fraction.
    dter = raw / 100.0 if raw > 1.5 else raw
    if n_ref > 0 and (n_err > 0 and dter == 0.0):
        dter = n_err / n_ref

    sent_details = result.get("sent_details") or []
    sd = sent_details[0] if sent_details else {}
    detail = {
        "word_align": list(sd.get("word_align") or []),
        "word_ter_class": list(sd.get("word_ter_class") or []),
        "ter_category_info": sd.get("ter_category_info") or summary.get("ter_category_info") or {},
    }
    return n_err, n_ref, dter, detail


def get_metrics(trans: str, reco: str, *args: str, pack_dir: str | Path | None = None):
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

    cmd = [str(metric_bin), "-t", trans, "-r", reco, "-o", str(output_dir)]
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


def _compute_eer(ref: str, hyp: str, pack_dir: str | Path | None = None) -> tuple[int, int, float]:
    """Return (n_err, n_ref, eer) by calling SpeechInsight GetMetrics.

    EER = (NumTransEnts - NumTransEntsMatched) / NumTransEnts, taken from the
    first ``EntityInfo`` block in ``UtteranceResults.json``. Returns zeros when
    GetMetrics fails or the utterance has no annotated entities.
    """
    record = get_metrics(ref, hyp, "--verbatim", pack_dir=pack_dir)
    if not record:
        return 0, 0, 0.0
    metric_list = record.get("Metrics") or []
    if not metric_list:
        return 0, 0, 0.0
    entity_info = metric_list[0].get("EntityInfo") or {}
    n_ref = int(entity_info.get("NumTransEnts") or 0)
    if n_ref == 0:
        return 0, 0, 0.0
    n_matched = int(entity_info.get("NumTransEntsMatched") or 0)
    n_err = max(n_ref - n_matched, 0)
    return n_err, n_ref, n_err / n_ref


def eval_score(solution_str: str, ground_truth: str, **kwargs):
    """Inhouse DTER/EER scorer for eval_asr.py.

    Reports only DisfluencyTolerant TER (DTER, via dfmetrics Python backend)
    and entity recognition error rate (EER, from SpeechInsight EntityInfo).
    ``score = 1 - dter`` is exposed for the trainer's reward aggregation.
    """
    pack_dir = kwargs.get("pack_dir", DEFAULT_PACK_DIR)
    compute_eer = kwargs.get("compute_eer", False)
    extra_info = kwargs.get("extra_info") or {}
    locale = _resolve_locale(kwargs.get("locale"), extra_info.get("language", kwargs.get("language")))
    # The dfmetrics TER backend used by `_compute_dter` shells out to `dotnet`
    # (via fstalign), so the SpeechInsight pack must be installed and its dotnet
    # runtime exported onto PATH/DOTNET_ROOT *before* DTER runs. `_compute_eer`
    # also relies on this, but it runs after DTER, so set it up up-front here.
    ensure_pack_dir(pack_dir)
    parsed = parse_asr_response(solution_str)
    hyp_text = parsed.get("text") or ""
    ref_text = _clean_ref(ground_truth)

    dter_n_err, dter_n_ref, dter, dter_detail = _compute_dter(ref_text, hyp_text, locale=locale)

    result = {
        "score": 1.0 - dter,
        "dter": dter,
        "dter_n_err": dter_n_err,
        "dter_n_ref": dter_n_ref,
        "dter_detail": dter_detail,
    }
    result.update(_category_edits((dter_detail or {}).get("ter_category_info")))
    if compute_eer:
        eer_n_err, eer_n_ref, eer = _compute_eer(ref_text, hyp_text, pack_dir=pack_dir)
        result["eer"] = eer
        result["eer_n_err"] = eer_n_err
        result["eer_n_ref"] = eer_n_ref
    return result
