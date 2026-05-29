from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

from recipe.phimm.utils.shared import parse_asr_response

# Strip "[start end]" timing tokens emitted anywhere in DisplayTranscription.
_TIME_MARKER_RE = re.compile(r"\[\s*-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s*\]")


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


@dataclass
class Error:
    n_sub: int = 0
    n_del: int = 0
    n_ins: int = 0
    n_ref: int = 0

    @property
    def n_err(self) -> int:
        return self.n_sub + self.n_del + self.n_ins

    @property
    def wer(self) -> float:
        if self.n_ref == 0:
            return 0.0
        return self.n_err / self.n_ref


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


_TER_BACKEND = None


def _get_ter_backend(locale: str = "en-us"):
    """Return a cached ``dfmetrics.ter.TER`` configured for DisfluencyTolerant TER.

    The ``ter`` wheel is installed by ``ensure_pack_dir``; importing here lets us
    bail out cleanly if it's not on the worker.
    """
    global _TER_BACKEND
    if _TER_BACKEND is None:
        from dfmetrics.ter import TER  # type: ignore
        _TER_BACKEND = TER(locale=locale, ter_type="disfluencytolerant")
    return _TER_BACKEND


def _compute_dter(ref: str, hyp: str) -> tuple[int, int, float]:
    """Return (n_err, n_ref, dter_fraction) for DisfluencyTolerant_TER.

    GetMetrics CLI emits an empty ``UtteranceTERMetrics`` for single-utterance
    runs, so go through the Python ``dfmetrics`` TER backend directly.
    """
    try:
        backend = _get_ter_backend()
        result = backend.compute_ter_from_strings(transcription=ref, recognition=hyp) or {}
    except Exception as e:
        logger.warning("DTER computation failed: %s", e)
        return 0, 0, 0.0

    info = (result.get("summary") or {}).get("ter_info") or {}
    n_err = int(info.get("number_of_edits") or 0)
    n_ref = int(info.get("number_of_tokens") or 0)
    raw = float(info.get("display_ter") or 0.0)
    # dfmetrics returns display_ter as a percentage (e.g. 42.85), normalize to fraction.
    dter = raw / 100.0 if raw > 1.5 else raw
    if n_ref > 0 and (n_err > 0 and dter == 0.0):
        dter = n_err / n_ref
    return n_err, n_ref, dter


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


def _extract_dter(record: dict) -> tuple[int, int, float]:
    """Return (n_err, n_ref, ter) for DisfluencyTolerant_TER if emitted by CLI.

    GetMetrics single-utterance runs usually leave ``UtteranceTERMetrics`` empty;
    prefer ``_compute_dter`` which calls the dfmetrics TER backend directly.
    """
    for metric in record.get("UtteranceTERMetrics") or []:
        if metric.get("MetricName") != "DisfluencyTolerant_TER":
            continue
        info = metric.get("ter_info") or {}
        n_err = int(info.get("number_of_edits") or 0)
        n_ref = int(info.get("number_of_tokens") or 0)
        raw = float(info.get("display_ter") or 0.0)
        ter = raw / 100.0 if raw > 1.5 else raw
        if n_ref > 0 and (n_err > 0 and ter == 0.0):
            ter = n_err / n_ref
        return n_err, n_ref, ter
    return 0, 0, 0.0


def _pick_int(d: dict, keys: list[str]) -> int | None:
    for k in keys:
        if k in d and d.get(k) is not None:
            try:
                return int(d.get(k))
            except (TypeError, ValueError):
                continue
    return None


def _extract_eer(entity_info: dict | None) -> tuple[int, int, float, int]:
    """Return (n_err, n_ref, eer, fallback_from_word_level).

    Primary path: SpeechInsight's ``EntityInfo`` exposes ``NumTransEnts`` (total
    entities in the reference) and ``NumTransEntsMatched`` (correctly recognized
    entities). EER = (NumTransEnts - NumTransEntsMatched) / NumTransEnts.

    Falls back to entity-word-level errors if entity-level counts are missing.
    """
    if not entity_info:
        return 0, 0, 0.0, 0

    n_ref = int(entity_info.get("NumTransEnts") or 0)
    if n_ref > 0:
        n_matched = int(entity_info.get("NumTransEntsMatched") or 0)
        n_err = max(n_ref - n_matched, 0)
        return n_err, n_ref, n_err / n_ref, 0

    # Fallback: word-level entity errors (only used if utterance had no entities,
    # in which case both counts are 0 anyway).
    w_sub = int(entity_info.get("NumEntWordSub") or 0)
    w_del = int(entity_info.get("NumEntWordDel") or 0)
    w_ins = int(entity_info.get("NumEntWordIns") or 0)
    w_ref = int(entity_info.get("NumEntWords") or 0)
    w_err = w_sub + w_del + w_ins
    if w_ref > 0:
        return w_err, w_ref, w_err / w_ref, 1
    return 0, 0, 0.0, 0


def get_wers(trans: str, reco: str, *args: str, pack_dir: str | Path | None = None):
    metrics = get_metrics(trans, reco, *args, pack_dir=pack_dir)
    if metrics is None:
        return None, None
    metric_list = metrics.get("Metrics", []) or []
    if not metric_list:
        return None, None
    first_metric = metric_list[0]
    ewer = Error(
        n_sub=first_metric["EntityInfo"]["NumEntWordSub"],
        n_del=first_metric["EntityInfo"]["NumEntWordDel"],
        n_ins=first_metric["EntityInfo"]["NumEntWordIns"],
        n_ref=first_metric["EntityInfo"]["NumEntWords"],
    )
    wer = Error(
        n_sub=first_metric["WERInfo"]["Substitutions"],
        n_del=first_metric["WERInfo"]["Deletions"],
        n_ins=first_metric["WERInfo"]["Insertions"],
        n_ref=first_metric["WERInfo"]["TXWords"],
    )
    return wer, ewer


def eval_score(solution_str: str, ground_truth: str, **kwargs):
    """Inhouse TER/EWER scorer for eval_asr.py.

    Returns keys compatible with the eval aggregator:
    score, n_err, n_ref, nb_err, nb_ref.
    """
    pack_dir = kwargs.get("pack_dir", DEFAULT_PACK_DIR)
    parsed = parse_asr_response(solution_str)
    hyp_text = parsed.get("text") or ""
    ref_text = _clean_ref(ground_truth)

    # --verbatim: tran/reco both display form; tran may have <disfluency> tags
    # and entity annotations. Produces WERInfo + EntityInfo + UtteranceTERMetrics
    # (including DisfluencyTolerant_TER) in UtteranceResults.json.
    record = get_metrics(ref_text, hyp_text, "--verbatim", pack_dir=pack_dir)
    if record is None:
        return {
            "score": 0.0,
            "n_err": 0,
            "n_ref": 0,
            "nb_err": 0,
            "nb_ref": 0,
            "dter": 0.0,
            "dter_n_err": 0,
            "dter_n_ref": 0,
            "eer": 0.0,
            "ne_err": 0,
            "ne_ref": 0,
            "metric_error": 1,
            "eer_from_word_level": 0,
        }

    metric_list = record.get("Metrics", []) or []
    if not metric_list:
        return {
            "score": 0.0,
            "n_err": 0,
            "n_ref": 0,
            "nb_err": 0,
            "nb_ref": 0,
            "dter": 0.0,
            "dter_n_err": 0,
            "dter_n_ref": 0,
            "eer": 0.0,
            "ne_err": 0,
            "ne_ref": 0,
            "metric_error": 1,
            "eer_from_word_level": 0,
        }

    first_metric = metric_list[0]
    wer = Error(
        n_sub=first_metric["WERInfo"]["Substitutions"],
        n_del=first_metric["WERInfo"]["Deletions"],
        n_ins=first_metric["WERInfo"]["Insertions"],
        n_ref=first_metric["WERInfo"]["TXWords"],
    )
    ewer = Error(
        n_sub=first_metric["EntityInfo"]["NumEntWordSub"],
        n_del=first_metric["EntityInfo"]["NumEntWordDel"],
        n_ins=first_metric["EntityInfo"]["NumEntWordIns"],
        n_ref=first_metric["EntityInfo"]["NumEntWords"],
    )

    # Try the CLI-emitted UtteranceTERMetrics first (rare for single-utterance
    # runs); fall back to the dfmetrics Python TER backend.
    dter_n_err, dter_n_ref, dter = _extract_dter(record)
    if dter_n_ref == 0:
        dter_n_err, dter_n_ref, dter = _compute_dter(ref_text, hyp_text)

    ne_err, ne_ref, eer, eer_from_word_level = _extract_eer(first_metric.get("EntityInfo") or {})

    return {
        "score": 1.0 - wer.wer,
        "n_err": wer.n_err,
        "n_ref": wer.n_ref,
        "nb_err": ewer.n_err,
        "nb_ref": ewer.n_ref,
        "dter": dter,
        "dter_n_err": dter_n_err,
        "dter_n_ref": dter_n_ref,
        "eer": eer,
        "ne_err": ne_err,
        "ne_ref": ne_ref,
        "eer_from_word_level": eer_from_word_level,
    }
