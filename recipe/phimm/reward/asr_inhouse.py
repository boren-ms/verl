# %%
from dataclasses import dataclass
import os
import subprocess
from pathlib import Path
import tempfile
import json
import uuid


@dataclass
class Error:
    n_sub: int = 0  # substitution
    n_del: int = 0  # deletion
    n_ins: int = 0  # insertion
    n_ref: int = 0  # total reference words

    @property
    def n_hit(self):
        return self.n_ref - self.n_sub - self.n_del

    @property
    def n_err(self):
        return self.n_sub + self.n_del + self.n_ins

    @property
    def accuracy(self):
        if self.n_ref == 0:
            return 0.0
        return self.n_hit / self.n_ref

    @property
    def wer(self):
        if self.n_ref == 0:
            return 0.0
        return self.n_err / self.n_ref


def get_metrics(trans, reco, *args, pack_dir=None):
    output_dir = Path(tempfile.gettempdir()) / "get_metrics" / str(uuid.uuid4())
    pack_dir = pack_dir or "/home/boren/data/packages/SpeechInsight"
    pack_dir = Path(pack_dir)
    metric_bin = pack_dir / "speechinsight_tools" / "linux-x64" / "framework-dependent" / "GetMetrics"
    dotnet_dir = pack_dir / "dotnet"
    lib_path = pack_dir / "speechinsight_tools" / "linux-x64" / "framework-dependent"

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DOTNET_ROOT"] = str(dotnet_dir)
    env["PATH"] = f"{dotnet_dir}:{env.get('PATH', '')}"

    cmd = [str(metric_bin), "-t", trans, "-r", reco, "-o", str(output_dir)]
    cmd += list(args)
    # print("CMD:", cmd)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    result_json = output_dir / "UtteranceResults.json"

    if result.returncode != 0 or not result_json.exists():
        return None
    results = json.loads(result_json.read_text())
    # Clean up output_dir after reading results
    # shutil.rmtree(output_dir)
    if not results:
        return None
    metrics = results[0].get("Metrics", [])
    if len(metrics) == 0:
        return None
    return metrics[0]


def get_wer(trans, reco, *args, pack_dir=None):
    metrics = get_metrics(trans, reco, *args, pack_dir=pack_dir)

    wer_info = metrics["WERInfo"]
    return Error(
        n_sub=wer_info["Substitutions"],
        n_del=wer_info["Deletions"],
        n_ins=wer_info["Insertions"],
        n_ref=wer_info["TXWords"],
    )


def get_dwer(trans, ref, *args, pack_dir=None):
    args = list(set(list(args) + ["-d"]))
    return get_wer(trans, ref, *args, pack_dir=pack_dir)


def get_ewer(trans, reco, *args, pack_dir=None):
    metrics = get_metrics(trans, reco, *args, pack_dir=pack_dir)
    ewer_info = metrics["EntityInfo"]
    return Error(
        n_sub=ewer_info["NumEntWordSub"],
        n_del=ewer_info["NumEntWordDel"],
        n_ins=ewer_info["NumEntWordIns"],
        n_ref=ewer_info["NumEntWords"],
    )


def get_wers(trans, reco, *args, pack_dir=None):
    metrics = get_metrics(trans, reco, *args, pack_dir=pack_dir)
    ewer = Error(
        n_sub=metrics["EntityInfo"]["NumEntWordSub"],
        n_del=metrics["EntityInfo"]["NumEntWordDel"],
        n_ins=metrics["EntityInfo"]["NumEntWordIns"],
        n_ref=metrics["EntityInfo"]["NumEntWords"],
    )
    wer = Error(
        n_sub=metrics["WERInfo"]["Substitutions"],
        n_del=metrics["WERInfo"]["Deletions"],
        n_ins=metrics["WERInfo"]["Insertions"],
        n_ref=metrics["WERInfo"]["TXWords"],
    )
    return wer, ewer


def compute_score(solution_str, ground_truth, **kwargs):
    """The scoring function for ASR with keywords."""
    pack_dir = kwargs.get("pack_dir", None)
    wer, ewer = get_wers(ground_truth, solution_str, "-d", pack_dir=pack_dir)
    return {
        "score": wer.n_err,
        "n_err": wer.n_err,
        "n_ref": wer.n_ref,
        "nb_err": ewer.n_err,
        "nb_ref": ewer.n_ref,
    }


def eval_score(solution_str, ground_truth, **kwargs):
    """The scoring function for ASR with keywords."""
    wer, ewer = get_wers(ground_truth, solution_str, "-d")
    return {
        "score": wer.n_err,
        "n_err": wer.n_err,
        "n_ref": wer.n_ref,
        "nb_err": ewer.n_err,
        "nb_ref": ewer.n_ref,
    }


# %%
# print(get_wers("Hello <NE>world</NE> 123", "hello world one two three"))

# %%
