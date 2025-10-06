#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import wandb
import fire
from more_itertools import unique_everseen
import os
import urllib.parse
from pathlib import Path


def get_run(path):
    parsed = urllib.parse.urlparse(path)
    path = parsed.path.replace("/runs/", "/").strip("/")
    assert path is not None, "Either url or path must be provided to get_run"
    try:
        return wandb.Api().run(path)
    except wandb.Error:
        return None


def to_list(x):
    if x is None:
        return []
    elif isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get_run_result(runs, prefix="metric", name=None):
    results = []
    name_pfx = to_list(name)
    for run in runs:
        run_dict = {"name": run.name}
        for key, value in run.summary.items():
            if key and key.startswith(prefix):
                new_key = key.split("/", 1)[-1]  # Remove prefix
                if name_pfx and not any(new_key.startswith(pfx) for pfx in name_pfx):
                    continue
                run_dict[new_key] = value
        results.append(run_dict)
    df = pd.DataFrame(results).set_index("name").T
    df.dropna(axis=1, how="all", inplace=True)  # Drop columns with all NaN values
    if df.empty:
        return None
    df[["dataset", "#bias", "metric"]] = df.index.to_series().str.rsplit("_", n=2, expand=True)
    df["#bias"] = pd.to_numeric(df["#bias"], errors="coerce").fillna(0).astype(int)  # Convert to int
    df = df[df["metric"].isin(["WER", "BWER", "UWER"])]
    df = df.sort_values(by=["dataset", "#bias", "metric"], ascending=[True, True, False])
    df = df.drop(columns=["dataset", "#bias", "metric"])
    return df


class WandbChecker:
    def __init__(self, entity=None, project=None, metric="metric", dataset=None, excel_dir=None):
        self.host = os.environ.get("WANDB_ORGANIZATION", "https://msaip.wandb.io")
        self.entity = entity or os.environ.get("WANDB_ENTITY", "genai")
        self.project = project or os.environ.get("WANDB_PROJECT", "verl_asr")
        key = os.environ.get("WANDB_API_KEY", "")
        print(f"Using W&B : {self.host}/{self.entity}/{self.project}")
        wandb.login(host=self.host, key=key, relogin=True)

        self.metric = metric
        self.dataset = dataset.split(",") if isinstance(dataset, str) else dataset
        self.excel_dir = Path(excel_dir) if excel_dir else Path.home() / "wandb_results"

    def check(self, run_url, key=None, nrows=10):
        run = get_run(run_url)
        if run is None:
            print(f"Run not found: {run_url}")
            return None
        df = get_run_result([run], prefix=self.metric, name=self.dataset)
        if df is None:
            print(f"No results [{self.metric}] found for run: {run_url}")
            return None
        self._to_excel(df, name=run.name)
        if key:
            df = df[df.index.str.contains(key)]
        df = df.head(nrows)
        print(df)

    def _to_excel(self, df, name=None):
        """Save DataFrame to Excel file."""
        self.excel_dir.mkdir(parents=True, exist_ok=True)
        datestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        name = name or "default"
        name = name.replace(" ", "_").replace("/", "_").replace("|", "_")
        name = "_".join(unique_everseen(name.split("_")))[:100]
        excel_path = self.excel_dir / f"{self.metric}_{datestamp}_{name}.xlsx"
        print(f"Writing {df.shape} results to {excel_path}")
        df.to_excel(excel_path, index=True)

    def search(self, run_name, key=None, nrows=10):
        """search runs"""
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}", filters={"display_name": {"$regex": run_name}})
        if not runs:
            print(f"No runs found matching '{run_name}'")
            return
        print(f"Found {len(runs)} runs matching '{run_name}'")
        df = get_run_result(runs, prefix=self.metric, name=self.dataset)
        if df is None:
            print(f"No results [{self.metric}] found for runs matching '{run_name}'")
            return None
        self._to_excel(df, name=run_name)
        if key:
            df = df[df.index.str.contains(key)]
        df = df.head(nrows)
        print(df)


if __name__ == "__main__":
    fire.Fire(WandbChecker)
