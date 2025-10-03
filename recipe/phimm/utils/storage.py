# %%

import os
import urllib
import fsspec
from cachetools import cached, FIFOCache


def get_region():
    """Get the region of the Kubernetes cluster from the environment variable."""
    rcall_kube_cluster = os.environ.get("RCALL_KUBE_CLUSTER", "")
    cluster_region = rcall_kube_cluster.split("-")[1] if "-" in rcall_kube_cluster else None
    return cluster_region


def storage_account(region=None):
    storage_dict = {
        "southcentralus": "orngscuscresco",
        "westus2": "orngwus2cresco",
        "uksouth": "orngcresco",
    }
    region = region or get_region()
    if region is None:
        region = "uksouth"
        print("No region is found, setting to [uksouth]")
    return storage_dict[region]


@cached(cache=FIFOCache(maxsize=100))
def azure_storage_options(account=None):
    cluster_account = storage_account()
    if account != cluster_account:
        print(f"Warning: Using cluster account [{cluster_account}], rather than provided account [{account}]")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    if client_id is None or client_secret is None or tenant_id is None:
        raise ValueError("AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID must be set in environment variables.")
    return {
        "account_name": cluster_account,
        "client_id": client_id,
        "client_secret": client_secret,
        "tenant_id": tenant_id,
    }


@cached(cache=FIFOCache(maxsize=100))
def azure_fs(account=None):
    options = azure_storage_options(account=account)
    return fsspec.filesystem("az", **options)


def get_fs_path(path, account=None):
    url = urllib.parse.urlparse(path)
    if url.scheme != "az":
        return fsspec.filesystem("file"), path
    account = account or url.netloc
    fs = azure_fs(account=account)
    return fs, url.path


def get_path_with_options(path, account=None):
    url = urllib.parse.urlparse(path)
    if url.scheme != "az":
        return path, None
    account = account or url.netloc
    options = azure_storage_options(account=account)
    fs_path = f"{url.scheme}://{url.path.lstrip('/')}"
    return fs_path, options


# %%
if __name__ == "__main__":
    az_file = "az://orngwus2cresco/data/speech/am_data/en/human_caption/Alignment/FY23Q2_AdjustBoundary_BiasLM_alignment/file_set_with_hc_v2_verbatim_topic.json"
    az_file = "~/data/"
    fs, path = get_fs_path(az_file)
    print(fs.exists(path))
    print(fs.ls(path))
    # %%
    ds_path = "az://orngwus2cresco/data/boren/data/cache_datasets/hcv2_sc3k_fn1/"

    from datasets import load_from_disk

    relpath, options = get_path_with_options(ds_path)
    ds = load_from_disk(relpath, storage_options=options)
