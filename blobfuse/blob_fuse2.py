#!/usr/bin/env python3
# pylint: disable=subprocess-run-check,raise-missing-from,redefined-outer-name
# %%
import fire
import subprocess
from pathlib import Path
import yaml
from datetime import datetime, timedelta, timezone
from azure.storage.blob import (
    generate_container_sas,
    ContainerSasPermissions,
    BlobServiceClient,
)
from azure.identity import AzureCliCredential


def check_login():
    """Get the Azure CLI Authentication"""
    try:
        subprocess.check_call(
            "az account show",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Please 'az login' firstly.")


def generate_sas_token(blob_client, container, days=7):
    """generate_sas_token."""
    start_time = datetime.now(timezone.utc)
    expiry_time = start_time + timedelta(days=days)

    delegation_key = blob_client.get_user_delegation_key(key_start_time=start_time, key_expiry_time=expiry_time)
    return generate_container_sas(
        account_name=blob_client.account_name,
        container_name=container,
        user_delegation_key=delegation_key,
        permission=ContainerSasPermissions(read=True, write=True, delete=True, list=True),
        expiry=expiry_time,
    )


def mount_container(account, container, sas_token, mount_dir=None):
    """Mount the container to the specified directory using blobfuse2."""
    tmp_dir = Path(f"/mnt/ramdisk/blobfusetmp_boren/{account}_{container}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mount_dir.mkdir(parents=True, exist_ok=True)
    conf_file = mount_dir.parent / f"{account}_{container}.yaml"

    # blobfuse2 uses YAML configuration
    config = {
        "file_cache": {"path": str(tmp_dir)},
        "azstorage": {
            "type": "block",
            "account-name": account,
            "endpoint": f"https://{account}.blob.core.windows.net",
            "sas": sas_token,
            "mode": "sas",
            "container": container,
        },
    }

    with open(conf_file, "w", encoding="utf8") as f:
        yaml.dump(config, f, default_flow_style=False)

    try:
        cmd = " ".join(
            [
                "blobfuse2",
                "mount",
                f"{mount_dir}",
                f"--config-file={conf_file}",
            ]
        )
        print("Cmd: ", cmd)
        subprocess.check_call(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return mount_dir, True
    except subprocess.CalledProcessError as e:
        print(e)
        return mount_dir, False


# Rest of the code remains the same
def is_mounted(mount_point):
    """Check if the mount point is mounted."""
    mount_point = str(Path(mount_point).resolve())
    with open("/proc/mounts", "r", encoding="utf8") as f:
        mounts = [line.strip() for line in f]

    for mount in mounts:
        if mount_point in mount:
            return True

    return False


def mount_containers(account_dict, days=7, work_dir=None):
    """Mount the container to the specified directory."""
    credential = AzureCliCredential()
    work_dir = Path(work_dir or Path(__file__).parent).resolve()
    print("Work dir:", work_dir)

    for account, containers in account_dict.items():
        account_url = f"https://{account}.blob.core.windows.net"
        blob_client = BlobServiceClient(account_url, credential=credential)
        for container in containers:
            sas_token = generate_sas_token(blob_client, container, days)
            mount_dir = work_dir / f"{account}_{container}"
            if is_mounted(mount_dir):
                print(f"Already mounted: {mount_dir}")
                continue
            status = mount_container(account, container, sas_token, mount_dir)
            if status:
                print(f"Mounted: {mount_dir}")
            else:
                print(f"Failed mount: {account}/{container}")


def main(accounts, containers, work_dir=None):
    """Mount the containers."""
    account_dict = {}
    print("Accounts:", accounts)
    print("Containers:", containers)
    accounts = accounts.split("@")
    containers = containers.split("@")
    for account in accounts:
        if isinstance(containers, str):
            containers = [containers]
        account_dict[account] = containers
    print(f"Mounting\n{account_dict}")
    mount_containers(account_dict, work_dir=work_dir)


if __name__ == "__main__":
    fire.Fire(main)
