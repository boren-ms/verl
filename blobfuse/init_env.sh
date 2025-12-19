#!/bin/bash
sudo apt-get update
sudo apt-get install blobfuse2 # Install blobfuse2
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash  # Install Azure CLI

sudo mkdir /mnt/ramdisk/
sudo chown codespace:codespace /mnt/ramdisk/

ln -s /mnt/ramdisk /home/codespace/ramdisk

pushd /workspaces/MoE/blobfuse
bash ./blobfuse2.sh
ln -s /workspaces/blobfuse/tsstd01uks_data /datablob1
# ln -s /mnt2/newhome/boren/blobfuse/tsstd01wus2_data /datablob
popd