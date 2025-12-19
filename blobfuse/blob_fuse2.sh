#!/bin/bash

# Array of storage accounts
accounts=(
    tsstd01uks
    tsstd01wus2
    tsstd01safn
    # zettaprod01eus2
    # highperf01eus
)

# Array of container names
containers=(
    data
)

# Base directories

# Set USER based on environment
if [ "$USER" == "codespace" ]; then
    BLOBFUSE_BASE="/workspaces/MoE/blobfuse"
    mnt_dir="/workspaces/blobfuse"
else
    BLOBFUSE_BASE="/home/${USER}/blobfuse"
    mnt_dir="/home/${USER}/blobfuse"
fi

echo "BLOBFUSE_BASE: ${BLOBFUSE_BASE}"
RAMDISK_BASE="/mnt/ramdisk/blobfusetmp_boren"

# Function to unmount and cleanup a mount point
cleanup_mount() {
    local mount_point="$1"
    local account_container="$2"
    local temp_dir="$RAMDISK_BASE/${account_container}"
    
    if [ -d "$mount_point" ]; then
        echo "Unmounting $mount_point..."
        
        # Kill any existing blobfuse processes for this mount
        pkill -f "blobfuse.*${account_container}"
        
        # Unmount with force (-z) option
        fusermount -uz "$mount_point" || echo "Failed to unmount $mount_point"
    fi
    if [ -d "$temp_dir" ]; then
        # Remove temporary directory
        echo "Removing temporary directory: $temp_dir"
        rm -fr "$temp_dir"
    fi
}

# Main loop
for account in "${accounts[@]}"; do
    for container in "${containers[@]}"; do
        mount_point="$BLOBFUSE_BASE/${account}_${container}"
        cleanup_mount "$mount_point" "${account}_${container}"
    done
done

# Prepare arguments for Python script
accounts_str=$(
    IFS=@
    echo "${accounts[*]}"
)
containers_str=$(
    IFS=@
    echo "${containers[*]}"
)

# Execute Python script
echo "Executing: python blob_fuse2.py $accounts_str $containers_str" --work_dir $mnt_dir
python blob_fuse2.py "$accounts_str" "$containers_str" --work_dir $mnt_dir
