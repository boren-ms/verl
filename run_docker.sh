
# Docker image
# reference: https://verl.readthedocs.io/en/latest/start/install.html
image=verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2
name=boren_verl
# podman pull $image
# podman create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v /home/boren:/home/boren -v /datablob1:/datablob1 --name ${name} ${image} sleep infinity
# podman start ${name}
# podman exec -it ${name} bash
#
podman stop ${name} && podman rm ${name}
podman run --rm -d --gpus all --ipc host -v /datablob1:/datablob1 -v  /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name ${name} ${image} tail -f /dev/null
podman exec -it ${name} bash

# git clone https://github.com/volcengine/verl && cd verl
# pip3 install -e .[vllm]
# # pip3 install -e .[sglang]