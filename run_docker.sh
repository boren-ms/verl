
# Docker image
# reference: https://verl.readthedocs.io/en/latest/start/install.html
image=verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2

# podman pull $image
podman create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v /home/boren:/home/boren -v /datablob1:/datablob1 --name verl $image sleep infinity
podman start verl
podman exec -it verl bash
#
# git clone https://github.com/volcengine/verl && cd verl
# pip3 install -e .[vllm]
# # pip3 install -e .[sglang]