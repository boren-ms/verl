
# Docker image
set -x
# reference: https://verl.readthedocs.io/en/latest/start/install.html
# image=verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2
# image=verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4
image=verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
name=boren_verl
# docker pull $image
# docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v /home/boren:/home/boren -v /datablob1:/datablob1 --name ${name} ${image} sleep infinity
# docker start ${name}
# docker exec -it ${name} bash
#
docker stop ${name} && docker rm ${name}
docker run  --rm -d --gpus all --ipc host  -v /home/boren/data:/home/boren/data -v /home/boren/verl:/home/boren/verl --name ${name} ${image} tail -f /dev/null
docker exec -it ${name} bash

# git clone https://github.com/volcengine/verl && cd verl
# pip3 install -e .[vllm]
# # pip3 install -e .[sglang]