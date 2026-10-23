# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=iridiumsdc.azurecr.io/rcall:crow-1676263
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-lc"]
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
COPY source/ /opt/verl-source/
WORKDIR /opt/verl-source
# Staged on orange-builder; mounted during build so wheels and credentials do
# not inflate the resulting image. Preserve the base entrypoint and Ray stack.
RUN --mount=type=bind,source=packages,target=/root/packages,rw \
    ray_before=$(python -c 'import ray; print(ray.__version__)') && \
    VERL_STAGED_PACKAGES=1 bash quick_install.sh --image-build && \
    test "$ray_before" = "$(python -c 'import ray; print(ray.__version__)')" && \
    mkdir -p /opt/verl-env && \
    python -m pip freeze > /opt/verl-env/requirements.freeze.txt && \
    cp /opt/verl-source/quick_install*.sh /opt/verl-env/ && \
    rm -rf /root/.cache/pip
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/root/.pyenv/versions/3.12.9/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}
