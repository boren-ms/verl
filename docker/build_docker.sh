#!/bin/bash
# Build and push the docker image to Azure Container Registry

# az login # the subscription:  Acoustic Modeling
# az acr login -n sramdevregistry -g devboxes
docker build -t sramdevregistry.azurecr.io/boren_dev:verl -f Dockerfile.verl .
docker push sramdevregistry.azurecr.io/boren_dev:verl

# sudo nvidia-docker run --net host --ipc host -v /home/boren:/home/boren --memory 416G --name boren_dev speechpipelineregistry01.azurecr.io/cascades:official
# sudo nvidia-docker exec -it cascades bash
