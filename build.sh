#!/bin/bash

docker_tag=aicregistry:5000/${USER}/test-img:v1

docker build --no-cache . -f Dockerfile \
 -t ${docker_tag} \
 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host
docker push ${docker_tag}