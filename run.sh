#!/bin/bash

job_name=test-san
work_dir=/nfs/home/spinnawala/Repos/monai-3d-spleen-seg-testing
docker_tag=aicregistry:5000/${USER}/test-img:v1

runai delete job ${job_name}
runai submit \
  ${job_name} \
  -i ${docker_tag} \
  -g 1 \
  -p spinnawala \
  -v /nfs:/nfs \
  --run-as-user \
  --large-shm \
  --backoff-limit 0 \
  --host-ipc \
  --cpu-limit 16 \
  --command -- python -u ${work_dir}/main.py fit --config ${work_dir}/config.yaml