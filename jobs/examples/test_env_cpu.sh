#!/bin/bash
conda activate cctv-apc

export MLFLOW_CREDS=$(pwd)/mlflow_creds/default.yaml # TODO change

# 5) submit job
python run.py \
--config-name=cpu \
run=test_run \
run.mlflow_creds=$MLFLOW_CREDS \
--multirun