#!/bin/bash
# This script is to only evaluate all runs within an experiment
# It creates a new exec folder and logs to a new experiment

# 0) select python environment 
conda activate cctv-apc

# 1) Choose experiment and checkpoint to evaluate
export EXP_NAME=2026-01-24-14-45-51_test
export EXEC=exec/$EXP_NAME
export CHKPT=val_avg_Acc
export RUN_ID=f321342bfd0a40f1bfe9fe7a2b672f7a # TODO change

# 2) choose where to log to
export MLFLOW_CREDS=$(pwd)/mlflow_creds/default.yaml # TODO change
export MLFLOW_TRACKING_URI=$(yq -r '.uri' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_USERNAME=$(yq -r '.username' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_PASSWORD=$(yq -r '.password' "$MLFLOW_CREDS")
mlflow experiments create --experiment-name $EXP_NAME

# 3) create new experiment folder
export EXEC2=$EXEC\_evaluated
mkdir $EXEC2
rsync -av --exclude='__pycache__/' $EXEC/ $EXEC2 # copy all files from napc folder to exec folder


cd $EXEC2

cp out/${RUN_ID}/config_unresolved.yaml napc/configs/run/${RUN_ID}.yaml

python run.py \
--config-name=cpu \
run=${RUN_ID}  \
run.mlflow_creds=$MLFLOW_CREDS \
run.ckpt_path=out/${RUN_ID}/checkpoints/${CHKPT}.ckpt \
hydra.launcher.timeout_min=15 \
run.trainer.logger.run_id=null \
run.trainer.logger.experiment_name=${EXP_NAME} \
run.mode=test \
--multirun