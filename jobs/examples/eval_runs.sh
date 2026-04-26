#!/bin/bash
# This script is to only evaluate all runs within an experiment
# It creates a new exec folder and logs to a new experiment

# 0) select python environment 
conda activate cctv-apc

# 1) Choose experiment and checkpoint to evaluate
export EXP_NAME=2026-02-05-22-44-36_it2_test_dataset
export EXEC=exec/$EXP_NAME
export CHKPT=val_avg_abs_grb

# 2) choose where to log to
export NEW_EXP_NAME=${EXP_NAME}\_eval\_$CHKPT
export MLFLOW_CREDS=$(pwd)/mlflow_creds/default.yaml # TODO change
export MLFLOW_TRACKING_URI=$(yq -r '.uri' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_USERNAME=$(yq -r '.username' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_PASSWORD=$(yq -r '.password' "$MLFLOW_CREDS")
mlflow experiments create --experiment-name $NEW_EXP_NAME

# 3) create new experiment folder
export EXEC2=exec/$NEW_EXP_NAME
mkdir $EXEC2
rsync -av --exclude='__pycache__/' $EXEC/ $EXEC2 # copy all files from napc folder to exec folder


cd $EXEC2

# Loop through all run folders in "out/" and launch test job via Submitit
for RUN_DIR in out/*/; do
    RUN_ID=$(basename "$RUN_DIR")

    # Copy config
    cp "out/${RUN_ID}/config_unresolved.yaml" "napc/configs/run/${RUN_ID}.yaml"

    # Launch job via Hydra + Submitit (non-blocking)
    python run.py \
        --config-name=gpu_slstm_math \
        run=${RUN_ID} \
        run.mlflow_creds=$MLFLOW_CREDS \
        run.ckpt_path="out/${RUN_ID}/checkpoints/${CHKPT}.ckpt" \
        hydra/launcher=submitit_slurm \
        hydra.launcher.timeout_min=15 \
        run.trainer.logger.run_id=null \
        run.trainer.logger.experiment_name=${NEW_EXP_NAME} \
        run.mode=test \
        --multirun &  # run in background

done

wait  # Wait for all background jobs to complete (optional)