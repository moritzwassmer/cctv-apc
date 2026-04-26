#!/bin/bash
conda activate cctv-apc

export MLFLOW_CREDS=$(pwd)/mlflow_creds/default.yaml # TODO change

# 5) submit job, overwrite defaults. "<value1>,<value2>", seperated means it will do a slurm array job doing a grid search on all values.
python run.py \
--config-name=gpu_slstm_math \
hydra.launcher.timeout_min=5 \
run=full_run \
run.mlflow_creds=$MLFLOW_CREDS \
run/model=conv_slstm_2x128 \
run.trainer.max_epochs=2 \
run.trainer.logger.experiment_name=test \
run.trainer.limit_train_batches=1 \
run.trainer.limit_val_batches=1 \
run.trainer.limit_test_batches=1 \
run.dm.batch_size=2 \
--multirun