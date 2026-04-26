#!/bin/bash
# This script is used to continue a single run expecting an existing exec folder

# 0) select python environment 
conda activate cctv-apc

# 1) enter the run ID and experiment path that you want to continue.
export EXEC_PATH=exec
formatted_date=2026-01-24-14-45-51 # TODO change
export EXP_NAME_SHORT=test # TODO change
export EXP_NAME=$formatted_date\_$EXP_NAME_SHORT # e.g. 2023-10-01-12-00-00_test
export EXP_PATH=${EXEC_PATH}/${EXP_NAME}
export RUN_ID=f321342bfd0a40f1bfe9fe7a2b672f7a

# 2) copy the config file to the napc/configs/run folder 
cp ${EXP_PATH}/out/${RUN_ID}/config.yaml ${EXP_PATH}/napc/configs/run/${RUN_ID}.yaml

cd ${EXP_PATH}

# 3) launch the job using slurm

python run.py \
--config-name=cpu \
run=${RUN_ID} \
hydra.launcher.name=${EXP_NAME_SHORT} \
run.ckpt_path=out/${RUN_ID}/checkpoints/last.ckpt \
run.trainer.logger.run_id=${RUN_ID} \
run.skip_meta=true \
--multirun