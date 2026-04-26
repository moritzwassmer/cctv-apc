#!/bin/bash
# This script is used to continue an multiple runs of an experiment expecting an existing exec folder. It expects a txt containing the
# runs to continue and the experiment name and path. It will launch jobs using slurm. 

# 0) select python environment 
conda activate cctv-apc

# 1) enter the path of the experiment that you want to continue.
export EXEC_PATH=exec
formatted_date=2026-01-24-14-45-51 # TODO change
export EXP_NAME_SHORT=test # TODO change
export EXP_NAME=${formatted_date}\_$EXP_NAME_SHORT # e.g. 2025-07-15-13-37-44_ex2_lstm
export EXP_PATH=${EXEC_PATH}/${EXP_NAME}

# 2) enter the path of a text file, containing all run ids that shall be continued. 
# Use notebooks/get_failed_runs.ipynb for example
RUN_LIST_FILE=$(pwd)/out/${EXP_NAME}.txt

# 3) optional: add/change/remove parameters
cd ${EXP_PATH}

while read -r RUN_ID; do
    echo "Continuing run: ${RUN_ID}"

    # 1. Copy config file
    cp ${EXP_PATH}/out/${RUN_ID}/config.yaml ${EXP_PATH}/napc/configs/run/${RUN_ID}.yaml

    # 2. Run continuation script
    python run.py \
    --config-name=gpu_big_math \
    run=${RUN_ID} \
    hydra.launcher.name=${EXP_NAME_SHORT} \
    hydra.launcher.timeout_min=4320 \
    run.ckpt_path=out/${RUN_ID}/checkpoints/last.ckpt \
    run.trainer.logger.run_id=${RUN_ID} \
    run.skip_meta=true \
    --multirun &

done < "$RUN_LIST_FILE"
