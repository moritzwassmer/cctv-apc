# This job is to restart failed runs in an existing experiment. it expects a txt file containing the run IDs to restart.

# 1) Select Experiment
export EXEC_PATH=exec
formatted_date=2026-01-24-14-45-51
export EXP_NAME_SHORT=test
export EXP_NAME=${formatted_date}\_$EXP_NAME_SHORT # e.g. 2025-07-15-13-37-44_ex2_lstm
export EXP_PATH=${EXEC_PATH}/${EXP_NAME}

# 2) Select Run List
RUN_LIST_FILE=$(pwd)/out/${EXP_NAME}.txt

# Restart the failed runs
cd ${EXP_PATH}
while read -r RUN_ID; do
    echo "restart run: ${RUN_ID}"

    # Copy config file
    cp ${EXP_PATH}/out/${RUN_ID}/config_unresolved.yaml ${EXP_PATH}/napc/configs/run/${RUN_ID}.yaml

    # Run continuation script
    python run.py \
    --config-name=cpu \
    run=${RUN_ID} \
    hydra.launcher.name=${EXP_NAME_SHORT} \
    --multirun &

done < "$RUN_LIST_FILE"
