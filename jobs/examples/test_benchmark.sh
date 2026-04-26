#!/bin/bash --login
#SBATCH --job-name=training_benchmark
#SBATCH --output=exec/%x.%a.out
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1,gpu_mem:30G,ccc:80


deviceQuery

# 0) select python environment 
conda activate cctv-apc 
# use environment_benchmark.yaml as environment.yaml will likely not work.

# 1) Set experiment name 
formatted_date=$(date +"%Y-%m-%d-%H-%M-%S")
echo "$formatted_date"
export EXP_NAME_SHORT=benchmark\_compile
export EXP_NAME=$formatted_date\_$EXP_NAME_SHORT # e.g. 2023-10-01-12-00-00_test

# 2) copy source files to exec folder
export EXEC=exec/$EXP_NAME 
export GIT_REPO=$(pwd)
export NAPC=$GIT_REPO/napc
mkdir $EXEC # Create directory where job is executed
cp "$0" $EXEC/job.sh # copy this job file
mkdir $EXEC/napc
rsync -av --exclude='__pycache__/' $NAPC $EXEC # copy all files from napc folder to exec folder
cp $GIT_REPO/run.py $EXEC/run.py # copy run.py
cp $GIT_REPO/environment.yaml $EXEC/environment.yaml # copy environment.yaml

# 3) create zip 
export ZIP_PWD="pE6#vTAJA2Eazt%9pVdtjB0y1W8Fad@f^bGSkCCPpTVGv3hP6VSJ8Vq8#yEqwqr0"
cd $GIT_REPO  # Ensure relative paths inside archive are correct
7z a -p"$ZIP_PWD" -mhe=on backup.7z napc run.py
mv backup.7z $EXEC/

# 4) create mlflow experiment, optionally choose a mlflow tracking server uri to log to
# this is done automatically in the python script, but there can be contention issues when multiple jobs are submitted at the same time
#export MLFLOW_TRACKING_URI=https://dagshub.com/dagshub.frisbee891/thesis.mlflow
#export MLFLOW_TRACKING_URI=file:/work/wassmer/cctv-apc/output_folder/thesis/mlflow
#export MLFLOW_TRACKING_USERNAME=dagshub.frisbee891
#export MLFLOW_TRACKING_PASSWORD=f4720093d28a7230d22132899679b6761e496ad2

export MLFLOW_CREDS=$GIT_REPO/mlflow_creds/default.yaml 
export MLFLOW_TRACKING_URI=$(yq -r '.uri' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_USERNAME=$(yq -r '.username' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_PASSWORD=$(yq -r '.password' "$MLFLOW_CREDS")
mlflow experiments create --experiment-name $EXP_NAME

# 5) execute training runs SEQUENTIALLY, SAME NODE
# slstm und mlstm
cd $EXEC
bash $GIT_REPO/jobs/it2/get_hw_info.sh

MODELS=(
  conv_mlstm_4x128 conv_slstm_4x128
)

DMS=(cctv_vd2_cut2)

SEEDS=(1)

for MODEL in "${MODELS[@]}"; do
  for DM in "${DMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      python run.py \
        --config-name=local_benchmark \
        run=full_run \
        hydra.launcher.name="$EXP_NAME_SHORT" \
        hydra.launcher.gpus_per_node=1 \
        run.trainer.logger.experiment_name="$EXP_NAME" \
        run/model="$MODEL" \
        run/dm="$DM" \
        run.dm.random_seed=$SEED \
        run.mode=train \
        run.trainer.max_epochs=11 \
        run.model.warmup_steps=20000 \
        run.model.decay_until_step=250000 \
        run.dm.train_ds_conf.prob_time_reverse=0.5 \
        run.trainer.gradient_clip_val=1.0 \
        run.trainer.gradient_clip_algorithm=norm \
        run.model.optimizer=beck24 \
        run.model.lr_scheduler=beck24 \
        run.model.intermediate.config.dropout=0.0 \
        run.model.min_lr=0 \
        run/trainer/callbacks=es \
        run.trainer.callbacks.lg_sys._target_=napc.callbacks.SystemMetricsLoggerBenchmark \
        run.trainer.enable_checkpointing=False \
        run.trainer.callbacks.time_tracker.reset_first_n_epochs=1 \
        run.mlflow_creds=$MLFLOW_CREDS \
        run.do_compile=true 
      rm -r /work/$USER/.triton_cache/*
    done
  done
done

# vlstm

MODELS=(
  conv_lstm_4x128
)

for MODEL in "${MODELS[@]}"; do
  for DM in "${DMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      python run.py \
        --config-name=local_benchmark \
        run=full_run \
        hydra.launcher.name="$EXP_NAME_SHORT" \
        hydra.launcher.gpus_per_node=1 \
        run.trainer.logger.experiment_name="$EXP_NAME" \
        run/model="$MODEL" \
        run/dm="$DM" \
        run.dm.random_seed=$SEED \
        run.mode=train \
        run.trainer.max_epochs=11 \
        run.model.warmup_steps=20000 \
        run.model.decay_until_step=250000 \
        run.dm.train_ds_conf.prob_time_reverse=0.5 \
        run.trainer.gradient_clip_val=1.0 \
        run.trainer.gradient_clip_algorithm=norm \
        run.model.optimizer=beck24 \
        run.model.lr_scheduler=beck24 \
        run.model.intermediate.dropout_rate=0.0 \
        run.model.min_lr=0 \
        run/trainer/callbacks=es \
        run.trainer.callbacks.lg_sys._target_=napc.callbacks.SystemMetricsLoggerBenchmark \
        run.trainer.enable_checkpointing=False \
        run.trainer.callbacks.time_tracker.reset_first_n_epochs=1 \
        run.mlflow_creds=$MLFLOW_CREDS \
        run.do_compile=true 
      rm -r /work/$USER/.triton_cache/*
    done
  done
done