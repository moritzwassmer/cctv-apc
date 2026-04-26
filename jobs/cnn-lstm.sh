#!/bin/bash

# This script runs a grid search for the CNN-LSTM architecture for different LSTM types and model sizes.

# 0) select python environment 
conda activate cctv-apc

# 1) Set experiment name 
formatted_date=$(date +"%Y-%m-%d-%H-%M-%S")
echo "$formatted_date"
export EXP_NAME_SHORT=cnn-lstm
export EXP_NAME=$formatted_date\_$EXP_NAME_SHORT # e.g. 2023-10-01-12-00-00_test

# 2) copy source files to exec folder
export EXEC=exec/$EXP_NAME
export GIT_REPO="$(pwd)"
export NAPC=$GIT_REPO/napc
mkdir $EXEC # Create directory where job is executed
cp "$0" $EXEC/job.sh # copy this job file
mkdir $EXEC/napc
rsync -av --exclude='__pycache__/' $NAPC $EXEC # copy all files from napc folder to exec folder
cp $GIT_REPO/run.py $EXEC/run.py # copy run.py
cp $GIT_REPO/environment.yaml $EXEC/environment.yaml 

# 3) create zip with napc directory and run.py file (backup, documentation)
export ZIP_PWD="pE6#vTAJA2Eazt%9pVdtjB0y1W8Fad@f^bGSkCCPpTVGv3hP6VSJ8Vq8#yEqwqr0"
cd $GIT_REPO  # Ensure relative paths inside archive are correct
7z a -p"$ZIP_PWD" -mhe=on backup.7z napc run.py
mv backup.7z $EXEC/

# 4) create mlflow experiment
export MLFLOW_CREDS=$GIT_REPO/mlflow_creds/default.yaml 
export MLFLOW_TRACKING_URI=$(yq -r '.uri' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_USERNAME=$(yq -r '.username' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_PASSWORD=$(yq -r '.password' "$MLFLOW_CREDS")
mlflow experiments create --experiment-name $EXP_NAME

# 5.a) submit jobs xlstm
cd $EXEC
python run.py \
--config-name=gpu_big_math \
hydra.launcher.timeout_min=2870 \
hydra.launcher.name=$EXP_NAME_SHORT \
run=full_run \
run.mlflow_creds=$MLFLOW_CREDS \
run/model=conv_mlstm_4x128,conv_mlstm_8x128,conv_mlstm_12x128,conv_mlstm_4x256,\
conv_slstm_4x128,conv_slstm_8x128,conv_slstm_12x128,conv_slstm_4x256 \
run.model.warmup_steps=20000 \
run.model.decay_until_step=250000 \
run.model.optimizer=beck24 \
run.model.lr_scheduler=beck24 \
run.model.intermediate.config.dropout=0.0 \
run.model.min_lr=0 \
run/dm=cctv_vd2_cut2_STD_global \
run.dm.random_seed=1,2,3,4,5,6,7,8,9,10 \
run.dm.train_ds_conf.prob_time_reverse=0.5 \
run/trainer/callbacks=cp-last-Acc-absgrb_es \
run.trainer.logger.experiment_name=$EXP_NAME \
run.trainer.max_epochs=1250 \
run.trainer.gradient_clip_val=1.0 \
run.trainer.gradient_clip_algorithm=norm \
--multirun &

# 5.b) submit jobs vlstm
python run.py \
--config-name=gpu_big_math \
hydra.launcher.timeout_min=2870 \
hydra.launcher.name=$EXP_NAME_SHORT \
run=full_run \
run.mlflow_creds=$MLFLOW_CREDS \
run/model=conv_lstm_4x128,conv_lstm_8x128,conv_lstm_4x256 \
run.model.warmup_steps=20000 \
run.model.decay_until_step=250000 \
run.model.optimizer=beck24 \
run.model.lr_scheduler=beck24 \
++run.model.intermediate.dropout_rate=0.0 \
run.model.min_lr=0 \
run/dm=cctv_vd2_cut2_STD_global \
run.dm.random_seed=1,2,3,4,5,6,7,8,9,10 \
run.dm.train_ds_conf.prob_time_reverse=0.5 \
run/trainer/callbacks=cp-last-Acc-absgrb_es \
run.trainer.logger.experiment_name=$EXP_NAME \
run.trainer.max_epochs=1250 \
run.trainer.gradient_clip_val=1.0 \
run.trainer.gradient_clip_algorithm=norm \
--multirun