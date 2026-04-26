#!/bin/bash
# This is a template run that starts a grid search for 2 seeds of conv_lstm_2x128

# 0) select python environment 
conda activate cctv-apc

# 1) Set experiment name 
formatted_date=$(date +"%Y-%m-%d-%H-%M-%S")
echo "$formatted_date"
export EXP_NAME_SHORT=test # TODO change
export EXP_NAME=$formatted_date\_$EXP_NAME_SHORT # e.g. 2023-10-01-12-00-00_test

# 2) copy source files to exec folder
# this ensures that the job runs with the exact same code as when it was submitted, even if the code changes in the meantime
export EXEC=exec/$EXP_NAME
export GIT_REPO="$(pwd)"
export NAPC=$GIT_REPO/napc
mkdir $EXEC # Create directory where job is executed
cp "$0" $EXEC/job.sh # copy this job file
mkdir $EXEC/napc
rsync -av --exclude='__pycache__/' $NAPC $EXEC # copy all files from napc folder to exec folder
cp $GIT_REPO/run.py $EXEC/run.py # copy run.py
cp $GIT_REPO/environment.yaml $EXEC/environment.yaml # copy environment.yaml

# 3) create zip with napc directory and run.py file
# this is mainly for reproducibility, so we have a backup of the exact code that was used for this experiment
export ZIP_PWD="pE6#vTAJA2Eazt%9pVdtjB0y1W8Fad@f^bGSkCCPpTVGv3hP6VSJ8Vq8#yEqwqr0"
cd $GIT_REPO  # Ensure relative paths inside archive are correct
7z a -p"$ZIP_PWD" -mhe=on backup.7z napc run.py
mv backup.7z $EXEC/

# 4) create mlflow experiment
# this is done automatically in the python script, but there can be contention issues when multiple trainings start at the same time each trying to create the experiment simultaneously
export MLFLOW_CREDS=$GIT_REPO/mlflow_creds/default.yaml # TODO change
export MLFLOW_TRACKING_URI=$(yq -r '.uri' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_USERNAME=$(yq -r '.username' "$MLFLOW_CREDS")
export MLFLOW_TRACKING_PASSWORD=$(yq -r '.password' "$MLFLOW_CREDS")
mlflow experiments create --experiment-name $EXP_NAME

# 5) submit job, overwrite defaults. 
#### SYNTAX ####
# For details see hydra manual https://hydra.cc/docs/intro/
# And omegaconf documentation https://omegaconf.readthedocs.io/en/2.1_branch/index.html
# use "." to describe the path to the parameter that should be changed e.g. run.dm.batch_size=8
# comma seperated values will run every possible combination of the values provided e.g. run.dm.random_seed=1,2 will run 2 trainings with random seed 1 and 2
# use "/" to describe the path to change predifined config files, e.g. run/model=conv_lstm_2x128 will replace the model that is selected by default with the specified model. the file is saved in napc/configs/run/model/conv_lstm_2x128.yaml
# example of a fully specified config in jobs/examples/config.yaml

# TODO change
cd $EXEC
python run.py \
--config-name=cpu \
hydra.launcher.timeout_min=10 \
run=full_run \
run.mlflow_creds=$MLFLOW_CREDS \
run/model=conv_lstm_2x128 \
run.trainer.max_epochs=10 \
run.trainer.logger.experiment_name=$EXP_NAME \
run.trainer.limit_train_batches=2 \
run.trainer.limit_val_batches=1 \
run.trainer.limit_test_batches=1 \
run.dm.batch_size=8 \
run.dm.random_seed=1,2 \
--multirun