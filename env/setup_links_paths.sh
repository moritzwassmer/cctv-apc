export USER=$(whoami)

# This is where source code is copied to and where experiments are executed
export EXEC=/work/$USER/exec # TODO change to whatever path you want
mkdir -p $EXEC
rm -f exec
ln -s $EXEC exec

# this is where experiment results (local mlflow, csv prediction files) are saved to
export OUTPUTS_PATH=/work/$USER/outputs # TODO change to whatever path you want
mkdir -p $OUTPUTS_PATH 
rm -f outputs
ln -s $OUTPUTS_PATH outputs
# create folder for mlflow local tracking server
mkdir -p $OUTPUTS_PATH/mlflow
# Create the YAML file
rm -rf mlflow_creds
mkdir -p mlflow_creds
cat > "mlflow_creds/default.yaml" <<EOL
uri: file://$OUTPUTS_PATH/mlflow
username: user
password: password
EOL


### OPTIONAL, for convenience ### 

# this is where cctv data resides
export DATA_PATH=/net/vericon/napc_data/cctv
rm -f cctv
ln -s $DATA_PATH cctv

# this is where the triton cache is located, only relevant when torch.compile is used. 
# it creates a symlink to avoid issues with limited inodes in home directory
# delete the contents occasionally to free up space: rm -rf /work/$USER/.triton_cache/*
export TRITON_CACHE="/work/$USER/.triton_cache" # TODO change to whatever path you want
rm -f /homes/informatik/$USER/.triton
ln -s $TRITON_CACHE /homes/informatik/$USER/.triton