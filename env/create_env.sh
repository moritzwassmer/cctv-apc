#!/bin/bash

# create conda environment
export USER=$(whoami)
mkdir -p /work/$USER/envs
conda config --add envs_dirs /work/$USER/envs 
conda env create --prefix /work/$USER/envs/napc_xlstm -f environment.yaml
conda env list