# add a conda cache folder to a location where you have write access, otherwise installing new env fails
export USER=$(whoami)
rm -rf /work/$USER/conda_cache
mkdir -p /work/$USER/conda_cache
conda config --add pkgs_dirs /work/$USER/conda_cache # echo " - /work/$USER/conda_cache">> ~/.condarc