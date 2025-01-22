#!/bin/bash
#SBATCH --ntasks=1                     
#SBATCH --nodes=1 
#SBATCH --account=dl_jobs
#SBATCH --output=logs/%j.out
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00-14:00:00 
# exit when any command fails

set -e

# Add environ var
export DATA='..'
export TMPDIR='tmp'
export SAVEDIR='logs'
export DATADIR='data'

# Go to the current folder
# SCRIPTDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
# SCRIPTDIR=$(realpath "$SCRIPTDIR")
# cd $(dirname "$SCRIPTDIR")

# setup weights and biases
export WANDB_API_KEY=$(cat "wandb.key")
export WANDB_DIR=${TMPDIR}
export WANDB_CACHE_DIR=${TMPDIR}
export WANDB_CONFIG_DIR=${TMPDIR}

# Run training
echo "Start training"
python -m src.scripts.grid_search --config_file "$(pwd)/configs/grid.yaml"
echo "Finished Grid Search"