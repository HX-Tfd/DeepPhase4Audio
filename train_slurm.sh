#!/bin/bash
#SBATCH --ntasks=1                     
#SBATCH --nodes=1 
#SBATCH --account=dl_jobs
#SBATCH --output=logs/%j.out
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00-6:00:00 

# Note that specifying TRES per node and task is not allowed on this cluster

# exit when any command fails
set -e

# Add environ var
export TMPDIR='tmp'
export SAVEDIR='logs'
export DATADIR='data'

# Go to the current folder
# SCRIPTDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
# SCRIPTDIR=$(realpath "$SCRIPTDIR")
# cd $(dirname "$SCRIPTDIR")


if [ $# -eq 0 ]; then
  echo "Error: No configuration file provided."
  echo "Usage: bash train.sh <path_to_config_file>"
  exit 1
fi

CONFIG_FILE=$1
echo "Using configuration file: $CONFIG_FILE"

# setup weights and biases
export WANDB_API_KEY=$(cat "wandb.key") # make sure you create this file and put your API key inside if you set --logging true
export WANDB_DIR=${TMPDIR}
export WANDB_CACHE_DIR=${TMPDIR}
export WANDB_CONFIG_DIR=${TMPDIR}


# Run training
echo "Start training"
python -m src.scripts.train --config_file "$CONFIG_FILE"
