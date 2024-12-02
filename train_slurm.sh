#!/bin/bash
#SBATCH --ntasks=1                     
#SBATCH --nodes=1 
#SBATCH --account=dl
#SBATCH --output=logs/test_output.out
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00-6:00:00 

# Note that specifying TRES per node and task is not allowed on this cluster

# exit when any command fails
# set -e

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
export WANDB_API_KEY=$(cat "wandb.key") # make sure you create this file and put your API key inside if you set --logging true
export WANDB_DIR=${TMPDIR}
export WANDB_CACHE_DIR=${TMPDIR}
export WANDB_CONFIG_DIR=${TMPDIR}

# Run training
echo "Start training"

python -m src.scripts.train \
  --logging no \
  --log_dir ${SAVEDIR} \
  --dataset_root ${DATADIR} \
  --name mock \
  --model_name pae \
  --optimizer adam \
  --optimizer_lr 0.01 \
  --batch_size 16 \
  --num_epochs 16 \
  --workers 0 \
  --workers_validation 0 \
  --batch_size_validation 4 \
  --optimizer_float_16 no \