#!/bin/bash
#SBATCH --ntasks=1                     
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4              
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00-12:00:00 

# Go to the current folder
SCRIPTDIR=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPTDIR=$(realpath "$SCRIPTDIR")
cd $(dirname "$SCRIPTDIR")

# Setup environment (adapt if necessary)
source /srv/beegfs-benderdata/scratch/${USER}/data/conda/etc/profile.d/conda.sh
conda activate py39

# Add environ var
export DATA=/srv/beegfs-benderdata/scratch/... # TODO
export TMPDIR=/scratch/${USER}
export SAVEDIR=/srv/beegfs-benderdata/scratch/${USER}/data/ex1_submission

# setup weights and biases
# export WANDB_API_KEY=$(cat "wandb.key")
# export WANDB_DIR=${TMPDIR}
# export WANDB_CACHE_DIR=${TMPDIR}
# export WANDB_CONFIG_DIR=${TMPDIR}

# Extract dataset (do not change this)
echo "Loading dataset"
mkdir -p ${TMPDIR}
mkdir -p ${SAVEDIR}
tar -xf ${DATA} -C ${TMPDIR}

# Run training
echo "Start training"

# You can specify the hyperparameters and the experiment name here.
python -m src.scripts.train \
  --log_dir ${SAVEDIR} \
  --dataset_root ${TMPDIR}/miniscapes \
  --name experiment_name \
  --model_name pae \
  --optimizer adam \
  --optimizer_lr 0.0001 \
  --batch_size 16 \
  --num_epochs 3 \
  --workers ${SLURM_CPUS_PER_TASK} \
  --workers_validation ${SLURM_CPUS_PER_TASK} \
  --batch_size_validation 16 \
  --optimizer_float_16 no \
  
# END YOUR CHANGES HERE