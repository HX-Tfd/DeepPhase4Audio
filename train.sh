# exit when any command fails
set -e

# Add environ var
export DATA='..'
export TMPDIR='tmp'
export SAVEDIR='logs'
export DATADIR='data'

# setup weights and biases
export WANDB_API_KEY=$(cat "wandb.key")
export WANDB_DIR=${TMPDIR}
export WANDB_CACHE_DIR=${TMPDIR}
export WANDB_CONFIG_DIR=${TMPDIR}


# Run training
echo "Start training"

# Specify the hyperparameters and the experiment name here.
# log_dir changed to local folder to bypass rule checks on the cluster
# for PAE configs, see src/utils/config.py and src/utils/constants.py
python -m src.scripts.train \
  --logging no \
  --log_dir ${SAVEDIR} \
  --dataset_root ${DATADIR} \
  --dataset 'audio_dataset' \
  --input_channels 1 \
  --embedding_channels 5 \
  --time_range 32000 \
  --window 2.0 \
  --name PAEInputFlattenedModel \
  --model_name pae_flat \
  --optimizer adam \
  --optimizer_lr 0.01 \
  --batch_size 16 \
  --num_epochs 16 \
  --workers 0 \
  --workers_validation 0 \
  --batch_size_validation 4 \
  --optimizer_float_16 no \