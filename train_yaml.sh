# Example loading config from yaml

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
python -m src.scripts.train --config_file "$(pwd)/configs/mock_config.yaml"