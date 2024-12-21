import argparse
import os
import json

import yaml

from src.utils.constants import *
from src.utils.helpers import DotDict, flatten_dict


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--name', type=str, required=True, help='Name for your run to easier identify it.')
    parser.add_argument(
        '--log_dir', type=expandpath, required=True, help='Place for artifacts and logs')
    parser.add_argument(
        '--logging', type=str2bool, required=True, help='Whether to use logging')
    parser.add_argument(
        '--dataset_root', type=expandpath, required=True, help='Path to dataset')
    parser.add_argument(
        '--resume', type=str, default=None, help='Resume training from checkpoint: path to a valid file')
    parser.add_argument(
        '--ckpt_save_dir', type=str, default=None, help='Path to save checkpoints at every epoch end')


    parser.add_argument(
        '--num_epochs', type=int, default=16, help='Number of training epochs')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Number of samples in a batch for training')
    parser.add_argument(
        '--batch_size_validation', type=int, default=8, help='Number of samples in a batch for validation')


    """
    Optimizer config
    """
    parser.add_argument(
        '--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
    parser.add_argument(
        '--optimizer_lr', type=float, default=0.01, help='Learning rate at start of training')
    parser.add_argument(
        '--optimizer_momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument(
        '--optimizer_weight_decay', type=float, default=0.001, help='Optimizer weight decay')
    parser.add_argument(
        '--optimizer_float_16', type=str2bool, default=False, help='Optimizer to use float16 precision')
    parser.add_argument(
        '--lr_scheduler', type=str, default='poly', choices=['poly'], help='Type of learning rate scheduler')
    parser.add_argument(
        '--lr_scheduler_power', type=float, default=0.9, help='Poly learning rate power')

    """
    Dataset config
    """
    parser.add_argument(
        '--dataset', type=str, default='audio_dataset', choices=['mock_dataset', 'audio_dataset'], help='Dataset name') 


    """
    Model config
    Based on https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/Network.py 
    """
    parser.add_argument(
        '--model_name', type=str, default='pae',
        choices=['pae', 'pae_flat'],
        help='name of the model')
    parser.add_argument(
        '--input_channels', type=int, default=input_channels, help="number of channels along time in the input data (here 3*J as XYZ-component of each joint)")
    parser.add_argument(
        '--embedding_channels', type=int, default=phase_channels, help="desired number of latent phase channels (usually between 2-10)")
    parser.add_argument(
        '--time_range', type=int, default=frames, help="number of frames, int(window * fps) + 1")
    parser.add_argument(
        '--window', type=float, default=window, help="time duration of the time window")
    parser.add_argument(
        '--model_encoder_name', type=str, default='', choices=[''], # TODO: replace with our own model names
        help='model encoder architecture')
    

    """
    Worker config
    """
    parser.add_argument(
        '--workers', type=int, default=16, help='Number of worker threads fetching training data')
    parser.add_argument(
        '--workers_validation', type=int, default=4, help='Number of worker threads fetching validation data')


    cfg = parser.parse_args()
    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))
    return cfg


def yaml_config_parser(flatten_config=False) -> DotDict:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--config_file', type=str, default='../configs/mock_config.yaml', help='path to yaml config')
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config_file, flatten=flatten_config)
    return cfg
    
    
def load_config(path, flatten) -> DotDict:
    with open(path) as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            if flatten:
                yaml_dict = flatten_dict(yaml_dict)
            return DotDict(yaml_dict)
        except yaml.YAMLError as exc:
            print("error loading config: ", exc)


EXPERIMENT_INVARIANT_KEYS = (
    'log_dir',
    'dataset_root',
    'prepare_submission',
    'batch_size_validation',
    'workers',
    'workers_validation',
    'num_steps_visualization_first',
    'num_steps_visualization_interval',
    'visualize_num_samples_in_batch',
    'visualize_img_grid_width',
    'observe_train_ids',
    'observe_valid_ids',
)