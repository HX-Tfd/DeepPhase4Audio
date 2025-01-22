import torch
import torch.nn as nn
from src.models.PAE import *

from src.utils.config import command_line_parser, yaml_config_parser
import src.utils as utils
from src.experiments.pae_flattened import PAEInputFlattenedModel
from pytorch_lightning import Trainer
from src.utils.helpers import *
from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID
from torch.utils.data import DataLoader
from src.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


"""change the config file to the one you want to use"""

cfg = utils.config.load_config("configs/paeflat200.yaml")

#model = PAEInputFlattenedModel(cfg)

#print(torch.load('logs/models/PAEInputFlattenedModel_epoch_50.ckpt', map_location=torch.device('cpu')))
model = PAEInputFlattenedModel.load_from_checkpoint(f"saved_models/PAEInputFlattenedModel_epoch_{cfg.num_epochs}.ckpt",cfg=cfg)
trainer=Trainer(accelerator=get_device_accelerator(preferred_accelerator='cpu'),
        devices=1,
        default_root_dir=cfg.ckpt_save_dir, 
        max_epochs=cfg.num_epochs,
        num_sanity_val_steps=1,
        precision=16 if cfg.optimizer_float_16 else 32,
        profiler='simple')

model.eval()

#now we need to load the data and test the model
dataset_class = resolve_dataset_class(cfg.dataset)
model.datasets = {split: dataset_class(dataset_root=cfg.dataset_root,split=split) 
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)}

for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
    print(f'Number of samples in {split} split: {len(model.datasets[split])}')

# Create DataLoader for the test dataset
test_loader = DataLoader(model.datasets[SPLIT_TEST], batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

# Define loss functions
mse_loss = torch.nn.MSELoss()
stft_loss = STFTLoss()


# Initialize metrics
total_mse_loss = 0.0
total_stft_sc_loss = 0.0
total_stft_mag_loss = 0.0
num_batches = 0
D, N, K = cfg.input_channels, cfg.time_range, cfg.embedding_channels
"""
# Evaluate the model on the test dataset
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        pred = model.model(batch)[0]
        if D == 1:
            batch = batch.reshape(batch.shape[0], N)
            pred = pred.reshape(pred.shape[0], N)
        else:
            pred = pred.reshape(pred.shape[0], D, N)
        mse = mse_loss(pred, batch)
        stft_l = stft_loss(batch, pred)
        stft_sc, stft_mag = stft_l[0], stft_l[1]

        total_mse_loss += mse.item()
        total_stft_sc_loss += stft_sc.item()
        total_stft_mag_loss += stft_mag.item()
        num_batches += 1

# Compute average losses
avg_mse_loss = total_mse_loss / num_batches
avg_stft_sc_loss = total_stft_sc_loss / num_batches
avg_stft_mag_loss = total_stft_mag_loss / num_batches

# Print results
print(f"Average MSE Loss: {avg_mse_loss}")
print(f"Average STFT Spectral Convergence Loss: {avg_stft_sc_loss}")
print(f"Average STFT Magnitude Loss: {avg_stft_mag_loss}")

# Write results to a report
with open("test_report.txt", "w") as report_file:
    report_file.write(f"Average MSE Loss: {avg_mse_loss}\n")
    report_file.write(f"Average STFT Spectral Convergence Loss: {avg_stft_sc_loss}\n")
    report_file.write(f"Average STFT Magnitude Loss: {avg_stft_mag_loss}\n")
"""
with torch.no_grad():
    batch = next(iter(test_loader))


    # define your mask, where 1 means keep and 0 means remove that signal
    fc_mask1 = np.ones(cfg.embedding_channels)
    latent_mask1 = np.zeros(cfg.embedding_channels)
    fm1 = 0
    lm1 =16
    #fc_mask1[:fm1]=1
    #latent_mask1[:fm1]=1
    


    fc_mask2 = np.zeros(cfg.embedding_channels)
    latent_mask2 = np.zeros(cfg.embedding_channels)

   
    print("latent_mask:",latent_mask1)

    pred1, latent1, signal1, params1 = model.model.masked_forward(batch,fc_mask1,latent_mask1)  # Adjust mask as needed
    #pred2, latent2, signal2, params2 = model.model.masked_forward(batch,fc_mask2,latent_mask2)  # Adjust mask as needed

    if D == 1:
            batch = batch.reshape(batch.shape[0], N)
            pred1 = pred1.reshape(pred1.shape[0], N)
            #pred2 = pred2.reshape(pred2.shape[0], N)
    else:
         pred1 = pred1.reshape(pred1.shape[0], D, N)
         #pred2 = pred2.reshape(pred2.shape[0], D, N)
    
   
    k = 0 
    x = batch[k, ...]
    y1 = pred1[k, ...]
    #y2 =pred2[k, ...]
    selected_latent_signal1 = signal1[k, ...]
    #selected_latent_signal2 = signal2[k, ...] 
    selected_params = [p[k, ...] for p in params1]
    img1=model._visualize_reconstruction(x, y1,selected_latent_signal1)
    #img2=model._visualize_reconstruction(x, y2,selected_latent_signal2)
 
    img1.save(f"reconstruction_plot_{cfg.num_epochs}_{fm1}_fcmask_{lm1}_latentmask.png")
    #img2.save(f"reconstruction_plot_{cfg.num_epochs}_nomask.png")







