import PIL
import PIL.Image
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID
from src.utils.metrics import MAE
from src.utils.helpers import resolve_dataset_class, resolve_lr_scheduler, resolve_model_class, resolve_optimizer
from src.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss

class VQ_PAEModel(pl.LightningModule):

    def __init__(self, cfg, **kwargs) -> None:
        super(VQ_PAEModel, self).__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset_config
        self.training_config = cfg.training_config
        self.model_config = cfg.model_config
        self.worker_config = cfg.worker_config
        self.loss_config = cfg.loss_config

        self.D, self.N, self.K = self.model_config.input_channels, self.model_config.time_range, self.model_config.embedding_channels
        dataset_class = resolve_dataset_class(self.dataset_cfg.dataset)
        self.datasets = {
            split: dataset_class(dataset_root=self.dataset_cfg.dataset_root, # dataset root is not important here
                                 split=split) 
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')


        self._instantiate_model()
        self.recon_loss = nn.MSELoss()
        self.stft_loss = STFTLoss()
        self.metric = MAE
        
        self.loss_weights = self.loss_config.loss_weights.to_dict()


    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)

        pred = self.model(batch) 
        
        loss_items = {}
        x_recon = pred["audio"]
        # z_pae = pred["z_pae"] # B, emb_ch, L
        # z_vq = pred["z_vq"]   # B, emb_ch, L
        # codes = pred["codes"] # B, n_codebooks, L
        # latents = pred["latents"] # B, n_codebooks x codebook_dim, L
        vq_comm_loss = pred["vq/commitment_loss"]
        vq_cb_loss = pred["vq/codebook_loss"]
        
        # add loss items
        loss_items["mse_loss"] = self.recon_loss(batch, x_recon)
        stft_loss = self.stft_loss(batch, x_recon)
        loss_items["stft_loss_spectral_convergence"], loss_items["stft_loss_magnitude"] = \
            stft_loss[0], stft_loss[1]
        loss_items["vq/vodebook_loss"] = vq_cb_loss
        loss_items["vq/commitment_loss"] = vq_comm_loss
        loss_items["total_loss"] = sum([v * loss_items[k] for k, v in self.loss_weights.items() if k in loss_items])
        self.log_dict(
            {f"loss_train/{k}": v for k, v in loss_items.items()}, 
            on_step=True, on_epoch=False, prog_bar=True
        )

        return {
            'loss': loss_items["total_loss"]
        }
        

    def validation_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred = self.model(batch) 
        
        loss_items = {}
        x_recon = pred["audio"]
        # z_pae = pred["z_pae"] # B, emb_ch, L
        # z_vq = pred["z_vq"]   # B, emb_ch, L
        # codes = pred["codes"] # B, n_codebooks, L
        # latents = pred["latents"] # B, n_codebooks x codebook_dim, L
        vq_comm_loss = pred["vq/commitment_loss"]
        vq_cb_loss = pred["vq/codebook_loss"]
        
        # add loss items
        loss_items["mse_loss"] = self.recon_loss(batch, x_recon)
        stft_loss = self.stft_loss(batch, x_recon)
        loss_items["stft_loss_spectral_convergence"], loss_items["stft_loss_magnitude"] = \
            stft_loss[0], stft_loss[1]
        loss_items["vq/vodebook_loss"] = vq_cb_loss
        loss_items["vq/commitment_loss"] = vq_comm_loss
        loss_items["total_loss"] = sum([v * loss_items[k] for k, v in self.loss_weights.items() if k in loss_items])
        self.log_dict(
            {f"loss_val/{k}": v for k, v in loss_items.items()}, 
            on_step=False, on_epoch=True, prog_bar=True
        )
        
        # log figures, randomly select one validation signal
        k = torch.randint(0, batch.shape[0], (1,)).item()
        x = batch[k, ...]
        y = x_recon[k, ...]
        
        title_prefix = 'images_val'
        fig_reconstruction = self._visualize_reconstruction(x, y)
        figure_log_spectogram = self._visualize_log_spectogram(x, y)
        self._log_figure(title=f"{title_prefix}/Reconstruction", caption="Input vs Reconstructed Signal", 
                         img=fig_reconstruction)
        self._log_figure(title=f"{title_prefix}/Log Spectogram", caption="Input and Output Log Spectograms", 
                         img=figure_log_spectogram)
    

    def train_dataloader(self):
        return self._create_train_dataloader()


    def val_dataloader(self):
        return self._create_val_test_dataloader(SPLIT_VALID)


    def test_dataloader(self):
        return self._create_val_test_dataloader(SPLIT_TEST)
    

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.training_config, self.model.parameters())
        lr_scheduler = resolve_lr_scheduler(self.training_config, optimizer)
        return [optimizer], [lr_scheduler]
    

    def _create_train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.training_config.batch_size,
            shuffle=True,
            num_workers=self.worker_config.workers,
            pin_memory=True,
            drop_last=True,
        )


    def _create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.training_config.batch_size_validation,
            shuffle=False,
            num_workers=self.worker_config.workers_validation,
            pin_memory=True,
            drop_last=False,
        )
    
    
    def _inference_step(self, batch):
        pass


    def _instantiate_model(self):
        self.model = resolve_model_class(self.model_config.model_name, self.model_config)
        if torch.cuda.is_available():
            self.model.to(self.device)
            
            
    def _log_figure(self, title: str, caption: str, img: PIL.Image) -> None:        
        # Log the figures to wandb
        if self.logger is not None and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                title: wandb.Image(img, caption=caption)
            })
        
        
    def _visualize_reconstruction(self, x, y) -> PIL.Image:
        """
            Plot input signal x against output signal y
        """
        assert len(x.shape) == len(y.shape) == 1, "int/out have to be flattened 1D tensors"

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        T = 500 # only show first T steps

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, T+1), x[:T], label="Input Signal", color="blue")
        plt.plot(np.arange(1, T+1), y[:T], label="Reconstructed Signal", color="green")
        plt.xlabel("Steps")
        plt.ylabel("Signal")
        plt.title("Input vs Reconstructed Signal")
        plt.legend()
        plt.grid()
        
        img = self._fig_to_PIL_img(fig)
        plt.close()

        return img

    
    
    def _visualize_log_spectogram(self, x, y, n_bins: int = 500) -> PIL.Image:
        """ plot input and output (log) Spectograms """
        assert len(x.shape) == len(y.shape) == 1, "int/out have to be flattened 1D tensors"
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
         
        # Compute Fourier Transforms
        input_fft = np.fft.fft(x)
        reconstructed_fft = np.fft.fft(y)

        # Compute log magnitudes
        log_magnitude_input = np.log1p(np.abs(input_fft)[:self.N // 2])
        log_magnitude_reconstructed = np.log1p(np.abs(reconstructed_fft)[:self.N // 2])

        # Create bins for histograms
        bins = np.linspace(0, np.max([log_magnitude_input.max(), log_magnitude_reconstructed.max()]), n_bins)

        # Plot histograms of the log magnitudes
        fig = plt.figure(figsize=(10, 6))

        plt.hist(log_magnitude_input, bins=bins, alpha=0.7, label="Input Signal (Log Magnitude)", color="blue")
        plt.hist(log_magnitude_reconstructed, bins=bins, alpha=0.7, label="Reconstructed Signal (Log Magnitude)", color="orange")

        plt.title("Histogram of Log Magnitudes of Fourier Transform")
        plt.ylabel("Log Magnitude")
        plt.xlabel("Frequency")
        plt.legend()
        plt.grid()
        
        img = self._fig_to_PIL_img(fig)
        plt.close()

        return img
    
        
    @staticmethod
    def _fig_to_PIL_img(fig) -> PIL.Image:
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = PIL.Image.frombytes(
            'RGB', 
            canvas.get_width_height(),
            canvas.tostring_rgb()
        )
        
        return img