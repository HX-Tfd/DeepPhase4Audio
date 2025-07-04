import PIL
import PIL.Image
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID
from src.utils.metrics import MAE
from src.utils.helpers import resolve_dataset_class, resolve_lr_scheduler, resolve_model_class, resolve_optimizer
from src.losses.stft_loss import STFTLoss, MultiResolutionSTFTLoss
import soundfile as sf


class PAEInputFlattenedModel(pl.LightningModule):

    def __init__(self, cfg, **kwargs) -> None:
        super(PAEInputFlattenedModel, self).__init__()
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset_config
        self.training_config = cfg.training_config
        self.model_config = cfg.model_config
        self.worker_config = cfg.worker_config
        
        # D = D+1 when using PE
        self.D, self.N, self.K = self.model_config.input_channels, self.model_config.time_range, self.model_config.embedding_channels
        self.name = self.model_config.experiment_name
        #self.save_hyperparameters()
        seed_everything(42)
        self.saved_once = False

        dataset_class = resolve_dataset_class(self.dataset_cfg.dataset)
        self.datasets = {
            split: dataset_class(dataset_root=self.dataset_cfg.dataset_root, # dataset root is not important here
                                 split=split) 
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')


        self._instantiate_model()
        self.loss = nn.MSELoss()
        self.stft_loss = STFTLoss()
        self.metric = MAE

    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred, _, _, param = self.model(batch) # output is (y, latent, signal, param)
        
        if self.D == 1:
            batch = batch[:, 0, :].reshape(batch.shape[0], self.N)
            pred = pred.reshape(pred.shape[0], self.N)
        else:
            pred = pred.reshape(pred.shape[0], self.D, self.N)
            
        loss = self.loss(batch, pred)
        stft_loss = self.stft_loss(batch, pred)
        stft_sc, stft_mag = stft_loss[0], stft_loss[1]
        amp_reg = self.loss(param[2],torch.zeros_like(param[2]))
        loss_total = self.cfg.mse_w*loss + self.cfg.stft_sc_w*stft_sc + self.cfg.stft_mag_w*stft_mag # TODO remove the hard coded values


        self.log_dict(
            {
                'loss_train/mse_loss': loss,
                'loss_train/stft_loss_spectral_convergence': stft_sc,
                'loss_train/stft_loss_magnitude': stft_mag,
                'loss_train/total_loss': loss_total,
            }, on_step=True, on_epoch=False, prog_bar=True
        )

        return {
            'loss': loss_total
        }


    def on_train_epoch_end(self):
        #all_preds = torch.stack(self.training_step_outputs)
        # do something with all preds
        #self.training_step_outputs.clear()  # free memory
        pass

    def validation_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred, latent, signal, params = self.model(batch)
        
        if self.D == 1:
            batch = batch[:, 0, :].reshape(batch.shape[0], self.N)
            pred = pred.reshape(pred.shape[0], self.N)
        else:
            pred = pred.reshape(pred.shape[0], self.D, self.N)
            
        loss = self.loss(batch, pred)
        stft_loss = self.stft_loss(batch, pred)
        metrics_mae = self.metric(batch, pred)
        stft_sc, stft_mag = stft_loss[0], stft_loss[1]
        loss_total = loss + stft_sc + stft_mag 
        
        # log scalars
        self.log_dict(
            {   
                'loss_val/mse_loss': loss,
                'loss_val/stft_loss_spectral_convergence':  stft_sc,
                'loss_val/stft_loss_magnitude': stft_mag,
                'metrics/MAE_val': metrics_mae,
                'loss_val/total_loss': loss_total,
            }, on_step=False, on_epoch=True
        )
        
        # log figures, randomly select one validation signal
        k = torch.randint(0, batch.shape[0], (1,)).item()
        x = batch[k, ...]
        y = pred[k, ...]
        selected_latent_signal = signal[k, ...]
        selected_params = [p[k, ...] for p in params]
        
        title_prefix = 'images_val'
        fig_reconstruction = self._visualize_reconstruction(x, y, selected_latent_signal)
        fig_latent_val = self._visualize_latent_values(selected_params)
        figure_log_spectogram = self._visualize_log_spectogram(x, y)
        self._log_figure(title=f"{title_prefix}/Reconstruction", caption="Input vs Reconstructed Signal", 
                         img=fig_reconstruction)
        self._log_figure(title=f"{title_prefix}/Latent Values", caption="Value Histogram of Learned Latent Parameters", 
                         img=fig_latent_val)
        self._log_figure(title=f"{title_prefix}/Log Spectogram", caption="Input and Output Log Spectograms", 
                         img=figure_log_spectogram)
        
    
    def on_validation_epoch_end(self):
        #all_preds = torch.stack(self.validation_step_outputs)
        # do something with all preds
        #self.validation_step_outputs.clear()  # free memory
        pass


    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred, _, _, _ = self.model(batch)
        metrics_mae = self.metric(batch, pred.reshape(pred.shape[0], self.D, self.N))
        
        for i in range(batch.shape[0]):
            if not self.device == 'cpu':
                act = batch[i,0].cpu().numpy()
                pre = pred[i].cpu().numpy()
            else:
                act = batch[i,0].numpy()
                pre = pred[i].numpy()
            
            if not self.saved_once:
                sf.write(f'save/actual_eps_{self.cfg.num_epochs}_{self.cfg.embedding_channels}.wav',act,16000)
                sf.write(f'save/pred_eps_{self.cfg.num_epochs}_emb_size_{self.cfg.embedding_channels}.wav', pre,16000)
                self.saved_once = True
        
        self.log_dict(
            {
                'metrics/MAE_test': metrics_mae
            }, on_step=False, on_epoch=True
        )
        
    def test(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred, _, _, _ = self.model(batch)
        metrics_mae = self.metric(batch, pred.reshape(pred.shape[0], self.D, self.N))
        
        for i in range(batch.shape[0]):
            if not self.device == 'cpu':
                act = batch[i,0].cpu().numpy()
                pre = pred[i].cpu().numpy()
            else:
                act = batch[i,0].numpy()
                pre = pred[i].numpy()
            
        return pred

    def test_end(self, outputs):
        return {}
    

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
    

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        #self.load_state_dict(checkpoint["model_state_dict"])


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
        
        
    def _visualize_reconstruction(self, x, y, signal) -> PIL.Image:
        """
            Plot input signal x against output signal y
        """
        assert len(x.shape) == len(y.shape) == 1, "int/out have to be flattened 1D tensors"
        assert len(signal.shape) == 2, "latent signal has to be a 2D tensor"
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        signal = signal.detach().cpu().numpy()
        
        T = 500 # only show first T steps

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, T+1), x[:T], label="Input Signal", color="blue")
        plt.plot(np.arange(1, T+1), y[:T], label="Reconstructed Signal", color="green")
        plt.xlabel("Steps")
        plt.ylabel("Signal")
        plt.title("Input vs Reconstructed Signal")
        plt.legend(loc="lower right")
        plt.grid()

        # Plot each of the k signals in a different color
        plt.subplot(1, 2, 2)
        for i in range(self.K):
            plt.plot(range(100), signal[i, :100], label=f"Signal {i+1}")
        plt.xlabel("Time/Index")
        plt.ylabel("Amplitude")
        plt.title(f"{self.K} Latent Signals")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc="lower left")
        
        img = self._fig_to_PIL_img(fig)
        plt.close()

        return img


    def _visualize_latent_values(self, params: tuple[torch.Tensor], n_bins: int = 100) -> PIL.Image:
        params = [p.detach().cpu().numpy().flatten() for p in params]
        p, f, a, b = params

        # Create subplots for histograms
        fig = plt.figure(figsize=(12, 8))

        # Phase (p)
        plt.subplot(2, 2, 1)
        plt.hist(p, bins=n_bins, color="blue", alpha=0.7, edgecolor="black")
        plt.title("Phase (p)")
        plt.xlabel("Phase")
        plt.ylabel("Frequency")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))


        # Frequency (f)
        plt.subplot(2, 2, 2)
        plt.hist(f, bins=n_bins, color="green", alpha=0.7, edgecolor="black")
        plt.title("Frequency (f)")
        plt.xlabel("Frequency")
        plt.ylabel("Frequency")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  


        # Amplitude (a)
        plt.subplot(2, 2, 3)
        plt.hist(a, bins=n_bins, color="red", alpha=0.7, edgecolor="black")
        plt.title("Amplitude (a)")
        plt.xlabel("Amplitude")
        plt.ylabel("Frequency")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  


        # Bias (b)
        plt.subplot(2, 2, 4)
        plt.hist(b, bins=n_bins, color="orange", alpha=0.7, edgecolor="black")
        plt.title("Bias (b)")
        plt.xlabel("Bias")
        plt.ylabel("Frequency")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  
        
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