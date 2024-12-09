import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate

from src.datasets.definitions import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VALID
from src.utils.metrics import simple_metric
from src.utils.helpers import resolve_dataset_class, resolve_lr_scheduler, resolve_model_class, resolve_optimizer


class PAEInputFlattenedModel(pl.LightningModule):

    def __init__(self, cfg, **kwargs) -> None:
        super(PAEInputFlattenedModel, self).__init__()
        self.cfg = cfg

        self.D, self.N = cfg.input_channels, cfg.time_range
        dataset_class = resolve_dataset_class(cfg.dataset)
        self.datasets = {
            split: dataset_class(dataset_root=cfg.dataset_root, # dataset root is not important here
                                 split=split) 
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')


        self._instantiate_model()
        self.loss = nn.MSELoss()
        self.metric = simple_metric


    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred, _, _, _ = self.model(batch) # output is (y, latent, signal, param)
        loss = self.loss(batch, pred.reshape(pred.shape[0], self.D, self.N))
        self.log_dict(
            {
                'loss_train/mse_loss': loss
            }, on_step=True, on_epoch=False, prog_bar=True
        )

        return {
            'loss': loss
        }


    def on_train_epoch_end(self):
        #all_preds = torch.stack(self.training_step_outputs)
        # do something with all preds
        #self.training_step_outputs.clear()  # free memory
        pass


    def validation_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            batch.to(self.device)
        pred, _, _, _ = self.model(batch)
        metrics_mae = self.metric(batch, pred.reshape(pred.shape[0], self.D, self.N))

        self.log_dict(
            {
                'metrics_MAE_val': metrics_mae
            }, on_step=False, on_epoch=True
        )


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

        self.log_dict(
            {
                'metrics_MAE_test': metrics_mae
            }, on_step=False, on_epoch=True
        )


    def test_end(self, outputs):
        return {}
    

    def train_dataloader(self):
        return self._create_train_dataloader()


    def val_dataloader(self):
        return self._create_val_test_dataloader(SPLIT_VALID)


    def test_dataloader(self):
        return self._create_val_test_dataloader(SPLIT_TEST)
    

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.cfg, self.model.parameters())
        lr_scheduler = resolve_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [lr_scheduler]
    

    def on_load_checkpoint(self, checkpoint):
        print("Custom logic when loading checkpoint")
        # modify cfg or other attributes if needed
        # self.cfg.some_param = checkpoint["hyper_param"]["some_param"]


    def _create_train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
        )

    def _create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.cfg.batch_size_validation,
            shuffle=False,
            num_workers=self.cfg.workers_validation,
            pin_memory=True,
            drop_last=False,
        )
    
    def _inference_step(self, batch):
        pass


    def _instantiate_model(self):
        self.model = resolve_model_class(self.cfg.model_name, self.cfg)
        if torch.cuda.is_available():
            self.model.to(self.device)
            