import os
import uuid

from datetime import datetime

from sklearn.model_selection import ParameterGrid
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from src.utils.config import command_line_parser, yaml_config_parser
from src.experiments.pae_deep import PAEDeepModel # don't remove this
from src.experiments.paella import PAEllaModel
from src.experiments.pae_wave import PAEWaveModel
from src.experiments.pae_flattened import PAEInputFlattenedModel
from src.experiments.ae import AEModel
from src.experiments.vq_pae import VQ_PAEModel
from src.utils.helpers import get_device_accelerator 


def main():
    cfg = yaml_config_parser() # note that the namespaces are accessed differently
    seed_everything(42)
    # Resolve name task
    model_name = cfg.model_config.experiment_name
    model = eval(model_name)(cfg)

    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f'T{timestamp}_{cfg.metadata.run_name}_{str(uuid.uuid4())[:5]}'

    """
    Define callbacks
    """
    if cfg.metadata.logging:
        csv_logger = CSVLogger(
            save_dir='logs',
            name=cfg.experiment_name,
        )
        wandb_logger = WandbLogger(
            name=run_name,
            project='Deep Learning',
            config=cfg
        )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.metadata.log_dir, 'checkpoints'),
        save_last=False,
        save_top_k=1,
        monitor='loss_val/total_loss',
        mode='min',
    )

    # if this doesn't work (e.g. rich install fails) use TQDM progressbar instead (below)
    rich_progress_bar = RichProgressBar( 
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )
    early_stop_callback = EarlyStopping(monitor="loss_train/stft_loss_spectral_convergence", min_delta=0.00, patience=50, verbose=False, mode="max")

    trainer = Trainer(
        logger=[wandb_logger, csv_logger] if cfg.logging else False,
        callbacks=[checkpoint_local_callback, rich_progress_bar],
        accelerator=get_device_accelerator(preferred_accelerator='cuda'),
        devices=1,
        default_root_dir=cfg.ckpt_save_dir, 
        max_epochs=cfg.num_epochs,
        num_sanity_val_steps=1,
        precision=16 if cfg.optimizer_float_16 else 32,
        log_every_n_steps=10,
        profiler='simple',
        enable_checkpointing=True,
    )

    # train and test the model
    trainer.fit(model, ckpt_path=cfg.resume)
    print("saving model...")
    trainer.save_checkpoint(f'logs/models/{model_name}_epoch_{cfg.num_epochs}.ckpt',weights_only=False)
    print("...done")
    trainer.test(model)


if __name__ == '__main__':
    main()