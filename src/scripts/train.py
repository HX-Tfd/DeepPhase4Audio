import os
import shutil
import uuid

from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from src.utils.rules import check_all_rules
from src.utils.config import command_line_parser
from src.experiments.experiment import MockExperiment # don't remove this
from src.utils.helpers import get_device_accelerator 


def main():
    cfg = command_line_parser()

    # Remove previous logs 
    # if os.path.isdir(cfg.log_dir):
    #     shutil.rmtree(cfg.log_dir)

    # Resolve name task
    experiment_name = "MockExperiment" 
    model = eval(experiment_name)(cfg)

    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f'T{timestamp}_{cfg.name}_{str(uuid.uuid4())[:5]}'

    """
    Define callbacks
    """
    if cfg.logging:
        csv_logger = CSVLogger(
            save_dir='logs',
            name='Mock Experiment',
        )
        wandb_logger = WandbLogger(
            name=run_name,
            project='Deep Learning',
            )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, 'checkpoints'),
        save_last=False,
        save_top_k=1,
        monitor='metrics_MAE',
        mode='max',
    )

    # if this doesn't work (e.g. rich install fails) use TQDM progressbar instead (below)
    # rich_progress_bar = RichProgressBar( 
    #     theme=RichProgressBarTheme(
    #         description="green_yellow",
    #         progress_bar="green1",
    #         progress_bar_finished="green1",
    #         progress_bar_pulse="#6206E0",
    #         batch_progress="green_yellow",
    #         time="grey82",
    #         processing_speed="grey82",
    #         metrics="grey82",
    #         metrics_text_delimiter="\n",
    #         metrics_format=".3e",
    #     )
    # )

    tqdm_progress_bar = TQDMProgressBar(refresh_rate=10)



    """
    Define trainer
    """
    # Log information to wandb
    trainer = Trainer(
        logger=[wandb_logger, csv_logger] if cfg.logging else False,
        callbacks=[checkpoint_local_callback, tqdm_progress_bar],
        accelerator='cuda',#get_device_accelerator(preferred_accelerator='auto'),
        devices=1,
        default_root_dir=cfg.ckpt_save_dir, # directory to save checkpoints at every epoch end
        max_epochs=cfg.num_epochs,
        num_sanity_val_steps=1,
        precision=16 if cfg.optimizer_float_16 else 32,
        log_every_n_steps=50,
        # limit_train_batches=200,
        # limit_val_batches=10,
        # limit_test_batches=10,
        # log_every_n_steps=10,
    )

    # train and test the model
    trainer.fit(model, ckpt_path=cfg.resume)
    trainer.test(model)


if __name__ == '__main__':
    main()