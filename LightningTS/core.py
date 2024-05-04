import os
import rich
import torch

import lightning.pytorch as pl

from glob import glob

from .data import DataModule
from .models.PatchTST import PatchTST

from .utils import (
    config_callbacks,
    config_loggers,
    ConfigUtils,
    calculate_mae,
    calculate_mape,
)


def get_archive_path(config):
    if "CSVLogger" in config["loggers"]:
        return "/".join(config["loggers"]["CSVLogger"]["save_dir"])
    else:
        if "TensorBoardLogger" in config["loggers"]:
            return "/".join(config["loggers"]["TensorBoardLogger"]["save_dir"])
        else:
            raise ValueError("No logger exists!")


def save_recent_config(config):
    archive_path = get_archive_path(config)

    try:
        os.makedirs(archive_path)
    except FileExistsError as e:
        msg = (
            "Found log directory exists. Continue? (Contents will be overwrited) (y/n)"
        )
        if input(msg).lower() == "n":
            raise FileExistsError("Log directory exists! Abort!")
        else:
            pass

    ConfigUtils.save_config(config, f"{archive_path}/config.yaml")
    return archive_path


def get_latest_checkpoint(archive_path):
    return glob(f"{archive_path}/checkpoints/*.ckpt")[-1]


def config_model(config, pretrain=False, chkpt_dir=None, datamodule=None):

    architecture = config["model"]["model_architecture"]

    match architecture:
        case "PatchTST":
            datamodule.setup("fit")
            train_dataloader = datamodule.train_dataloader()

            model_func = PatchTST
            model_kwargs = {
                "config": config,
                "steps_per_epoch": len(train_dataloader),
            }
        case _:
            raise ValueError("Selected model architecture not available.")

    if pretrain:
        return model_func(chkpt_dir, is_fit=False, **model_kwargs)
    else:
        return model_func(**model_kwargs)


def preflight_config(config, stage="fit", pretrain=False, chkpt_dir=None):
    trainer_config = config["trainer"]

    trainer = pl.Trainer(
        max_epochs=trainer_config["max_epochs"],
        devices=trainer_config["accelerator"]["n_device"],
        accelerator=trainer_config["accelerator"]["type"],
        logger=config_loggers(config["loggers"], stage),
        callbacks=config_callbacks(config),
    )

    datamodule = DataModule(config=config)

    model = config_model(
        config, datamodule=datamodule, pretrain=pretrain, chkpt_dir=chkpt_dir
    )

    return trainer, datamodule, model


def train_process(
    config, run_validate: bool, run_test: bool, calc_metrics: bool
) -> None:

    archive_path = save_recent_config(config)

    trainer, model, datamodule = preflight_config(config, stage="fit")

    trainer.fit(model, datamodule=datamodule)

    if run_validate:
        saved_config = ConfigUtils.load_config(f"{archive_path}/config.yaml")
        chkpt_dir = get_latest_checkpoint(archive_path)

        trainer, model, datamodule = preflight_config(
            saved_config, stage="validate", pretrain=True, chkpt_dir=chkpt_dir
        )

        datamodule.setup("validate")

        trainer.validate(model, datamodule=datamodule)

    if run_test:
        saved_config = ConfigUtils.load_config(f"{archive_path}/config.yaml")
        chkpt_dir = get_latest_checkpoint(archive_path)

        trainer, model, datamodule = preflight_config(
            saved_config, stage="test", pretrain=True, chkpt_dir=chkpt_dir
        )

        datamodule.setup("test")

        trainer.validate(model, datamodule=datamodule)
