import os

from lightning.pytorch.loggers import (
    CSVLogger,
    TensorBoardLogger,
)


def config_csv_logger(config, stage):
    csv_config = config.copy()
    csv_save_dir = os.path.join(
        *csv_config["save_dir"],
    )
    del csv_config["save_dir"]

    # Since CSV logger cannot config subdir, so we need to define by ourselves.
    if stage != "fit":
        csv_config["version"] = f"{csv_config['version']}/{stage}"

    csv_logger = CSVLogger(
        save_dir=csv_save_dir,
        **csv_config,
    )
    return csv_logger


def config_tensorboard_logger(config, stage):
    tb_config = config.copy()
    tb_save_dir = os.path.join(
        *tb_config["save_dir"],
    )
    del tb_config["save_dir"]
    tb_logger = TensorBoardLogger(
        save_dir=tb_save_dir,
        sub_dir=None if stage == "fit" else stage,
        **tb_config,
    )
    return tb_logger


def config_loggers(config, stage="fit"):
    loggerList = []

    if "CSVLogger" in config["loggers"]:
        loggerList.append(config_csv_logger(config["loggers"]["CSVLogger"], stage))

    if "TensorBoardLogger" in config["loggers"]:
        loggerList.append(
            config_tensorboard_logger(config["loggers"]["TensorBoardLogger"], stage)
        )

    return loggerList
