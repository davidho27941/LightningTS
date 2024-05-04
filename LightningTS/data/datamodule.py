import torch
import numpy as np
import pandas as pd

from lightning.pytorch import LightningDataModule

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from .dataset import StructureDataset

from .io import CSVDataIO, ParquetDataIO


class DataModule(LightningDataModule):
    def __init__(
        self,
        config,
        **kwargs,
    ) -> None:

        self.config = config
        self.kwargs = kwargs

        self.data_config = config["data"]
        self.data_dir = config["data"]["directory"]
        self.data_format = config["data"]["format"]

        self.batch_size = config["common"]["batch_size"]
        self.num_workers = config["dataloader"]["num_worker"]

        self.if_apply_time_encoding = config["data"]["preprocessing"]["time_encoding"]

        self.train_dataloader_kargs = {
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "drop_last": False,
        }

        self.val_dataloader_kargs = {
            "shuffle": False,
            "batch_size": 1,
            "num_workers": self.num_workers,
            "drop_last": True,
        }

        self.test_dataloader_kargs = {
            "shuffle": False,
            "batch_size": 1,
            "num_workers": self.num_workers,
            "drop_last": True,
        }

    def load_data(self):
        match self.data_format:
            case "csv":
                IOModule = CSVDataIO(self.data_dir)
                IOModule.read_single()

                return IOModule.data

            case "parquet":
                IOModule = ParquetDataIO(self.data_dir)
                IOModule.read_single()

                return IOModule.data

            case _:
                raise ValueError("Provided data format is not supported.")

    def setup(self, stage):
        dataset = self.load_data()

        match stage:
            case "fit":

                self.train_dataset = StructureDataset(
                    dataset,
                    self.config,
                    stage,
                    need_label=self.data_config["need_label"],
                )

                self.val_dataset = StructureDataset(
                    dataset,
                    self.config,
                    stage,
                    need_label=self.data_config["need_label"],
                )

            case "validate":
                ...
            case "test":
                ...
            case "predict":
                ...
                
    def train_dataloader(self):
        """Return train dataloder."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            **self.train_dataloader_kargs,
        )

    def val_dataloader(self):
        """Return val dataloder."""

        return DataLoader(
            self.val_dataset,
            **self.val_dataloader_kargs,
        )

    def test_dataloader(self):
        """Return test dataloder."""

        return DataLoader(
            self.test_dataset,
            **self.test_dataloader_kargs,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            **self.test_dataloader_kargs,
        )

    @property
    def transformer(self):
        return self._transformer