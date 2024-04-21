import os

from typing import Any, Callable

from torch.utils.data import Dataset

from abc import (
    ABC,
    abstractmethod,
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)

from joblib import dump, load

model_need_label = ["Autoformer"]


class BaseDataIO(ABC):
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    @abstractmethod
    def read(self):
        raise NotImplementedError(
            "A child class of `BaseDataIO` must implement a `read` method."
        )


class BaseDataset(Dataset, ABC):
    def __init__(self, data, config):
        self.data = data
        self.config = config

        self.data_config = config["data"]
        self.preprocess_config = config["data"]["preprocess"]
        self.time_series_config = config["data"]["series_config"]

        self.freq = self.preprocess_config["freq"]
        self.features = self.data_config["features"]
        self.targets = self.data_config["targets"]

        self.column_name = [*self.features, *self.targets]

        self.time_encoding = self.preprocess_config["time_encoding"]

        self.need_label = (
            True
            if self.config["model"]["model_architecture"] in model_need_label
            else False
        )

        self.preprocess_method = (
            None
            if self.preprocess_config["normalize_method"] == "None"
            else self.preprocess_config["normalize_method"]
        )

        self.param_path = (
            None
            if self.preprocess_config["normalize_param"] == "None"
            else os.path.join(*self.preprocess_config["normalize_param"])
        )

    def config_transformer(self):
        preprocessing_method = self.preprocess_config["normalize_method"]
        match preprocessing_method:
            case "MinMax":
                self._transformer = MinMaxScaler()
            case "StandardScalar":
                self._transformer = StandardScaler()
            case _:
                raise ValueError(
                    f"Selected standardlize method: {preprocessing_method} is not available."
                )

    @abstractmethod
    def __prepare__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    def transformer(self) -> Callable:
        return self._transformer

    def _fit(self, data):
        self._transformer = self._transformer.fit(data)

    def _transform(self, data):
        return self._transformer.transform(data)

    def _inverse_transform(self, data):
        return self._transformer.inverse_transform(data)

    def save_parameter(self) -> None:

        log_dir = self.config["common"]["log_dir"]
        model_name = self.config["common"]["model_name"]
        model_version = self.config["common"]["model_version"]

        save_path = (
            f"{log_dir}/{model_name}/{model_version}/transformer_param.pkl"
            if self.param_path is None
            else self.param_path
        )

        dump(self._transformer, save_path)

    def load_parameter(self, filename) -> None:
        self._transformer = load(filename)
