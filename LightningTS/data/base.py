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

        self.freq = self.preprocess_config["freq"]
        self.features = self.data_config["features"]
        self.targets = self.data_config["targets"]

        self.column_name = [*self.features, *self.targets]

        self.time_encoding = self.preprocess_config["time_encoding"]

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

    def _fit(self): ...

    def _transform(self): ...

    def save_parameter(self) -> None:

        log_dir = self.config["common"]["log_dir"]
        model_name = self.config["common"]["model_name"]
        model_version = self.config["common"]["model_version"]

        save_path = f"{log_dir}/{model_name}/{model_version}/transformer_param.pkl"

        dump(self._transformer, save_path)

    def load_parameter(self, filename) -> None:
        self._transformer = load(filename)
