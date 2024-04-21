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

from sklearn.externals import joblib


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

    def config_transformer(self):
        preprocessing_method = self.config["data"]["preprocessing"]["normalize_method"]
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

        joblib.dump(self._transformer, save_path)

    def load_parameter(self, filename) -> None:
        self._transformer = joblib.load(filename)
