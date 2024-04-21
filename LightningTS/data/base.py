from typing import Any

from torch.utils.data import Dataset

from abc import (
    ABC,
    abstractmethod,
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)


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
        preprocessing_method = self.config["data"]["preprocessing"][
            "standardlize_method"
        ]
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
