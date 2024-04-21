import os

import pandas as pd

from glob import glob

from .base import BaseDataset

from typing import (
    Dict,
    Any,
    Union,
    Tuple,
)


class StructureDataset(BaseDataset):
    def __init__(
        self,
        raw_data: pd.DataFrame,
        config: Dict[str, Any],
        stage: str,
        **kwargs: Dict[str, Any],
    ) -> None:

        super(StructureDataset, self).__init__(
            raw_data,
            config,
        )

        self.data = raw_data
        self.config = config
        self.stage = stage
        self.kwargs = kwargs

        self.features = self.config["data"]["features"]
        self.targets = self.targets["data"]["targets"]

        self.column_name = [*self.features, *self.targets]

        self.time_encoding = self.config["data"]["preprocessing"]["time_encoding"]

        self.__prepare__()

    def __prepare__data__(self, stage): ...

    def __prepare__transformer__(self):

        param_path = self.config["data"]["preprocessing"]["normalize_param"]

        if os.path.isfile(param_path):
            self.is_transform_fitted = True
            self.load_parameter()
        else:
            self.is_transform_fitted = False
            self.config_transformer()

    def __apply_transform__(self, data):
        if self.config["data"]["preprocessing"]["normalize_method"] is not None:
            self.__prepare__transformer__()

            if self.is_transform_fitted:
                data_transformed = pd.DataFrame(self._transform(data))
                self.data_processed = data_transformed[self.column_name].values
            else:
                train_data, train_ts = self.__prpare_data__(self.stage)
                self._fit(train_data)
                data_transformed = pd.DataFrame(self._transform(data))
                self.data_processed = data_transformed[self.column_name].values

        else:
            self.data_processed = self.data.values

    def __apply_time_encoding__(self): ...

    def __prepare__(self):
        data, timestamp = self.get_data(self.stage)

        self.__apply_transform__(data)

        posix_ts = timestamp["timestamp"].map(lambda ts: ts.timestamp()).to_list()

        if self.time_encoding:
            ...
        else:
            ...
