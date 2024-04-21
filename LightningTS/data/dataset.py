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
        data: pd.DataFrame,
        config: Dict[str, Any],
        stage: str,
        **kwargs: Dict[str, Any],
    ) -> None:

        super(StructureDataset, self).__init__(
            data,
            config,
        )

        self.data = data
        self.config = config
        self.stage = stage
        self.kwargs = kwargs

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

    def __prepare__(self):
        data, timestamp = self.get_data(self.stage)

        self.__prepare__transformer__()

