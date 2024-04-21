import os
import torch

import numpy as np
import pandas as pd

from glob import glob

from .base import BaseDataset
from .time_encoding import TimeEncoding

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

        self.input_length = self.time_series_config["input_length"]
        self.label_length = self.time_series_config["label_length"]
        self.output_length = self.time_series_config["output_length"]

        self.__prepare__()

    def __prepare__data__(self, stage): ...

    def __prepare__transformer__(self) -> None:

        param_path = self.preprocess_config["normalize_param"]

        if os.path.isfile(param_path):
            self.is_transform_fitted = True
            self.load_parameter()
        else:
            self.is_transform_fitted = False
            self.config_transformer()

    def __apply_transform__(self, data: pd.DataFrame) -> np.ndarray:
        if self.preprocess_config["normalize_method"] is not None:
            self.__prepare__transformer__()

            if self.is_transform_fitted:
                data_transformed = pd.DataFrame(self._transform(data))
                data_processed = data_transformed[self.column_name].values
            else:
                train_data, train_ts = self.__prpare_data__(self.stage)
                self._fit(train_data)
                data_transformed = pd.DataFrame(self._transform(data))
                data_processed = data_transformed[self.column_name].values

        else:
            data_processed = self.data.values

        return data_processed

    def __apply_time_encoding__(self, timestamp: pd.DataFrame) -> np.ndarray:
        if self.time_encoding:
            timestamp_encoded = TimeEncoding.time_features(
                pd.to_datetime(timestamp)["timestamp"].values,
                freq=self.freq,
            )
            timestamp_encoded = timestamp_encoded.transpose(1, 0)

        else:
            timestamp["month"] = timestamp.datetime.apply(lambda row: row.month, 1)
            timestamp["day"] = timestamp.datetime.apply(lambda row: row.day, 1)
            timestamp["weekday"] = timestamp.datetime.apply(lambda row: row.weekday, 1)
            timestamp["hour"] = timestamp.datetime.apply(lambda row: row.hour, 1)
            timestamp["minute_decimal"] = timestamp.datetime.apply(
                lambda row: row.minute, 1
            )
            timestamp["minute"] = timestamp.minute_decimal.map(lambda x: x // 15)
            timestamp = timestamp.drop(columns=["date", "minute_decimal"])

            timestamp_encoded = timestamp.values

        return timestamp_encoded

    def __prepare__(self):
        data, timestamp = self.get_data(self.stage)

        data_processed = self.__apply_transform__(data)

        self.timestamp = self.__apply_time_encoding__(timestamp)

        posix_ts = timestamp["timestamp"].map(lambda ts: ts.timestamp()).to_list()

        self.input = data_processed
        self.label = data_processed

        self.posix_ts = posix_ts

    def __len__(self) -> int:

        return len(self.input) - self.sequence_length - self.predict_length + 1

    def __getitem__(self, index) -> Any:
        input_begin = index
        input_end = index + self.input_length

        sqeuence_input = torch.FloatTensor(self.input[input_begin:input_end])
        sqeuence_input_mask = torch.FloatTensor(self.timestamp[input_begin:input_end])

        if self.need_label:
            label_begin = input_end - self.label_length
            label_end = label_begin + self.label_length + self.output_length

            sqeuence_label = torch.FloatTensor(self.input[label_begin:label_end])
            sqeuence_label_mask = torch.FloatTensor(
                self.timestamp[label_begin:label_end]
            )
        else:
            sqeuence_label = torch.zeros(self.label_length)
            sqeuence_label_mask = torch.zeros(self.label_length)

        posix_ts = torch.LongTensor([self.posix_ts[index]])

        return (
            sqeuence_input,
            sqeuence_label,
            sqeuence_input_mask,
            sqeuence_label_mask,
            posix_ts,
        )
