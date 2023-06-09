from abc import ABC, abstractmethod
from typing import List
import logging

import pandas as pd

import kpe.util as kputil

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    def __init__(self, index_cols: List[str]):
        self.index_cols = index_cols

    def fit(self, input_dir: str) -> None:
        """
        Fit the feature pipeline to the data. This should be done on the training data only.
        """
        df = kputil.read_df(input_dir)
        if set(df.index.names) != set(self.index_cols):
            df = df.set_index(self.index_cols)
        self.fit_df(df)

    @abstractmethod
    def fit_df(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, input_dir: str, output_dir: str) -> None:
        """
        Read data in from input_dir, transform the data, and save to output_dir
        """
        df = kputil.read_df(input_dir)
        if set(df.index.names) != set(self.index_cols):
            df = df.set_index(self.index_cols)
        df = self.transform_df(df)
        df.to_parquet(output_dir)

    @abstractmethod
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, input_dir: str, output_dir: str) -> None:
        self.fit(input_dir)
        self.transform(input_dir, output_dir)
