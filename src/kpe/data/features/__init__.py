# This submodule is for performing tasks that require a train/test split (and the train/test split itself)
# such as categorical encoding, feature selection, filling NA's, etc.
# The result of the pipeline should be a dataset that requires ZERO (and I mean ZERO) work to plug into a
# machine learning model
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging

import pandas as pd

import lightgbm as lgb

from kp.data.features import FeatureSelectionPipeline
from kp.data.features.moment import MomentPruner
from kp.data.features.rfe import RFEPruner

import kpe.util as kputil
from kpe.data.pipeline import BasePipeline
import kpe.data.features.s3e16 as s3e16
import kpe.model as kpm

logger = logging.getLogger(__name__)


class FeaturePipeline(BasePipeline):
    def __init__(
            self,
            target_col: str,
            index_cols: List[str],
            istrain: bool = False
        ):
        """
        Initialize the feature pipeline.
        
        Parameters
        ----------
        target_col : str
            The name of the target column
        index_cols : List[str]
            The names of the index columns
        """
        super().__init__(index_cols)

        self.target_col = target_col
        self.istrain = istrain

        self.fsp = FeatureSelectionPipeline(
            [
                MomentPruner(max_corr=0.95),
                #RFEPruner(lgb.LGBMRegressor(), n_features_to_select=20, target_col=target_col),
            ]
        )

    def fit_df(self, df: pd.DataFrame) -> None:
        self.fsp.fit(df)

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with all columns (except the target column) transformed
        """
        df = self.fsp.transform(df)
        return df
