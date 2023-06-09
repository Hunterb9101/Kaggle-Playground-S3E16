# The model subpackage will include things that we need to train/predict using our models. While it might be weird
# to add a thin layer of abstraction between scikit-learn/lightgbm/xgboost/pytorch and the other scripts, I assure you
# that it is worth it. It also makes it really easy to train an ensemble of models.
from typing import Dict, Any, List
import pickle
import os
import logging

import pandas as pd

import sklearn
import sklearn.metrics as sm
from sklearn.model_selection import KFold
import yaml
from tqdm import tqdm

import kpe.util as kpu

logger = logging.getLogger(__name__)

class SKLearnModel:
    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        save_path: str,
        model_kwargs: Dict[str, Any],
        idx_cols: List[str],
        target_col: str,
    ):
        self.model_type = model
        self.model = None
        self.save_path = save_path
        self.model_kwargs = model_kwargs
        self.target_col = target_col
        self.idx_cols = idx_cols

    def train(self, df: pd.DataFrame) -> None:
        """
        Train this object's estimator using the given pandas DataFrame
        """
        self.model = self.model_type(**self.model_kwargs)

        df = kpu.try_set_index(df, self.idx_cols)
        self.model.fit(df.drop(columns=self.target_col), df[self.target_col])

        for attr in ["feature_importances_", "coef_", "feature_name_"]:
            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))

    def save(self):
        """
        Serialize this object's estimator and save to a file
        """
        with open(self.save_path, "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions for the given pandas DataFrame using this object's fitted estimator
        """
        df = df.copy()
        df = kpu.try_set_index(df, self.idx_cols)
        if self.target_col in df.columns:
            df = df.drop(columns=self.target_col)

        preds = self.model.predict(X=df)
        return preds


class EnsembleLearner:
    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        num_models: int,
        save_dir: str,
        model_kwargs: Dict[str, Any],
        idx_cols: List[str],
        target_col: str = "Age",
    ):
        self.model_type = model
        self.num_models = num_models
        self.models = []
        self.save_dir = save_dir
        self.model_kwargs = model_kwargs
        self.target_col = target_col
        self.idx_cols = idx_cols
        self._perfs = {"tr": [], "val": []}

    def train(self, df: pd.DataFrame):
        kf = KFold(n_splits=self.num_models, shuffle=True, random_state=42)

        for i, (train_idx, val_idx) in tqdm(enumerate(kf.split(df))):
            mdl = SKLearnModel(
                model=self.model_type,
                save_path=f"{self.save_dir}/model_{i}.pkl",
                model_kwargs=self.model_kwargs,
                idx_cols=self.idx_cols,
                target_col=self.target_col,
            )
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            mdl.train(train_df)
            self.models.append(mdl)
            y_pred_tr =  mdl.predict(train_df)
            y_pred_val = mdl.predict(val_df)

            self._perfs['tr'].append(float(sm.mean_absolute_error(train_df[self.target_col], y_pred_tr)))
            self._perfs['val'].append(float(sm.mean_absolute_error(val_df[self.target_col], y_pred_val)))

        perfs = {
            "overall": {
                "tr":  self.score(istrain=True),
                "val": self.score(istrain=False),
            }
        }
        perfs.update(self._perfs)

        with open(os.path.join(self.save_dir, "performance.yaml"), 'w+') as f:
            f.write(yaml.safe_dump(perfs))
        with open(os.path.join(self.save_dir, "hyperparams.yaml"), 'w+') as f:
            kwargs = self.model_kwargs.copy()
            # Convert any non-serializable objects to strings
            for k, v in kwargs.items():
                if not isinstance(v, (int, float, str, bool)):
                    kwargs[k] = str(v)
            f.write(yaml.safe_dump(kwargs))
        with open(os.path.join(self.save_dir, "model.pkl"), "wb") as f:
            pickle.dump(self, f)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions for the given pandas DataFrame using this object's fitted estimators
        """
        preds = pd.DataFrame(index=df.index)

        df = df.copy()
        if self.target_col in df.columns:
            df = df.drop(columns=self.target_col)

        for i, mdl in enumerate(self.models):
            preds[f"model_{i}"] = mdl.predict(df)
        preds["score"] = preds.median(axis=1)
        return preds["score"]

    def score(self, istrain: bool) -> float:
        """
        Return the average score of the models in the ensemble.

        Parameters
        ----------
        istrain : bool
            Whether to return the score on the training set or the validation set.

        Returns
        -------
        float
            The average score of the models in the ensemble.
        """
        flag = "tr" if istrain else "val"
        return sum(self._perfs[flag]) / len(self.models)
