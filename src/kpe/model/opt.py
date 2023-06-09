from typing import Dict, Any, Callable, List, Optional
import os
import logging
from uuid import uuid4

import pandas as pd
import sklearn
import optuna

from kpe.model import EnsembleLearner
import kpe.util as kpu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Optimizer:
    def __init__(
        self,
        model_class: sklearn.base.BaseEstimator,
        model_kwargs: Dict[str, Callable[..., Any]],
        model_kwarg_invariants: Dict[str, Any],
        save_dir: str,
        idx_cols: List[str],
        num_samples: int = 100,
        models_per_sample: int = 3,
    ):
        """
        Initialize the hyperparameter optimizer.

        Parameters
        ----------
        model_class : sklearn.base.BaseEstimator
            A valid model class that can be instantiated with the model_kwargs.
        model_kwargs : Dict[str, Callable[..., Any]]
            Expecting input that looks like:
            ```
            max_samples:
                func: np.random.uniform
                kwargs:
                low: 0.6
                high: 1.0
            ```
        model_kwarg_invariants : Dict[str, Any]
            A dictionary of kwargs that should be passed to the model constructor, but
            remain constant throughout the hyperparameter optimization process
        save_dir : str
            The directory to save the hyperparameter optimization results to.
        idx_cols: List[str]
            The list of columns to use as the index for the DataFrame. Required for the
            EnsembleLearner.
        num_samples : int, optional
            The number of hyperparameter samples to try, by default 100
        models_per_sample : int, optional
            The number of models to train in the EnsembleLearner, by default 3
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model_kwarg_invariants = model_kwarg_invariants
        self.num_samples = num_samples
        self.models_per_sample = models_per_sample
        self.save_dir = save_dir
        self.idx_cols = idx_cols
        self.__train_df: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Randomly search the hyperparameter space, and write models to disk.

        TODO: Might be good to have the model and highest score in a property.
        """
        self.__train_df = train_df
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.num_samples)

    def objective(self, trial: optuna.Trial) -> float:
        func_map = {
            "int": trial.suggest_int,
            "float": trial.suggest_float,
            "categorical": trial.suggest_categorical,
            "loguniform": trial.suggest_loguniform,
            "uniform": trial.suggest_uniform,

        }
        model_kwargs = self.model_kwarg_invariants.copy()
        for k, v in self.model_kwargs.items():
            func = v["func"]
            kwargs = v["kwargs"]
            model_kwargs[k] = func_map[func](name=k, **kwargs)

        sample_dir = os.path.join(self.save_dir, f"sample_{uuid4()}")
        os.makedirs(sample_dir, exist_ok=True)
        model = EnsembleLearner(
            model=self.model_class,
            num_models=self.models_per_sample,
            save_dir=sample_dir,
            model_kwargs=model_kwargs,
            idx_cols=self.idx_cols
        )
        model.train(self.__train_df)
        return model.score(istrain=False)
