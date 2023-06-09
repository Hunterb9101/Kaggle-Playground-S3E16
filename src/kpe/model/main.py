import logging
import os
import pickle

import pandas as pd
import yaml

from kpe.config import cfg
import kpe.model as kpm
import kpe.util as kpu
import kpe.model.opt as kpo

logger = logging.getLogger(__name__)

def train_model():
    """
    Fit an ensemble of models on the training data
    """
    for m in cfg.lookup("models").keys():
        el = kpm.EnsembleLearner(
            model=kpu.global_from_name(name=cfg.lookup(f"models.{m}.type")),
            num_models=cfg.lookup(f"models.{m}.num_models"),
            save_dir=cfg.lookup(f"path.{m}_model_dir"),
            model_kwargs=cfg.lookup(f"models.{m}.kwargs"),
            idx_cols=cfg.lookup("data.idx_cols"),
            target_col=cfg.lookup("data.target_col"),
        )
        # If this was a normal case, we'd load the following file. However, since we are doing a CV split
        # we will need to pass the raw data path, and create a variety of pipelines.
        # cfg.lookup("path.dmatrix_tr")
        load_path = cfg.lookup("path.dmatrix_tr")
        logging.debug("Training ensemble learner with data from %s", load_path)
        df = kpu.read_df(load_path)
        el.train(df)

        with open(cfg.lookup(f"path.{m}_performance_path"), "r") as f:
            logging.info("MAE-%s: %s", m, yaml.safe_load(f)["overall"])


def score_model():
    """
    Load a fitted, serialized ensemble model and make predictions for the training and testing datasets
    """
    for m in cfg.lookup("models").keys():
        with open(os.path.join(cfg.lookup(f"path.{m}_model_dir"), "model.pkl"), "rb") as f:
            el = pickle.load(f)

        for flag in ["tr", "te"]:
            df = kpu.read_df(cfg.lookup(f"path.dmatrix_{flag}"))

            df_out = pd.DataFrame(index=df.index)
            df_out[cfg.lookup("data.target_col")] = el.predict(df)

            df_out.to_csv(cfg.lookup(f"path.{m}_{flag}_pred"), index=True)


def optimize():
    opt = kpo.Optimizer(
        num_samples = 100,
        model_class=kpu.global_from_name(name=cfg.lookup("models.lgbm.type")),
        save_dir=cfg.lookup("path.lgbm_opt_dir"),
        model_kwargs=cfg.lookup("models.lgbm.opt.model"),
        model_kwarg_invariants=cfg.lookup("models.lgbm.opt.model_constants"),
        idx_cols=cfg.lookup("data.idx_cols"),
    )

    opt.fit(kpu.read_df(cfg.lookup("path.dmatrix_tr")))
    logger.info("Sample %s score: %s", str(opt.study.best_params), format(opt.study.best_value, ".3f"))
