import logging
import pickle

from kpe.data.raw import RawDataPipeline
import kpe.data.features as kpfeat
from kpe.data.post import PostPipeline
from kpe.config import cfg
import kpe.util as kputil

logger = logging.getLogger(__name__)

def compute_train_dmatrices():
    """
    Transform the training and testing datasets and save them as files for later use in model development
    """
    # Removing this, as we probably want to make use of 100% of data to train models
    # kpfeat.train_val_split(
    #     input_path=cfg.lookup("path.input_data"),
    #     output_paths=(cfg.lookup("path.raw_tr_data"), cfg.lookup("path.raw_val_data")),
    #     val_size=cfg.lookup("data.val_size"),
    # )
    rdp = RawDataPipeline(
        target_col=cfg.lookup("data.target_col"),
        idx_cols=cfg.lookup("data.idx_cols"),
        cat_cols=cfg.lookup("data.cat_cols"),
        standardize=True,
        istrain=True
    )

    fp = kpfeat.FeaturePipeline(
        cfg.lookup('data.target_col'),
        cfg.lookup('data.idx_cols'),
        istrain=True
    )

    rdp.fit_transform(
        input_dir=cfg.lookup("path.input_tr_data"),
        output_dir=cfg.lookup("path.raw_tr")
    )

    fp.fit_transform(
        input_dir=cfg.lookup("path.raw_tr"),
        output_dir=cfg.lookup("path.dmatrix_tr"),
    )

    with open(cfg.lookup("path.raw_pipeline"), 'wb') as f:
        pickle.dump(rdp, f)
    with open(cfg.lookup("path.fsp_pipeline"), 'wb') as f:
        pickle.dump(fp, f)

def compute_serve_dmatrices():
    with open(cfg.lookup("path.raw_pipeline"), 'rb') as f:
        rdp = pickle.load(f)
    with open(cfg.lookup("path.fsp_pipeline"), 'rb') as f:
        fp = pickle.load(f)

    rdp.istrain = False
    fp.istrain = False

    import pandas as pd
    rdp.transform(cfg.lookup("path.input_te_data"), cfg.lookup("path.raw_te"))
    fp.transform(cfg.lookup("path.raw_te"), cfg.lookup("path.dmatrix_te"))

def postprocess():
    post = PostPipeline(
        models=list(cfg.lookup("models").keys()),
        idx_cols=cfg.lookup("data.idx_cols"),
        target_col=cfg.lookup("data.target_col")
    )

    post.fit_transform(
        score_paths=[cfg.lookup(f"path.{m}_tr_pred") for m in cfg.lookup("models").keys()],
        y=kputil.read_df(cfg.lookup("path.dmatrix_tr"))[cfg.lookup("data.target_col")]
    )

    out = post.transform(
        score_paths=[cfg.lookup(f"path.{m}_te_pred") for m in cfg.lookup("models").keys()],
    )

    out.to_csv(cfg.lookup("path.dmatrix_te_pred"), index=False)
