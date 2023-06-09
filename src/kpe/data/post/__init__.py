from typing import List, Optional
import logging

import pandas as pd
import sklearn.metrics as sm
import kp.data.post.blend as kpb

import kpe.util as kputil

logger = logging.getLogger(__name__)


class PostPipeline:
    def __init__(
        self,
        models: List[str],
        idx_cols: List[str],
        target_col: str
    ):
        self.models = models
        self.idx_cols = idx_cols
        self.target_col = target_col
        self.blender = kpb.Blender(
            weights=[1 for _ in models],
            cols=[self._col_name(m) for m in models],
            out_col=target_col
        )

    def _col_name(self, model: str) -> str:
        return f"{self.target_col}_{model}"

    def fit(self, score_paths: List[str], y: Optional[pd.DataFrame] = None) -> None:
        df = self.create_blend_df(score_paths, self.models)
        logger.info(f"\n%s", str(df.drop(columns=self.idx_cols).corr()))

    def transform(self, score_paths: List[str], y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = self.create_blend_df(score_paths, self.models)
        out = self.blender.transform(df)[self.idx_cols + [self.target_col]].copy()
        df = df.astype("int32")
        out = out.astype("int32")

        if y is not None:
            m_errs = {}
            for m in self.models:
                m_errs[m] = sm.mean_absolute_error(y, df[self._col_name(m)])
            logger.info(m_errs)
            err = sm.mean_absolute_error(y, out[self.target_col])
            logger.info("Overall: %s", err)
        return out

    def fit_transform(self, score_paths: List[str], y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.fit(score_paths, y)
        return self.transform(score_paths, y)


    def create_blend_df(self, score_paths: List[str], models: List[str]) -> pd.DataFrame:
        out = kputil.read_df(score_paths[0]).rename(columns={self.target_col: self._col_name(models[0])})
        dfs = [kputil.read_df(p).rename(columns={self.target_col: self._col_name(m)})[self._col_name(m)]
            for m, p in zip(models[1:], score_paths[1:])
        ]
        out = pd.concat([out] + dfs, axis=1)
        return out
