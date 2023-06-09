from typing import List, Optional, Tuple, TypedDict

import pandas as pd
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import sklearn.decomposition as skd

import kp.data.raw.interaction as kpint
import kp.data.raw.encoder as kpenc
import kp.data.raw.outliers as kpout

from kpe.data.pipeline import BasePipeline
import kpe.data.features.s3e16 as s3e16

class ModelData(TypedDict):
    float_df: pd.DataFrame
    cat_df: pd.DataFrame
    y: Optional[pd.Series]


class RawDataPipeline(BasePipeline):
    def __init__(
            self,
            target_col: str,
            idx_cols: List[str],
            cat_cols: Optional[List[str]] = None,
            standardize: bool = False,
            istrain: bool = True
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
        super().__init__(idx_cols)

        self.target_col = target_col
        self.index_cols = idx_cols
        self.cat_cols = cat_cols or []
        self.standardize = standardize
        self.istrain = istrain

        # Encoders/Data Augmentors
        self.cf = s3e16.CustomFeatures(ratio=self._ratios())
        self.ss = skp.StandardScaler()
        #self.pca = skd.PCA(n_components=12)
        self.bins = skp.KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile")

        # Categorical Encoding
        self.te = kpenc.TargetEncoder(columns=cat_cols, target_col=target_col)

    def _ratios(self) -> List[Tuple[str, str]]:
        """
        Returns a list of tuples of the form (numerator, denominator) for necessary columns
        """
        ratios = []

        for x in ["Viscera Weight", "Shell Weight", "Shucked Weight"]:
            ratios.append((x, "Weight"))
            ratios.append(("Weight", x))
        ratios.append(("Length", "Diameter"))
        return ratios

    def fit_df(self, df: pd.DataFrame) -> None:
        if not self.istrain:
            raise ValueError("Cannot fit on a test dataset")
        self._shared_fit_transform(df, is_fit=False)

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with all columns (except the target column) transformed
        """
        df_parts  = self._shared_fit_transform(df, is_fit=True)
        assert self.target_col not in df_parts["cat_df"].columns
        df = pd.concat((x for x in df_parts.values()), axis=1)
        return df

    def _shared_fit_transform(self, df: pd.DataFrame, is_fit: bool = True) -> ModelData:
        df = self.cf.transform(df)
        df_parts = self._split_df_to_parts(df)

        df_parts["float_df"] = self.clean_float_cols(df_parts["float_df"], is_fit=is_fit)

        # Do something with categoricals
        transform_call = "transform" if is_fit else "fit_transform"
        if not self.istrain and not is_fit:
            raise ValueError("Cannot fit target encoder when not training")
        df_parts["cat_df"] = getattr(self.te, transform_call)(
            pd.concat([df_parts["cat_df"], df_parts["y"]], axis=1)
        )
        if self.istrain:
            df_parts["cat_df"] = df_parts["cat_df"].drop(columns=self.target_col)

        return df_parts

    def _split_df_to_parts(
            self,
            df: pd.DataFrame
    ) -> ModelData:
        target = [self.target_col] if self.istrain else []
        # Split features into groups
        float_df = df.drop(columns=target+self.cat_cols)
        cat_df = df[self.cat_cols]
        target_df = df[self.target_col] if self.istrain else None

        return ModelData(float_df=float_df, cat_df=cat_df, y=target_df)

    def clean_float_cols(self, float_df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        transform_call = "transform" if is_fit else "fit_transform"

        # Standardize and compute PCA
        float_feats = getattr(self.ss, transform_call)(float_df)
        float_df = pd.DataFrame(float_feats, columns=float_df.columns, index=float_df.index)
        bin_feats = getattr(self.bins, transform_call)(float_df)
        bin_df = pd.DataFrame(
            bin_feats.toarray(),
            columns=[f"bin_{i}" for i in range(bin_feats.shape[1])], index=float_df.index
        )
        #pca_feats = getattr(self.pca, transform_call)(float_df)
        #pca_df = pd.DataFrame(pca_feats, columns=[f"pca_{i}" for i in range(pca_feats.shape[1])], index=float_df.index)
        float_df = pd.concat([float_df,  bin_df], axis=1) # pca_df,

        return float_df
