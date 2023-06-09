from pathlib import Path
import builtins
import importlib
from typing import List, Any

import pandas as pd

def read_df(path: str) -> pd.DataFrame:
    """
    Read datafile at given path into a pandas DataFrame using the appropriate pandas.read_* function
    """
    match Path(path).suffix:
        case ".csv":
            return pd.read_csv(path)
        case ".parquet":
            return pd.read_parquet(path)
        case _:
            raise ValueError(f"Unrecognized file extension for path {path}")


def try_set_index(df: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
    """
    Try to set an index, but don't fail if the index is already set to the given index columns
    """
    if df.index.names != index_cols:
        df = df.set_index(index_cols)
    return df


def global_from_name(name: str) -> Any:
    """
    Get a global variable/class from a dotted string name.

    For example, if name is "kp.model.SKLearnModel",
    this will return the SKLearnModel class from the kp.model module.

    Parameters
    ----------
    name : str
        The dotted string name of the global variable/class to get
    
    Returns
    -------
    Any
    """
    pkgsplit = name.rsplit(".", 1)
    if len(pkgsplit) == 1:
        return getattr(builtins, name, None)
    try:
        pkg = importlib.import_module(pkgsplit[0])
    except ImportError:
        return None
    return getattr(pkg, pkgsplit[1], None)
