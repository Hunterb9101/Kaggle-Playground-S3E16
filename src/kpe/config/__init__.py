from typing import Any, Dict, Optional
import importlib.resources
import os
import copy

import yaml

class Config:
    def __init__(self, config_file: Optional[str] = None, home_dir: Optional[str] = None):
        """
        Initialize the configuration object.

        Parameters
        ----------
        config_file : Optional[str], optional
            The path to the configuration file. If None, the src/kp/config/config.yaml file is used.
        home_dir : Optional[str], optional
            The path to the home directory. If None, the src/ directory  is used
        """
        # TODO: Remove lazy load stuff, and just use the property to lazily load the tree.
        self._tree: Optional[Dict[str, Any]] = None
        self.config_file = config_file or importlib.resources.files('kpe') / 'config' / 'config.yaml'
        self.home_dir = home_dir or importlib.resources.files('kpe') / '..'

    @property
    def tree(self):
        """
        Returns a deep copy of the configuration tree. This is to prevent accidental modification of the
        configuration tree, and force it to be static.
        """
        if self._tree is None:
            self._lazy_load()
        return copy.deepcopy(self._tree)

    def _lazy_load(self):
        with open(self.config_file) as f:
            self._tree = yaml.safe_load(f)

        data_tag = self.lookup("data_tag", "default")
        model_tag = self.lookup("model_tag", "default")
        self._tree["path"] = self.compute_paths(data_tag=data_tag, model_tag=model_tag)

        for path in self._tree["path"].values():
            # Make all the paths up front
            write_path = path if os.path.isdir(path) else os.path.dirname(path)
            os.makedirs(write_path, exist_ok=True)

    def lookup(self, key: str, default: Any = None) -> Any:
        """
        Look up a key in the configuration tree using dotted syntax. If the key is not found, return the default
        value.

        Parameters
        ----------
        key : str
            The key to look up in the configuration tree. This is a dotted path, e.g. "path.to.key".
        default : Any, optional
            The default value to return if the key is not found. Defaults to None.
        
        Returns
        -------
        Any
        """
        keys = key.split(".")
        tree = self.tree
        for k in keys:
            if k not in tree:
                return default
            tree = tree[k]
        return tree

    def compute_paths(self, data_tag: str, model_tag: str) -> Dict[str, str]:
        """
        Computes relevant data paths to add to the overall configuration. This is in a python function due to
        the amount of repeated path parts.

        Parameters
        ----------
        data_tag : str
            The tag to use for the data paths.
        model_tag : str
        """
        home_dir = importlib.resources.files('kpe') / '..' / '..' / 'out'
        _path_cfg = {}

        # Input data paths
        _path_cfg["input_tr_data"] = os.path.join(home_dir, "train.csv")
        _path_cfg["input_te_data"] = os.path.join(home_dir, "test.csv")
        _path_cfg["original_data"] = os.path.join(home_dir, 'WildBlueBerryPollinationSimulationData.csv')

        # Preprocessed data paths
        data_dir = os.path.join(home_dir, 'data', data_tag)
        for flag in ["tr", "val", "te"]:
            _path_cfg[f"raw_{flag}"] = os.path.join(data_dir, f"raw_{flag}.parquet")
            _path_cfg[f"dmatrix_{flag}"] = os.path.join(data_dir, f"dmatrix_{flag}.parquet")
            _path_cfg[f"dmatrix_{flag}_pred"] = os.path.join(
                data_dir, "predictions", model_tag, f"dmatrix_{flag}_pred.csv"
            )

        # Pipeline paths
        _path_cfg["raw_pipeline"] = os.path.join(data_dir, "raw_pipeline.pkl")
        _path_cfg["fsp_pipeline"] = os.path.join(data_dir, "fsp_pipeline.pkl")

        models_dir = os.path.join(home_dir, 'model', model_tag)

        for m in self.lookup("models").keys():
            model_dir = os.path.join(models_dir, m)
            _path_cfg[f"{m}_model_dir"] = model_dir
            _path_cfg[f"{m}_opt_dir"] = os.path.join(model_dir, "opt")
            _path_cfg[f"{m}_performance_path"] = os.path.join(model_dir, "performance.yaml")

            for flag in ["tr", "val", "te"]:
                _path_cfg[f"{m}_{flag}_pred"] = os.path.join(model_dir, f"{m}_{flag}_pred.csv")
        return _path_cfg

cfg = Config()
