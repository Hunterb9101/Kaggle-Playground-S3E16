import argparse
import sys
import logging
import json

from kpe.config import cfg
from kpe.data.main import compute_train_dmatrices, compute_serve_dmatrices, postprocess
from kpe.model.main import train_model, score_model, optimize

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def main():
    """
    Main entry point for the application. Any function in this module can be called by passing its name through
    the --action argument.

    For example, to run the `prepare_data` function, run the following command:
    `python -m kp.main --action prepare_data`
    """
    args = parse_args()
    logger.debug("Loading configuration:\n%s", json.dumps(cfg.tree, indent=4))
    globals()[args.action]()


def parse_args() -> argparse.Namespace:
    """
    Parse arguments from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=dir(sys.modules[__name__]), help="The action to perform.", required=True)
    args = parser.parse_args()
    return args


def debug():
    logger.debug("Debug")


def prepare_data():
    """
    Send data through the data pipeline to prepare it for modeling.
    """
    compute_train_dmatrices()
    logger.debug("Prepared data")


def opt():
    """
    Run hyperparameter optimization with the configuration specified in config.yaml.
    """
    optimize()
    logger.debug("Optimized model")


def fit_model():
    """
    Fit an EnsembleLearner with the configuration specified in config.yaml.
    """
    train_model()
    logger.debug("Fitted model")


def score():
    """
    Run predictions on the training and testing datasets.
    """
    compute_serve_dmatrices()
    score_model()
    logger.debug("Scored model")
    postprocess()
    logger.debug("Finished postprocessing")


if __name__ == "__main__":
    main()
