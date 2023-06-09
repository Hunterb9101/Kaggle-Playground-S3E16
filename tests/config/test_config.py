import pytest
import yaml

import kpe.config as kpc

@pytest.fixture
def sample_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    cfg = {
        "data": {
            "tag": "data",
        },
        "paths": {
            "dir": "path/to/dir",
            "dir2": "path/to/dir2",
        },
        "models": {
            "x": {},
            "y": {},
            "z": {},
        }
    }
    with open(config_file, "w") as f:
        f.write(yaml.safe_dump(cfg))
    yield config_file


def test_config_init():
    """ Force the config initialization to have a configuration file ready, but not have it loaded. """
    config = kpc.Config()
    assert config.config_file is not None
    assert config._tree is None


def test_config_tree(sample_config):
    """ Test that the configuration tree is loaded correctly. """
    config = kpc.Config(sample_config)
    assert "data" in config.tree
    assert "path" in config.tree


def test_immutable(sample_config):
    """ Test that the configuration tree is immutable. """
    config = kpc.Config(sample_config)
    tree = config.tree

    tree["new_data"] = "blah"
    assert "new_data" not in config.tree


def test_lookup(sample_config):
    config = kpc.Config(sample_config)
    assert config.lookup("data.tag") == "data"
    assert config.lookup("paths.dir") == "path/to/dir"
