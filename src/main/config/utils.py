import logging.config
import yaml
from pathlib import Path
from os.path import realpath


def get_config_from_file(filename, section=None):
    """ Read a section from YAML configuration file """
    path = Path(realpath(__file__)).resolve().parents[3] / 'resources'

    with open(path / filename, 'r') as yamlfile:
        cfg = yaml.load(yamlfile)
        if section is None:
            return cfg
        return cfg[section]


def setup_logging(path):
    """
        Setup logging configuration
    """
    with open(path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
