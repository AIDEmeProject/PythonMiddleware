import yaml
from os.path import join
import logging.config

from definitions import ROOT_DIR

def get_path_to_config(filename):
    return join(ROOT_DIR, 'resources', filename)

def get_config_from_file(filename, section=None):
    """ Read a section from YAML configuration file """
    path = get_path_to_config(filename)

    with open(path, 'r') as yamlfile:
        cfg = yaml.safe_load(yamlfile)
        return cfg if section is None else cfg[section]

def setup_logging():
    """
        Setup logging configuration
    """
    config = get_config_from_file('logging.yml')
    logging.config.dictConfig(config)
