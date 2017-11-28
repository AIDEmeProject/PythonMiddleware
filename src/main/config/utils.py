import logging.config
import os
import yaml

path = os.path.dirname(os.path.realpath(__file__))
path_to_file = path + '/{0}'


def get_config_from_file(filename, section=None):
    """ Read a section from YAML configuration file """
    with open(path_to_file.format(filename), 'r') as yamlfile:
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
