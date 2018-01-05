from os.path import join
import logging.config
import yaml

from definitions import RESOURCES_DIR


def get_config_from_file(filename, section=None):
    """ Read a section from YAML configuration file """
    path = join(RESOURCES_DIR, filename)

    with open(path, 'r') as yamlfile:
        cfg = yaml.safe_load(yamlfile)
        return cfg if section is None else cfg[section]


def setup_logging():
    """
        Setup logging configuration
    """
    config = get_config_from_file('logging.yml')
    logging.config.dictConfig(config)


def get_config_from_resources(file, name):
    path = join(RESOURCES_DIR, file)

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config[name]


read_dataset_config = lambda x: get_config_from_resources('datasets.yml', x)

read_connection_config = lambda x: get_config_from_resources('connections.yml', x)['connection_string']

read_task_config = lambda x: get_config_from_resources('tasks.yml', x)
