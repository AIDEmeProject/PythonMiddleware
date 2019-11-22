import os
import yaml

from definitions import RESOURCES_DIR


def get_config_from_resources(config, section=None):
    """
    Read an specific section of a YAML configuration file

    :param config: config file to read
    :param section: section to read
    :return: configuration as dict
    """
    path = os.path.join(RESOURCES_DIR, config + '.yaml')

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config[section] if section else config
