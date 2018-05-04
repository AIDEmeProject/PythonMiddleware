from os.path import join

from yaml import safe_load

from definitions import RESOURCES_DIR


def get_config_from_resources(config, section):
    """
    Read an specific section of a YAML configuration file

    :param config: config file to read
    :param section: section to read
    :return: configuration as dict
    """
    path = join(RESOURCES_DIR, config + '.yaml')

    with open(path, 'r') as file:
        config = safe_load(file)
        return config[section]
