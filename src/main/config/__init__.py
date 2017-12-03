from .configparser import DatasetConfigurationParser, UserConfigurationParser
from .preprocessor import Preprocessor
from .utils import get_config_from_file

__all__ = ['get_dataset_and_user', 'get_config_from_file']


def get_dataset_and_user(name, columns=None, true_predicate='', preprocessing_list=None):
    config = get_config_from_file('tasks.yml', name)

    if columns:
        config['dataset']['columns'] = columns

    if true_predicate:
        config['user']['true_predicate'] = true_predicate

    if preprocessing_list is not None:
        config['preprocessing'] = preprocessing_list

    dataset_config = DatasetConfigurationParser()
    dataset_config.set(config['dataset'])
    data = dataset_config.get()

    user_config = UserConfigurationParser()
    user_config.set(config['user'], dataset_config['connection_string'])
    user = user_config.get(data)

    preprocessor = Preprocessor()
    preprocessor.set(config.get('preprocessing', []))
    data = preprocessor.transform(data)

    return data, user


def get_dataset(name, columns=None, preprocessing_list=None):
    config = get_config_from_file('tasks.yml', name)

    if columns:
        config['dataset']['columns'] = columns

    if preprocessing_list is not None:
        config['preprocessing'] = preprocessing_list

    dataset_config = DatasetConfigurationParser()
    dataset_config.set(config['dataset'])
    data = dataset_config.get()

    preprocessor = Preprocessor()
    preprocessor.set(config.get('preprocessing', []))
    data = preprocessor.transform(data)

    return data

