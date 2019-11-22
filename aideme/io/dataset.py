import os

import pandas as pd

from .utils import get_config_from_resources


def read_dataset(tag, columns=None, distinct=False):
    """
    Read a given dataset from a CSV file or database, as specified in the resources/datasets.yml file.

    :param tag: dataset to be read, defined on config.py file
    :param columns: list of columns to read (if None, all columns are read)
    :param distinct: whether to remove duplicates or not
    :return: pandas Dataframe
    """
    config = get_config_from_resources('datasets', tag)

    source = config['source']

    if source == 'filesystem':
        data = read_from_file(config, columns)

    elif source == 'postgres':
        data = read_from_database(config, columns)

    else:
        raise ValueError("Unknown source value: " + source)

    if distinct:
        data = data.drop_duplicates()

    return data


def read_from_file(config, columns=None):
    """
    Reads data from a file. We support reading from CSV and Pickle files.

    :param config: dataset configuration dictionary
    :param columns: columns to read (if None, all columns are read)
    :param key: column to set as index
    :return pandas Dataframe
    """
    path = get_path_to_data(config)
    key = config.get('key', None)

    if path.endswith('.csv'):
        return pd.read_csv(path, usecols=columns, index_col=key)

    elif path.endswith('.p'):
        data = pd.read_pickle(path)

        if key:
            data.set_index(key, inplace=True)

        if columns:
            data = data[columns]

        return data

    raise ValueError('Unsupported extension for file'.format(path))


def get_path_to_data(config):
    source_config = get_config_from_resources('sources', config['source'])
    return os.path.join(source_config['path'], config['filename'])


def read_from_database(config, columns=None):
    """
    Reads data from a database.

    :param connection_string: database string URI, on the format "dialect+driver://username:password@host:port/database"
    :param table: table name
    :param columns: list of columns to read
    :param key: column to set as index
    :return: pandas Dataframe
    """
    connection_string = get_connection_string(config)

    table = config['table']
    key = config.get('key', None)

    return pd.read_sql_table(table, connection_string, index_col=key, columns=columns)


def get_connection_string(config):
    source = config['source']

    if source != 'postgres':
        raise ValueError('Querying data only supported over "postgres" sources, but got {}'.format(source))

    source_config = get_config_from_resources('sources', source)
    source_config['database'] = config['database']

    return 'postgresql://{user}:{password}@{address}:{port}/{database}'.format(**source_config)


def query_keys_matching_predicate(tag, predicate):
    config = get_config_from_resources('datasets', tag)

    connection_string = get_connection_string(config)

    key = config['key']
    query = "SELECT {} FROM {} WHERE {}".format(key, config['table'], predicate)

    return pd.read_sql_query(query, connection_string)[key]
