from os.path import join

from pandas import read_sql_table, read_csv

from .utils import get_config_from_resources


def read_from_database(connection_string, table, columns=None, key=None):
    """
    Reads data from a database.

    :param connection_string: database string URI, on the format "dialect+driver://username:password@host:port/database"
    :param table: table name
    :param columns: list of columns to read
    :param key: column to set as index
    :return: pandas Dataframe
    """
    return read_sql_table(table_name=table, con=connection_string, index_col=key, columns=columns)


def read_from_file(path, key=None):
    """
    Reads data from a CSV file

    :param path: path to file
    :param key: column to set as index
    :return pandas Dataframe
    """
    return read_csv(path, index_col=key)


def read_dataset(name, columns=None, distinct=False):
    """
    Read a given dataset from a CSV file or database, as specified in the resources/datasets.yml file.

    :param name: dataset to be read, defined on config.py file
    :param columns: list of columns to read (if None, all columns are read)
    :param distinct: whether to remove duplicates or not
    :return: pandas Dataframe
    """
    # read main configs
    config = get_config_from_resources('datasets', name)

    source = config.pop('source')
    source_uri = get_config_from_resources('sources', source)

    # set dataset config
    if source == 'file':
        path = join(source_uri, name + '.csv')
        data = read_from_file(path, **config)

    elif source == 'postgres':
        database = config.pop('database')
        connection_string = '{}/{}'.format(source_uri, database)
        data = read_from_database(connection_string, columns=columns, **config)

    else:
        raise ValueError("Unknown source value: " + source)

    # drop duplicates if necessary
    return data.drop_duplicates() if distinct else data
