from os.path import join

from pandas import Series, read_sql_table, read_csv, read_sql_query

from .preprocessing import preprocess
from .utils import get_config_from_resources


def read_task_from_database(connection_string, table, predicate, key, columns=None):
    """
    Read data and labels by running queries over a database.

    :param connection_string: database string URI, on the format "dialect+driver://username:password@host:port/database"
    :param table: table name
    :param predicate: SQL predicate encoding the user's interest
    :param key: index key on the table
    :param columns: subset of columns to read

    :return: pandas Series, indexed similarly to the data, of value 1 if positive and 0 if negative
    """
    # read data
    data = read_sql_table(table_name=table, con=connection_string, index_col=key, columns=columns)

    # run query over dataframe
    query = "SELECT {} FROM {} WHERE {}".format(key, table, predicate)
    query_result = read_sql_query(query, connection_string)[key]

    # convert query result to 0-1 Series
    labels = Series(data=0., index=data.index)
    labels[labels.index.isin(query_result)] = 1.

    return data, labels


def read_task_from_csv(path, key=None, true_class=1.0):
    """
    Reads data and labels from CSV file

    :param path: path to file
    :param key: column to set as index

    :return pandas Dataframe
    """
    # read data
    data = read_csv(path, index_col=key)

    # extract label
    labels = data['class']
    labels = (labels == true_class).astype('float')

    # drop label column from data
    data.drop('class', axis=1, inplace=True)

    return data, labels


def read_data(config):
    """
    Reads data and labels from configuration

    :param config: dict-like object containing all dataset and label configurations
    :return:
    """
    # add dataset config
    name = config.pop('name')
    config.__update(get_config_from_resources('datasets', name))

    # add source config
    source = config.pop('source')
    source_uri = get_config_from_resources('sources', source)

    if source == 'file':
        path = join(source_uri, name + '.csv')
        return read_task_from_csv(path, **config)

    elif source == 'postgres':
        database = config.pop('database')
        connection_string = '{}/{}'.format(source_uri, database)
        return read_task_from_database(connection_string, **config)

    else:
        raise ValueError("Unknown source value: " + source)


def read_task(task_name, distinct=False, get_raw=False):
    """
    Read a given task's data and labels from a CSV file or database, as specified in the resources configuration files

    :param name: dataset to be read, defined on config.py file
    :param columns: list of columns to read (if None, all columns are read)
    :param distinct: whether to remove duplicates or not
    :return: pandas Dataframe
    """
    # read task configuration
    task_config = get_config_from_resources('tasks', task_name)

    # merge dataset and user configs
    data, labels = read_data(task_config['data'])

    # preprocessing
    if not get_raw:
        preprocess_list = task_config.get('preprocessing', [])
        data = preprocess(data, preprocess_list)

    # remove duplicates if necessary
    # Must be done AFTER preprocessing, so the same scaling is used during both exploring and evaluation
    if distinct:
        data = data.drop_duplicates()
        labels = labels.loc[data.index]

    return data, labels
