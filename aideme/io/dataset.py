#  Copyright (c) 2019 École Polytechnique
# 
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
# 
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
# 
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.
from __future__ import annotations

import os
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd

from .utils import get_config_from_resources

if TYPE_CHECKING:
    from ..utils.types import Config


def read_dataset(tag: str, columns: Optional[Sequence[str]] = None, distinct: bool = False) -> pd.DataFrame:
    """
    Read a given dataset from a CSV file or database, as specified in the resources/datasets.yml file.

    :param tag: dataset to be read, defined on config.py file
    :param columns: list of columns to read (if None, all columns are read)
    :param distinct: whether to remove duplicates or not
    :return: pandas Dataframe
    """
    config = get_config_from_resources('datasets', tag)

    source: str = config['source']

    if source == 'filesystem':
        data = read_from_file(config, columns)
    elif source == 'postgres':
        data = read_from_database(config, columns)
    else:
        raise ValueError("Unknown source value: " + source)

    if distinct:
        data = data.drop_duplicates()

    return data


def read_from_file(config: Config, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
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

    raise ValueError('Unsupported extension for file {}'.format(path))


def get_path_to_data(config: Config) -> str:
    source_config = get_config_from_resources('sources', config['source'])
    return os.path.join(source_config['path'], config['filename'])


def read_from_database(config: Config, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
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


def get_connection_string(config: Config) -> str:
    source = config['source']

    if source != 'postgres':
        raise ValueError('Querying data only supported over "postgres" sources, but got {}'.format(source))

    source_config = get_config_from_resources('sources', source)
    source_config['database'] = config['database']

    return 'postgresql://{user}:{password}@{address}:{port}/{database}'.format(**source_config)


def query_keys_matching_predicate(tag: str, predicate: str) -> pd.Series:
    config = get_config_from_resources('datasets', tag)

    connection_string = get_connection_string(config)

    key = config['key']
    query = "SELECT {} FROM {} WHERE {}".format(key, config['table'], predicate)

    return pd.read_sql_query(query, connection_string)[key]
