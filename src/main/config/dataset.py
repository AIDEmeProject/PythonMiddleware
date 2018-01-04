from os.path import join
from pandas import read_sql_table, read_csv
from .utils import read_dataset_config, read_connection_config


def read_dataset_from_file(dataset, columns=None, key=None):
    # get data folder dir
    data_dir = read_connection_config('datafolder')
    path = join(data_dir, dataset, dataset) + '.data'

    # if both columns and index_col are not None, must add index_col to columns list
    if columns and key:
        columns += [key]

    return read_csv(path, usecols=columns, index_col=key)


def read_dataset_from_postgres(name, database, columns=None, key=None):
    database_connection_string = read_connection_config('postgres') + '/' + database
    return read_sql_table(name, database_connection_string, index_col=key, columns=columns)


def read_dataset(dataset, columns=None, keep_duplicates=False):
    # get config
    dataset_config = read_dataset_config(dataset)
    dataset_config['columns'] = columns

    # read data
    if dataset_config.pop('connection') == 'postgres':
        data = read_dataset_from_postgres(**dataset_config)
    else:
        data = read_dataset_from_file(dataset, **dataset_config)

    # set dataset index to rowid if None
    if data.index.name is None:
        if 'rowid' in data.columns:
            raise RuntimeError("Couldn't name index in dataframe because 'rowid' column already exists.")
        data.index.name = 'rowid'

    # remove duplicates if needed
    return data if keep_duplicates else data.drop_duplicates()


if __name__ == '__main__':
    data = read_dataset('iris')
    #data = read_dataset('housing', columns=['town', 'price'], keep_duplicates=False)
    #data = read_dataset('cars', columns=['make'], keep_duplicates=True)
    print(data.shape)
    print(data.head())