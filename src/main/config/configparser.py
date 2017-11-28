from copy import deepcopy
import pandas as pd
from .utils import get_config_from_file
from .preprocessor import Preprocessor
from .task import Task
from ..active_learning import learner_configs
from ..user import IndexUser, DummyUser

datafolder_connection_string = '{base}/{name}/{name}.{ext}'
postgres_connection_string = '{base}/{database}'


class ConfigurationParser:
    def __init__(self):
        self._config = {}

    def __getitem__(self, item):
        return self._config[item]

    def set(self, config, extra=None, copy=True):
        self._config = deepcopy(config) if copy else config
        self._parse_new_config(extra)

    def _parse_new_config(self, extra=None):
        pass

    def get(self):
        raise NotImplementedError


class DatasetConfigurationParser(ConfigurationParser):
    def _flatten(self, config_name, config_file):
        new_config = get_config_from_file(config_file, self._config[config_name])
        self._config.update(new_config)

    def __merge_connection(self):
        connection = self._config['connection']
        conn_string = self._config['connection_string']

        if connection == 'datafolder':
            name = self._config['name']
            self._config['connection_string'] = datafolder_connection_string.format(base=conn_string,
                                                                                    name=name,
                                                                                    ext='data')

        elif connection == 'postgres':
            database = self._config.pop('database')
            self._config['connection_string'] = postgres_connection_string.format(base=conn_string,
                                                                                  database=database)

    def _parse_new_config(self, extra=None):
        # flatten both 'name' and 'connection' variables
        self._flatten('name', 'datasets.yml')
        self._flatten('connection', 'connections.yml')

        # build connection string
        self.__merge_connection()

        # set columns and key
        self._set_columns()
        self._set_key()

    def _set_columns(self):
        if self._config.get('columns', '*') == '*':
            self._config['columns'] = None

    def _set_key(self):
        columns = self._config['columns']
        key = self._config.get('key', None)

        if columns and key not in columns:
            self._config['key'] = None

    def __get_dataset(self, name, connection, connection_string, columns=None, key=None):
        if connection == 'datafolder':
            data = pd.read_csv(connection_string, usecols=columns, index_col=key)

        elif connection == 'postgres':
            if columns:
                query = "SELECT DISTINCT {0} FROM {1}".format(','.join(columns), name)
                data = pd.read_sql_query(query, con=connection_string, index_col=key)
            else:
                data = pd.read_sql_table(name, con=connection_string, index_col=key)

        else:
            raise ValueError("Dataset configuration must contain either 'path' or 'connection_string' parameters.")

        if key:
            data = data.reset_index(drop=True)

        return data

    def get(self):
        return self.__get_dataset(**self._config)


class UserConfigurationParser(ConfigurationParser):
    def _parse_new_config(self, extra=None):
        if 'path' not in self._config:
            self._config['path'] = extra[:-4] + 'labels'

    def __get_user(self, data, max_iter, path='', true_predicate='', true_class=None):
        if true_predicate:
            predicate = true_predicate.replace('AND', '&').replace('OR', '|').replace('=', '==')
            true_index = data.query(predicate).index
            return IndexUser(true_index, max_iter)

        elif path:
            labels = pd.read_csv(path).values.ravel()
            if true_class is not None:
                labels = 2. * (labels == true_class) - 1.
            return DummyUser(labels, max_iter)

        raise RuntimeError(
            "User configuration must contain either 'true_predicate' or 'path' parameters, "
            "or provide an in-disk .labels file by setting 'read_labels' to True."
        )

    def get(self, data):
        return self.__get_user(data, **self._config)


class ActiveLearnerConfigurationParser(ConfigurationParser):
    def __get_learner(self, name, params):
        name = name.strip().replace(" ", "").lower()
        learner = learner_configs[name](**params)
        return learner

    def get(self):
        return self.__get_learner(**self._config)


class TaskConfigurationParser(ConfigurationParser):
    def __init__(self):
        super().__init__()
        self.__dataset_config_parser = DatasetConfigurationParser()
        self.__user_config_parser = UserConfigurationParser()
        self.__learner_config_parser = ActiveLearnerConfigurationParser()
        self.__preprocessor = Preprocessor()

    def _parse_new_config(self, extra=None):
        self.__dataset_config_parser.set(self._config['dataset'], copy=False)
        self.__user_config_parser.set(self._config['user'], self.__dataset_config_parser['connection_string'], copy=False)
        self.__learner_config_parser.set(self._config['learner'], copy=False)
        self.__preprocessor.set(self._config['preprocessing'])

    def get(self):
        data = self.__dataset_config_parser.get()
        user = self.__user_config_parser.get(data)
        learner = self.__learner_config_parser.get()
        return Task(self.__preprocessor.transform(data), user, learner)


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