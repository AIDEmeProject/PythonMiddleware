from os.path import join
from pandas import read_csv

from .utils import read_connection_config
from ..user import FakeUser, DummyUser


def get_user_from_file(dataset_name, max_iter, true_class=1.0, noise=0.0):
    data_dir = read_connection_config('datafolder')
    path = join(data_dir, dataset_name, dataset_name) + '.labels'

    y_true = read_csv(path).iloc[:, 0]  # to get a series
    return DummyUser(y_true, max_iter, true_class, noise)


def get_user(data, max_iter, true_predicate=None, dataset_name=None, true_class=1.0, noise=0.0):
    if true_predicate:
        return FakeUser(data, true_predicate, max_iter, noise)
    else:
        return get_user_from_file(dataset_name, max_iter, true_class, noise)
