from os.path import join
from pandas import read_csv

from .utils import read_connection_config
from ..user import FakeUser, DummyUser


def get_user_from_file(dataset, max_iter, true_class=1.0):
    data_dir = read_connection_config('datafolder')
    path = join(data_dir, dataset, dataset) + '.labels'

    y_true = read_csv(path).iloc[:, 0]  # to get a series
    return DummyUser(y_true, max_iter, true_class)


def get_user(data, max_iter, true_predicate=None, dataset=None, true_class=None):
    if true_predicate:
        return FakeUser(data, true_predicate, max_iter)
    else:
        return get_user_from_file(dataset, max_iter, true_class)
