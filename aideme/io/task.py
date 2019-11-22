from pandas import Series

from .dataset import read_dataset, query_keys_matching_predicate
from .preprocessing import preprocess
from .utils import get_config_from_resources


def read_task(tag, distinct=False, get_raw=False):
    task_config = get_config_from_resources('tasks', tag)

    # read data
    dataset_config = {'distinct': distinct}
    dataset_config.update(task_config['dataset'])
    data = read_dataset(**dataset_config)

    # read labels
    positive_indexes = read_positive_indexes(task_config['labels'], dataset_config)
    labels = Series(data=0., index=data.index)
    labels[labels.index.isin(positive_indexes)] = 1.

    # preprocessing
    if not get_raw:
        preprocess_list = task_config.get('preprocessing', [])
        data = preprocess(data, preprocess_list)

    return data, labels


def read_positive_indexes(labels_config, dataset_config):
    if 'positive_indexes' in labels_config:
        return labels_config['positive_indexes']

    elif 'predicate' in labels_config:
        return query_keys_matching_predicate(dataset_config['tag'], labels_config['predicate'])

    raise ValueError("Either 'positive_indexes' or 'predicate' option must be present in labels configuration.")
