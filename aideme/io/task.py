import pandas as pd

from .dataset import read_dataset, query_keys_matching_predicate
from .preprocessing import preprocess
from .utils import get_config_from_resources


def read_task(tag, distinct=False, get_raw=False, read_factorization=False):
    task_config = get_config_from_resources('tasks', tag)

    # read data
    dataset_config = {'distinct': distinct}
    dataset_config.update(task_config['dataset'])
    data = read_dataset(**dataset_config)

    # read labels
    positive_indexes = read_positive_indexes(task_config['labels'], dataset_config['tag'])
    labels = indexes_to_labels(positive_indexes, data.index)

    # preprocessing
    if not get_raw:
        preprocess_list = task_config.get('preprocessing', [])
        data = preprocess(data, preprocess_list)

    # factorization
    output = {'data': data, 'labels': labels}

    if read_factorization and 'factorization' in task_config:
        output['factorization_info'] = read_factorization_information(task_config['factorization'], dataset_config['tag'], data)

    return output


def read_positive_indexes(labels_config, dataset_tag):
    if 'positive_indexes' in labels_config:
        return labels_config['positive_indexes']

    elif 'predicate' in labels_config:
        return query_keys_matching_predicate(dataset_tag, labels_config['predicate'])

    raise ValueError("Either 'positive_indexes' or 'predicate' option must be present in labels configuration.")


def read_factorization_information(factorization_config, dataset_tag, data):
        feature_groups, subpredicates = factorization_config['feature_groups'], factorization_config['subpredicates']
        categorical = factorization_config.get('categorical', [False] * len(feature_groups))

        if len(feature_groups) != len(subpredicates):
            raise ValueError("Incompatible sizes of 'feature_groups' and 'subpredicates': {} != {}.".format(len(feature_groups), len(subpredicates)))

        if len(categorical) != len(feature_groups):
            raise ValueError("Incompatible size for 'categorical': found {}, but expected {}".format(len(categorical), len(subpredicates)))

        for i, (gr, pred) in enumerate(zip(feature_groups, subpredicates)):
            if all(col not in pred for col in gr):
                raise ValueError('Found disjoint feature group and subpredicate in partition #{}.'.format(i))

        # partition
        partition = [[data.columns.get_loc(col) for col in gr] for gr in feature_groups]

        # partial labels
        partial_labels_dict = {}
        for i, predicate in enumerate(subpredicates):
            positive_indexes = query_keys_matching_predicate(dataset_tag, predicate)
            partial_labels_dict[i] = indexes_to_labels(positive_indexes, data.index)

        partial_labels = pd.DataFrame.from_dict(partial_labels_dict)

        return {
            'partition': partition,
            'partial_labels': partial_labels,
            'categorical': categorical,
        }


def indexes_to_labels(positive_indexes, all_indexes):
    labels = pd.Series(data=0., index=all_indexes)
    labels[labels.index.isin(positive_indexes)] = 1
    return labels
