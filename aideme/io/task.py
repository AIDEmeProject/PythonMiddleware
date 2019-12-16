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
    output = {}

    feature_groups, subpredicates = factorization_config['feature_groups'], factorization_config['subpredicates']

    if len(feature_groups) != len(subpredicates):
        raise ValueError("Incompatible sizes of 'feature_groups' and 'subpredicates': {} != {}.".format(len(feature_groups), len(subpredicates)))

    for i, (gr, pred) in enumerate(zip(feature_groups, subpredicates)):
        if all(col not in pred for col in gr):
            raise ValueError('Found disjoint feature group and subpredicate in partition #{}.'.format(i))

    # partition
    output['partition'] = [[data.columns.get_loc(col) for col in gr] for gr in feature_groups]

    # mode
    if 'mode' in factorization_config:
        mode = factorization_config['mode']

        if len(mode) != len(feature_groups):
            raise ValueError("Incompatible sizes of 'feature_groups' and 'mode': {} != {}.".format(len(feature_groups), len(mode)))

        output['mode'] = mode

    # partial labels
    indexes = read_partial_positive_indexes(factorization_config, dataset_tag)
    partial_labels_dict = {i: indexes_to_labels(idx, data.index) for i, idx in enumerate(indexes)}
    output['partial_labels'] = pd.DataFrame.from_dict(partial_labels_dict)

    return output


def read_partial_positive_indexes(factorization_config, dataset_tag):
    if 'partial_positive_indexes' in factorization_config:
        return factorization_config['partial_positive_indexes']

    elif 'subpredicates' in factorization_config:
        predicates = factorization_config['subpredicates']
        return [query_keys_matching_predicate(dataset_tag, p) for p in predicates]

    raise RuntimeError("Missing 'subpredicates' or 'partial_positive_indexes' from factorization config.")


def indexes_to_labels(positive_indexes, all_indexes):
    labels = pd.Series(data=0., index=all_indexes)
    labels[labels.index.isin(positive_indexes)] = 1
    return labels
