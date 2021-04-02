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

from collections import defaultdict
from typing import TYPE_CHECKING, Sequence, List, Optional

import pandas as pd

from .dataset import read_dataset, query_keys_matching_predicate
from .preprocessing import preprocess_data
from .utils import get_config_from_resources

if TYPE_CHECKING:
    from ..utils.types import Config


def read_task(tag: str, distinct: bool = True, sort_index: bool = True, preprocess: bool = True, read_factorization: bool = False) -> Config:
    task_config = get_config_from_resources('tasks', tag)

    # read data
    dataset_config: Config = {'distinct': distinct, 'sort_index': sort_index}
    dataset_config.update(task_config['dataset'])
    data = read_dataset(**dataset_config)

    # read labels
    if 'labels' in data.columns:
        labels = data['labels']
        data.drop('labels', axis=1, inplace=True)
    else:
        positive_indexes = read_positive_indexes(task_config['labels'], dataset_config['tag'])
        labels = indexes_to_labels(positive_indexes, data.index)

    # preprocessing
    if preprocess:
        preprocess_list = task_config.get('preprocessing', [])
        data = preprocess_data(data, preprocess_list)

    # factorization
    output = {'data': data, 'labels': labels, 'one_hot_groups': compute_groups(data)}

    if read_factorization and 'factorization' in task_config:
        output['factorization_info'] = read_factorization_information(task_config['factorization'], dataset_config['tag'], data)

    return output


def read_positive_indexes(labels_config: Config, dataset_tag: str) -> Sequence:
    if 'positive_indexes' in labels_config:
        return labels_config['positive_indexes']

    elif 'predicate' in labels_config:
        return query_keys_matching_predicate(dataset_tag, labels_config['predicate'])

    raise ValueError("Either 'positive_indexes' or 'predicate' option must be present in labels configuration.")


def read_factorization_information(factorization_config: Config, dataset_tag: str, data: pd.DataFrame) -> Config:
    output = {}

    feature_groups, subpredicates = factorization_config['feature_groups'], factorization_config['subpredicates']

    if len(feature_groups) != len(subpredicates):
        raise ValueError("Incompatible sizes of 'feature_groups' and 'subpredicates': {} != {}.".format(len(feature_groups), len(subpredicates)))

    for i, (gr, pred) in enumerate(zip(feature_groups, subpredicates)):
        if all(col not in pred for col in gr):
            raise ValueError('Found disjoint feature group and subpredicate in partition #{}.'.format(i))

    # partition
    if data.columns.nlevels == 1:
        output['partition'] = [[data.columns.get_loc(col) for col in gr] for gr in feature_groups]
    else:
        encoded_position = defaultdict(list)
        for i, col in enumerate(data.columns.get_level_values(0)):
            encoded_position[col].append(i)

        partition = []
        for gr in feature_groups:
            encoded_gr = []
            for attr in gr:
                encoded_gr.extend(encoded_position[attr])
            partition.append(encoded_gr)

        output['partition'] = partition

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


def read_partial_positive_indexes(factorization_config: Config, dataset_tag: str) -> Sequence[Sequence]:
    if 'partial_positive_indexes' in factorization_config:
        return factorization_config['partial_positive_indexes']

    elif 'subpredicates' in factorization_config:
        predicates = factorization_config['subpredicates']
        return [query_keys_matching_predicate(dataset_tag, p) for p in predicates]

    raise RuntimeError("Missing 'subpredicates' or 'partial_positive_indexes' from factorization config.")


def indexes_to_labels(positive_indexes: Sequence, all_indexes: Sequence) -> pd.Series:
    labels = pd.Series(data=0., index=all_indexes)
    labels[labels.index.isin(positive_indexes)] = 1
    return labels


def compute_groups(data: pd.DataFrame) -> Optional[List[List[int]]]:
    columns = data.columns

    if columns.nlevels != 2:
        return None

    return [columns.get_locs((col,)).tolist() for col in columns.unique(0)]
