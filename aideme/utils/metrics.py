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

"""
This module contains helper functions for computing a few of the most popular metrics. A valid "metric" is any function
with the following signature:

    def metric(data, active_learner):
        return a dict of key-value pair with the computed metrics

Here, 'data' is an PartitionedDataset instance and 'active_learner' is a ActiveLearner instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sklearn

from aideme.active_learning.dsm import DualSpaceModel

if TYPE_CHECKING:
    from .types import Metrics, Callback
    from aideme.active_learning import ActiveLearner
    from aideme.explore import PartitionedDataset


def three_set_metric(data: PartitionedDataset, active_learner: ActiveLearner) -> Metrics:
    """
    :return: TSM score, which is a lower bound for F-score. Only available when running the DualSpaceModel.
    """
    if not isinstance(active_learner, DualSpaceModel):
        return {}

    pred = active_learner.polytope_model.predict(data.data.data)  # TODO: shorten data access

    pos = (pred == 1).sum()
    unknown = (pred == 0.5).sum()
    tsm = pos / (pos + unknown)

    return {'tsm': tsm}


def classification_metrics(y_test, *score_functions: str, X_test=None) -> Callback:
    """
    :param y_test: true labels of test set
    :param score_functions: list of metrics to be computed. Available metrics are: 'true_positive', 'false_positive',
    'true_negative', 'false_negative', 'accuracy', 'precision', 'recall', 'fscore'
    :param X_test: an optional test set. If None, the entire dataset will be used.
    :return: all classification scores
    """
    diff = set(score_functions) - __classification_metrics.keys()
    if len(diff) > 0:
        raise ValueError("Unknown classification metrics: {0}. Supported values are: {1}".format(sorted(diff), sorted(__classification_metrics.keys())))

    def compute(data: PartitionedDataset, active_learner: ActiveLearner) -> Metrics:
        X = X_test if X_test is not None else data.data.data
        y_pred = active_learner.predict(X)

        cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
        return {score: __classification_metrics[score](cm) for score in score_functions}

    return compute


__classification_metrics = {
    'true_positive': lambda cm: cm[1, 1],
    'false_positive': lambda cm: cm[0, 1],
    'false_negative': lambda cm: cm[1, 0],
    'true_negative': lambda cm: cm[0, 0],
    'accuracy': lambda cm: true_divide(cm[0, 0] + cm[1, 1], cm.sum()),
    'precision': lambda cm: true_divide(cm[1, 1], cm[1, 1] + cm[0, 1]),
    'recall': lambda cm: true_divide(cm[1, 1], cm[1, 1] + cm[1, 0]),
    'fscore': lambda cm: true_divide(2 * cm[1, 1], 2 * cm[1, 1] + cm[0, 1] + cm[1, 0]),
}


def true_divide(x: float, y: float) -> float:
    return 0. if y == 0 else x / y
