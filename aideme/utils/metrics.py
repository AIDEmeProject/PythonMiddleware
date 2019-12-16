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

import sklearn


def three_set_metric(X, y, active_learner):
    if not hasattr(active_learner, 'polytope_model'):
        return {}

    pred = active_learner.polytope_model.predict(X)

    pos = (pred == 1).sum()
    unknown = (pred == 0.5).sum()
    tsm = pos / (pos + unknown)

    return {'tsm': tsm}


def classification_metrics(*score_functions, labeling_function='AND'):
    diff = set(score_functions) - __classification_metrics.keys()
    if len(diff) > 0:
        raise ValueError("Unknown classification metrics: {0}. Supported values are: {1}".format(sorted(diff), sorted(__classification_metrics.keys())))

    if isinstance(labeling_function, str):
        labeling_function = __labeling_functions.get(labeling_function.upper())

    def compute(X, y, active_learner):
        if y.ndim > 1:
            y = labeling_function(y)

        y_pred = active_learner.predict(X)
        cm = sklearn.metrics.confusion_matrix(y, y_pred, labels=[0, 1])
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

__labeling_functions = {
    'AND': lambda y: y.min(axis=1),
    'OR': lambda y: y.max(axis=1),
}


def true_divide(x, y):
    return 0 if y == 0 else x / y
