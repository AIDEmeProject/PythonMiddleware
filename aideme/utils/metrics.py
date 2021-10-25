#  Copyright 2019 École Polytechnique
#
#  Authorship
#    Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#    Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Disclaimer
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
#    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#    IN THE SOFTWARE.

"""
This module contains helper functions for computing a few of the most popular metrics. A valid "metric" is any function
with the following signature:

    def metric(dataset, active_learner):
        return a dict of key-value pair with the computed metrics

Here, 'dataset' is an PartitionedDataset instance and 'active_learner' is a ActiveLearner instance.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import sklearn
from scipy.special import xlogy

if TYPE_CHECKING:
    from .types import Metrics, Callback
    from aideme.active_learning import ActiveLearner
    from aideme.explore import PartitionedDataset


def three_set_metric(dataset: PartitionedDataset, active_learner: ActiveLearner) -> Metrics:
    """
    :return: TSM score, which is a lower bound for F-score. Only available when running the DualSpaceModel.
    """
    if not hasattr(active_learner, 'polytope_model'):
        return {}

    pred = active_learner.polytope_model.predict(dataset.data)  # type: ignore

    pos = (pred == 1).sum()
    unknown = (pred == 0.5).sum()
    tsm = pos / (pos + unknown)

    return {'tsm': tsm}


def prediction_entropy(dataset: PartitionedDataset, active_learner: ActiveLearner):
    """
    :return: the average of the classification probability entropy over the entire data
    """
    p = active_learner.predict_proba(dataset.data)
    mp = 1 - p

    entropy = -xlogy(p, p)
    entropy -= xlogy(mp, mp)
    return {'prediction_entropy': entropy.mean()}


def classification_metrics(X_test: np.ndarray, y_test: np.ndarray, score_functions: Sequence[str], prefix: str = '') -> Callback:
    """
    :param X_test: the test dataset
    :param y_test: true labels of test set
    :param score_functions: list of metrics to be computed. Available metrics are: 'true_positive', 'false_positive',
    'true_negative', 'false_negative', 'accuracy', 'precision', 'recall', 'fscore'
    :return: all classification scores
    :param prefix: optional prefix to be added to each score function name
    """
    X_test, y_test = sklearn.utils.check_X_y(X_test, y_test)
    calculator = __classification_metrics_calculator(score_functions, prefix)
    return lambda dataset, active_learner: calculator(y_test, active_learner.predict(X_test))


def training_classification_metrics(score_functions: Sequence[str], prefix: str = '') -> Callback:
    calculator = __classification_metrics_calculator(score_functions, prefix)

    def compute(dataset: PartitionedDataset, active_learner: ActiveLearner):
        X_train, y_train = dataset.training_set()
        return calculator(y_train, active_learner.predict(X_train))

    return compute


def __classification_metrics_calculator(score_functions, prefix: str = ''):
    diff = set(score_functions) - __classification_metrics.keys()
    if len(diff) > 0:
        raise ValueError("Unknown classification metrics: {0}. Supported values are: {1}".format(sorted(diff), sorted(__classification_metrics.keys())))

    def compute(y_test: np.ndarray, y_pred: np.ndarray) -> Metrics:
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
        return {prefix + score: __classification_metrics[score](cm) for score in score_functions}

    return compute


def __true_divide(x: float, y: float) -> float:
    return 0. if y == 0 else x / y


__classification_metrics = {
    'true_positive': lambda cm: cm[1, 1],
    'false_positive': lambda cm: cm[0, 1],
    'false_negative': lambda cm: cm[1, 0],
    'true_negative': lambda cm: cm[0, 0],
    'accuracy': lambda cm: __true_divide(cm[0, 0] + cm[1, 1], cm.sum()),
    'precision': lambda cm: __true_divide(cm[1, 1], cm[1, 1] + cm[0, 1]),
    'recall': lambda cm: __true_divide(cm[1, 1], cm[1, 1] + cm[1, 0]),
    'fscore': lambda cm: __true_divide(2 * cm[1, 1], 2 * cm[1, 1] + cm[0, 1] + cm[1, 0]),
}
