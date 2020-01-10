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
All functions in the module are possible criteria for stopping the exploration process. A valid "convergence criteria"
is any function with the following signature:

    def convergence_criteria(manager, metrics):
        return True if exploration can stop, False otherwise

Here, 'manager' is an ExplorationManager instance containing the current state of the AL exploration (i.e. data and learner),
and 'metrics' is the dictionary of all metrics which have been computed in the last iteration.
"""

import math
from .validation import assert_non_negative_integer


def max_iter_reached(max_exploration_iter, max_initial_sampling_iter=math.inf):
    """
    :param max_exploration_iter: maximum number of exploration iterations to run. Must be a positive integer.
    :param max_initial_sampling_iter: maximum number of initial sampling iterations to run. Must be a positive integer.
    :return: a convergence criteria which stops the exploration process after the specified number of iterations
    """
    assert_non_negative_integer(max_exploration_iter, 'max_exploration_iter', allow_inf=True)
    assert_non_negative_integer(max_initial_sampling_iter, 'max_initial_sampling_iter', allow_inf=True)

    def converged(manager, metrics):
        return manager.exploration_iters > max_exploration_iter or manager.initial_sampling_iters > max_initial_sampling_iter

    return converged


def all_points_are_known(manager, metrics):
    """
    Stop exploration once no unknown points remain.
    """
    return manager.data.unknown_size == 0


def all_points_are_labeled(manager, metrics):
    """
    Stop exploration once no unlabeled points remain.
    """
    return manager.data.unlabeled_size == 0


def metric_reached_threshold(metric, threshold):
    """
    Computes a convergence criteria which stops exploration when a given callback metric reaches a certain threshold.
    Note that exploration may take a few iterations more to stop depending on the 'callback_skip' exploration parameter.

    :param metric: metric name
    :param threshold: maximum value of metric before stopping the exploration
    """
    def converged(manager, metrics):
        if metric not in metrics:
            return False

        return metrics[metric] >= threshold

    return converged
