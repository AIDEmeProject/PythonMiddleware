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

    def convergence_criteria(iteration, metrics):
        return True if exploration can stop, False otherwise

Here, 'iteration' is an Iteration instance containing the current state of the AL exploration (i.e. data and learner),
and 'metrics' is the dictionary of all callback_metrics which have been computed in this iteration.
"""

from .validation import assert_positive_integer


def max_iter_reached(max_iter):
    """
    :param max_iter: number of iterations after which we stop the exploration process. Must be a positive integer.
    :return: a convergence criteria which stops the exploration process after the specified number of iterations
    """
    assert_positive_integer(max_iter, 'max_iter', allow_inf=True)

    def converged(iteration, metrics):
        return iteration.iter >= max_iter

    return converged


def all_points_are_known(iteration, metrics):
    """
    :return: whether no points remain in the unknown partition
    """
    return iteration.data.unknown_size == 0


def metric_reached_threshold(metric, threshold):
    """
    Computes a convergence criteria which stops exploration when a given callback metric reaches a certain threshold.
    Note that exploration may take a few iterations more to stop depending on the 'callback_skip' exploration parameter.

    :param metric: metric name
    :param threshold:
    :return: a convergence criteria
    """
    def converged(iteration, metrics):
        if metric not in metrics:
            return False

        return metrics[metric] >= threshold

    return converged
