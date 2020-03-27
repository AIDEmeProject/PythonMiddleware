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

from functools import wraps
from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Metrics


__metrics: Metrics = {}


def flush() -> None:
    """
    Clears all stored metrics.
    """
    global __metrics
    __metrics = {}


def get_metrics() -> Metrics:
    """
    :return: the current metrics dictionary
    """
    return __metrics


def log_execution_time(key: str, on_duplicates: str = 'overwrite'):
    """
    Decorator for logging the execution time of functions
    :param key: metric's name
    :param on_duplicates: de-duplication strategy. See 'log_metric' for available options.
    :return:
    """
    def time_decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            t0 = perf_counter()
            res = func(*args, **kwargs)
            log_metric(key, perf_counter() - t0, on_duplicates=on_duplicates)
            return res
        return wrapped_func

    return time_decorator


def log_metrics(metrics: Metrics, on_duplicates: str = 'overwrite') -> None:
    """
    Logs a dict of metrics
    :param metrics: dict of metrics to be logged
    :param on_duplicates: de-duplication strategy. See 'log_metric' for available options.
    """
    for key, value in metrics.items():
        log_metric(key, value, on_duplicates=on_duplicates)


def log_metric(key: str, value: Any, on_duplicates: str = 'overwrite') -> None:
    """
    Logs a single metric.
    :param key: metric's name
    :param value: metric's value
    :param on_duplicates: de-duplication strategy. Available options are:
        - 'OVERWRITE' (default): replaces the current value with the new one
        - 'SUM': will attempt to store the sum 'current_value + new_value'
        - 'APPEND': store all values in a list
    """
    __metrics[key] = __get_value(key, value, on_duplicates)


def __get_value(key, value, on_duplicates):
    on_duplicates = on_duplicates.upper()

    if on_duplicates == 'OVERWRITE':
        return value

    if on_duplicates == 'SUM':
        cur_val = __metrics.get(key, None)
        return value if cur_val is None else cur_val + value

    if on_duplicates == 'APPEND':
        cur_val = __metrics.get(key, [])
        cur_val.append(value)
        return cur_val

    raise ValueError("Unknown option {} for on_duplicates. Available options are: 'overwrite', 'sum', or 'append'".format(on_duplicates))
