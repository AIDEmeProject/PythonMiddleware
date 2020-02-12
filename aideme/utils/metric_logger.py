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


def get_metrics() -> Metrics:
    return __metrics


def log_metric(key: str, value: Any) -> None:
    __metrics[key] = value


def log_metrics(d: Metrics) -> None:
    __metrics.update(d)


def get_metric(key: str) -> Any:
    return __metrics[key]


def log_execution_time(key: str):
    def time_decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            t0 = perf_counter()
            res = func(*args, **kwargs)
            log_metric(key, perf_counter() - t0)
            return res
        return wrapped_func

    return time_decorator


def flush() -> None:
    global __metrics
    __metrics = {}
