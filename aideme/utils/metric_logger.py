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
