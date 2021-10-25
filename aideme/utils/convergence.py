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
All functions in the module are possible criteria for stopping the exploration process. A valid "convergence criteria"
is any function with the following signature:

    def convergence_criteria(manager: ExplorationManager, metrics: Metrics):
        return True if exploration can stop, False otherwise

Here, 'manager' is an ExplorationManager instance containing the current state of the AL exploration (i.e. data and learner),
and 'metrics' is the dictionary of all metrics which have been computed in the last iteration.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .validation import assert_non_negative_integer

if TYPE_CHECKING:
    from .types import Convergence, Metrics
    from ..explore import ExplorationManager


def max_iter_reached(max_iters: int) -> Convergence:
    """
    :param max_iters: maximum number of iterations to run. Must be a positive integer.
    :return: a convergence criteria which stops the exploration process after the specified number of iterations
    """
    assert_non_negative_integer(max_iters, 'max_exploration_iter', allow_inf=True)

    return lambda manager, metrics: manager.iters > max_iters


def all_points_are_labeled(manager: ExplorationManager, metrics: Metrics) -> bool:
    """
    Stop exploration once no unlabeled points remain.
    """
    return manager.data.unlabeled_size == 0


def metric_reached_threshold(metric: str, threshold: float) -> Convergence:
    """
    Computes a convergence criteria which stops exploration when a given callback metric reaches a certain threshold.
    Note that exploration may take a few iterations more to stop depending on the 'callback_skip' exploration parameter.

    :param metric: metric name
    :param threshold: maximum value of metric before stopping the exploration
    """
    def converged(manager: ExplorationManager, metrics: Metrics) -> bool:
        if metric not in metrics:
            return False

        return metrics[metric] >= threshold

    return converged
