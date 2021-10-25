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
from typing import Callable

from scipy.optimize import OptimizeResult, minimize_scalar

from aideme.utils import assert_positive, assert_positive_integer


__all__ = ['StepSizeScheduler', 'FixedScheduler', 'LineSearchScheduler', 'PowerDecayScheduler', 'ExponentialDecayScheduler']


class StepSizeScheduler:
    def __call__(self, result: OptimizeResult, func: Callable) -> float:
        return self.compute_step_size(result, func)

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        raise NotImplementedError


class FixedScheduler(StepSizeScheduler):
    def __init__(self, step_size: float):
        assert_positive(step_size, 'step_size')
        self.step_size = step_size

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return self.step_size


class LineSearchScheduler(StepSizeScheduler):
    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return minimize_scalar(lambda step: func(result.x - step * result.search_dir), method='Brent').x


class PowerDecayScheduler(StepSizeScheduler):
    def __init__(self, step_size: float, power: float, adapt_every: int = 1):
        assert_positive(step_size, 'step_size')
        assert_positive(power, 'power')
        assert_positive_integer(adapt_every, 'adapt_every')
        self.step_size = step_size
        self.power = power
        self.adapt_every = adapt_every

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return self.step_size / (1 + result.it // self.adapt_every) ** self.power


class ExponentialDecayScheduler(StepSizeScheduler):
    def __init__(self, step_size: float, decay: float, adapt_every: int = 1):
        assert_positive(step_size, 'step_size')
        assert_positive(decay, 'decay')
        assert_positive_integer(adapt_every, 'adapt_every')
        self.step_size = step_size
        self.decay = decay
        self.adapt_every = adapt_every

    def compute_step_size(self, result: OptimizeResult, func: Callable) -> float:
        return self.step_size * self.decay ** (result.it // self.adapt_every)
