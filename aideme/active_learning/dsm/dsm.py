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

from typing import Any, TYPE_CHECKING, Union, Sequence, Optional

from .base import DualSpaceModelBase
from .model import PolytopeModel, FactorizedPolytopeModel, PolytopeModelBase
from ..active_learner import FactorizedActiveLearner

if TYPE_CHECKING:
    from ..active_learner import ActiveLearner
    from ...utils.types import Partition


class DualSpaceModel(DualSpaceModelBase):
    """
    Dual Space Model algorithm: no factorization
    """
    def __init__(self, active_learner: ActiveLearner, mode: Union[str] = 'persist',
                 sample_unknown_proba: float = 0.5, tol: float = 1e-12):
        super().__init__(PolytopeModel(mode, tol), active_learner, sample_unknown_proba)


class FactorizedDualSpaceModel(DualSpaceModelBase, FactorizedActiveLearner):
    """
    Dual Space Model algorithm with factorization
    """
    def __init__(self, active_learner: ActiveLearner, sample_unknown_proba: float = 0.5,
                 partition: Optional[Partition] = None, mode: Union[str, Sequence[str]] = 'persist',
                 tol: float = 1e-12):
        super().__init__(FactorizedPolytopeModel(partition, mode, tol), active_learner, sample_unknown_proba)
        self.__tol = tol

    def set_factorization_structure(self, **factorization_info: Any) -> None:
        partition, mode = factorization_info['partition'], factorization_info['mode']
        self.polytope_model: PolytopeModelBase = FactorizedPolytopeModel(partition, mode, self.__tol)
