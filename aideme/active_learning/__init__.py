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
from .active_learner import ActiveLearner, FactorizedActiveLearner
from .dsm import DualSpaceModel, FactorizedDualSpaceModel
from .entropy import EntropyReductionLearner
from .factorization.active_learning import SwapLearner, SimplifiedSwapLearner, FLMUncertaintySampler
from .margin import *
from .nlp import TwoStepsLearner
from .query_by_disagreement import QueryByDisagreement
from .random import RandomSampler
from .uncertainty import UncertaintySampler
from .version_space import *

__all__ = [
    'ActiveLearner', 'FactorizedActiveLearner', 'UncertaintySampler', 'RandomSampler', 'SimpleMargin', 'RatioMargin',
    'DualSpaceModel', 'FactorizedDualSpaceModel',
    'LinearVersionSpace', 'KernelVersionSpace',
    'SubspatialVersionSpace', 'SubspatialSimpleMargin', 'TwoStepsLearner',
    'QueryByDisagreement', 'EntropyReductionLearner', 'SwapLearner', 'SimplifiedSwapLearner', 'FLMUncertaintySampler',
]
