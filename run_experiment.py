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

from aideme.active_learning import *
from aideme.experiments import Experiment
from aideme.explore import PoolBasedExploration
from aideme.initial_sampling import StratifiedSampler
from aideme.utils import *

class tag:
    def __init__(self, tag, learner, factorize=True):
        self.tag = tag
        self.learner = learner
        self.factorized = factorize if isinstance(learner, FactorizedActiveLearner) else False

    def __repr__(self):
        return "tag={}, learner={}, factorize={}".format(self.tag, self.learner, self.factorized)

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError('Only integer indexes are supported.')

        if item == 0:
            return self.tag
        if item == 1:
            return self.learner
        if item == 2:
            return self.factorized
        raise IndexError('Index {} out-of-bounds.'.format(item))

    def __len__(self):
        return 3

def get_user_study(ls):
    assert set(ls).issubset(range(1, 19))

    process = lambda x: str(x) if x >= 10 else '0' + str(x)
    return [('User Study Query ' + num, 'user_study_' + num) for num in map(process, ls)]

def get_sdss(queries):
    assert set(queries).issubset(range(1, 12))
    return [("SDSS Query {}".format(q), "sdss_q{}".format(q)) for q in queries]

# DATASETS
datasets_list = get_user_study([2])  # range(1,13)
#datasets_list = get_sdss([3], [1, 0.1])

# LEARNERS
active_learners_list = [
    ("Simple Margin C=1e7", SimpleMargin(C=1e7)),
    ("DSM C=1024 m=persist", DualSpaceModel(SimpleMargin(C=1024), mode='persist')),
    #("DSM no Fact", DualSpaceModel(SimpleMargin(C=1e7), mode='persist'), False),
    ("Version Space ss=16 w=1000 t=100", KernelQueryByCommittee(n_samples=16, warmup=1000, thin=100, rounding=True)),
    ("Fact VS GREEDY ss=8 w=100 t=10", SubspatialVersionSpace(loss='GREEDY', n_samples=8, warmup=100, thin=10, rounding=True)),
    #("Bayesian Kernel QBC", KernelQueryByCommittee(gamma=0.5, n_samples=32, warmup=100, thin=1, sampling='bayesian', sigma=10000.0, kernel='rbf')),
]

# RUN PARAMS
TIMES = 10
NUMBER_OF_ITERATIONS = 100
SUBSAMPLING = float('inf')  # 50000, float('inf')
INITIAL_SAMPLER = StratifiedSampler(pos=1, neg=1, assert_neg_all_subspaces=False)
CALLBACK = [
    classification_metrics('fscore'),
    three_set_metric,
]
CONVERGENCE_CRITERIA = [
    #metric_reached_threshold('fscore', 0.9),
    #all_points_are_known,
]
CALLBACK_SKIP = 10
PRINT_CALLBACK_RESULT = False

# run experiment
active_learners_list = [tag(*elem) for elem in active_learners_list]
explore = PoolBasedExploration(NUMBER_OF_ITERATIONS, INITIAL_SAMPLER, SUBSAMPLING, CALLBACK, CALLBACK_SKIP, PRINT_CALLBACK_RESULT, CONVERGENCE_CRITERIA)
Experiment().run(datasets_list, active_learners_list, times=TIMES, explore=explore)
