from aideme.active_learning import *
from aideme.experiments import Experiment
from aideme.explore import PoolBasedExploration
from aideme.initial_sampling import StratifiedSampler
from aideme.utils.metrics import *

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
    datasets_list = []
    for i in ls:
        num = str(i)
        if i < 10:
            num = '0' + num

        datasets_list.append(('User Study Query ' + num, 'user_study_' + num))
    return datasets_list

def get_sdss(queries, sels):
    ls = []
    for query in queries:
        for sel in sels:
            ls.append(("SDSS Query {0} - {1}% selectivity".format(query, sel), "sdss_Q{0}_{1}%".format(query, sel)))

    return ls

# DATASETS
datasets_list = get_user_study([2])  # range(1,13)
#datasets_list = get_sdss([3], [1, 0.1])

# LEARNERS
active_learners_list = [
    #("Simple Margin", SimpleMargin(C=1e7)),
    ("DSM Fact", DualSpaceModel(SimpleMargin(C=1024), mode='persist')),
    ("DSM no Fact", DualSpaceModel(SimpleMargin(C=1024), mode='persist'), False),
    #("Kernel QBC", KernelQueryByCommittee(n_samples=32, warmup=1000, thin=100, sampling='deterministic', rounding=True, kernel='rbf')),
    #("Bayesian Kernel QBC", KernelQueryByCommittee(gamma=0.5, n_samples=32, warmup=100, thin=1, sampling='bayesian', sigma=10000.0, kernel='rbf')),
]

# RUN PARAMS
TIMES = 5
NUMBER_OF_ITERATIONS = 50
SUBSAMPLING = float('inf')  # 50000, float('inf')
INITIAL_SAMPLER = StratifiedSampler(pos=1, neg=1, assert_neg_all_subspaces=False)
CALLBACK = [
    classification_metrics('fscore'),
    #three_set_metric,
]
CALLBACK_SKIP = 10
PRINT_CALLBACK_RESULT = False


# run experiment
active_learners_list = [tag(*elem) for elem in active_learners_list]
explore = PoolBasedExploration(NUMBER_OF_ITERATIONS, INITIAL_SAMPLER, SUBSAMPLING, CALLBACK, CALLBACK_SKIP, PRINT_CALLBACK_RESULT)
Experiment().run(datasets_list, active_learners_list, times=TIMES, explore=explore)
