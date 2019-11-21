from sklearn.metrics import f1_score

from aideme.initial_sampling import StratifiedSampler
from aideme.explore import PoolBasedExploration
from aideme.experiments import Experiment
from aideme.active_learning import *
from aideme.utils.metrics import *


def get_user_study(ls):
    datasets_list = []
    for i in ls:
        num = str(i)
        if i < 10:
            num = '0' + num

        datasets_list.append(('User Study Query ' + num, 'user_study_query' + str(i)))
    return datasets_list

def get_sdss(queries, sels):
    ls = []
    for query in queries:
        for sel in sels:
            ls.append(("SDSS Query {0} - {1}% selectivity".format(query, sel), "sdss_Q{0}_{1}%".format(query, sel)))

    return ls

# DATASETS
#datasets_list = get_user_study([11, 12])  # range(1,13)
datasets_list = get_sdss([3], [1, 0.1])

# LEARNERS
active_learners_list = [
    #("Simple Margin", SimpleMargin(kernel='rbf', C=1024)),
    ("Kernel QBC", KernelQueryByCommittee(gamma=0.01, n_samples=32, warmup=100, thin=1, sampling='deterministic', rounding=True, kernel='rbf')),
    #("Bayesian Kernel QBC", KernelQueryByCommittee(gamma=0.5, n_samples=32, warmup=100, thin=1, sampling='bayesian', sigma=10000.0, kernel='rbf')),
]

# RUN PARAMS
TIMES = 1
NUMBER_OF_ITERATIONS = 100
SUBSAMPLING = float('inf')  # 50000, float('inf')
INITIAL_SAMPLER = StratifiedSampler(pos=1, neg=1)
CALLBACK = [
    classification_metrics('fscore'),
    #three_set_metric,
]
CALLBACK_SKIP = 10
PRINT_CALLBACK_RESULT = False


# run experiment
explore = PoolBasedExploration(NUMBER_OF_ITERATIONS, INITIAL_SAMPLER, SUBSAMPLING, CALLBACK, CALLBACK_SKIP, PRINT_CALLBACK_RESULT)
Experiment().run(datasets_list, active_learners_list, times=TIMES, explore=explore)
