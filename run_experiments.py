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
from sklearn.svm import SVC

from aideme.active_learning import *
from aideme.experiments import run_all_experiments, Tag
from aideme.experiments.folder import RootFolder
from aideme.initial_sampling import stratified_sampler
from aideme.utils import *


def get_aluma(sizes, dims, original=True):
    suffix = 'original' if original else 'preprocessed'

    tasks = []
    for size in sizes:
        for dim in dims:
            tasks.append('aluma_size={}_dim={}_{}'.format(size, dim, suffix))
    return tasks

def get_user_study(ls):
    assert set(ls).issubset(range(1, 19))
    process = lambda x: str(x) if x >= 10 else '0' + str(x)
    return ['user_study_' + num for num in map(process, ls)]


def get_sdss(queries):
    assert set(queries).issubset(range(1, 12))
    return ["sdss_q{}".format(q) for q in queries]


# TASKS
task_list = get_sdss([
    1, 2, 3,
    5, 6, 7,
    8, 9, 10, 11,
])
#task_list = get_user_study([7])  # range(1,13)
#task_list = ['sdss_q8_dsm']   # DSM queries
#task_list = get_aluma([10000,], [2,], original=False)  # run kernel classifier
#task_list = get_aluma([10000,], [2,], original=True)   # run linear classifier

# LEARNERS
# State-of-the-art VS algorithms
SM = Tag(SimpleMargin, C=1e5)
QBD = Tag(QueryByDisagreement, learner=Tag(SVC, C=1e5), background_sample_size=200, background_sample_weight=1e-5)
ALUMA = Tag(LinearVersionSpace, single_chain=False, n_samples=100, warmup=2000, rounding=False)

# DSM
DSM = Tag(DualSpaceModel, active_learner=Tag(SimpleMargin, C=1e5), mode='positive')
FactDSM = Tag(FactorizedDualSpaceModel, active_learner=Tag(SimpleMargin, C=1e5))

# ICDM 2019: VS + rounding (no decomposition, no optimizations)
kvs_global_params = {'decompose': False, 'n_samples': 16, 'warmup': 100, 'thin': 100, 'rounding': True, 'rounding_cache': False, 'rounding_options': {'strategy': 'default'}}
KVS       = Tag(KernelVersionSpace, **kvs_global_params)
GreedyKVS = Tag(SubspatialVersionSpace, loss='GREEDY' , **kvs_global_params)
ProdKVS   = Tag(SubspatialVersionSpace, loss='PRODUCT', **kvs_global_params)

# New VS algorithm: VS + rounding + decomposition (no optimizations)
lvs_global_params = {'decompose': True, 'n_samples': 16, 'warmup': 100, 'thin': 100, 'rounding': True, 'rounding_cache': False, 'rounding_options': {'strategy': 'default'}}
LVS       = Tag(KernelVersionSpace, **kvs_global_params)
GreedyLVS = Tag(SubspatialVersionSpace, loss='GREEDY' , **lvs_global_params)
ProdLVS   = Tag(SubspatialVersionSpace, loss='PRODUCT', **lvs_global_params)

# LVS + Optimizations: VS + rounding + decomposition + optimizations (caching + 'opt' strategy)
vs_global_params = {'decompose': True, 'n_samples': 16, 'warmup': 100, 'thin': 100, 'rounding': True, 'rounding_cache': True, 'rounding_options': {'strategy': 'opt', 'z_cut': True, 'sphere_cuts': True}}
OptVS =        Tag(KernelVersionSpace, **vs_global_params)
GreedyFactVS = Tag(SubspatialVersionSpace, loss='GREEDY' , **vs_global_params)
SquareFactVS = Tag(SubspatialVersionSpace, loss='SQUARE' , **vs_global_params)
ProdFactVS =   Tag(SubspatialVersionSpace, loss='PRODUCT', **vs_global_params)

# No Cat
GreedyFactVSNoCat = Tag(SubspatialVersionSpace, loss='GREEDY' , numerical_only=True, **vs_global_params)
ProdFactVSNoCat =   Tag(SubspatialVersionSpace, loss='PRODUCT', numerical_only=True, **vs_global_params)

# Entropy reduction VS
entropy_global_params = {'decompose': True, 'warmup': 100, 'thin': 100, 'rounding': True, 'rounding_cache': True, 'rounding_options': {'strategy': 'opt', 'z_cut': True, 'sphere_cuts': True}}
Entropy = Tag(EntropyReductionLearner, data_sample_size=256, n_samples=32, **entropy_global_params)

# SwapLearner
swap_global_params = {'num_subspaces': 10, 'prune_threshold': 0.99, 'prune': True}
def Swap(swap_iter, penalty=1e-4, train_sample_size=None, retries=1, prune=True, refine_max_iter=10000) -> Tag:
    return Tag(SimplifiedSwapLearner, swap_iter=swap_iter, penalty=penalty,
               train_sample_size=train_sample_size, retries=retries, prune=prune, refine_max_iter=refine_max_iter)

active_learners_list = [
    # STATE-OF-THE-ART VS ALGORITHMS
    #SM,
    #ALUMA,
    #QBD,

    # DUAL SPACE MODEL
    #DSM,
    #FactDSM,

    # KVS
    #KVS,        # VS + rounding
    #GreedyKVS,  # FactVS + rounding
    #ProdKVS,    # FactVS + rounding

    # LVS
    #LVS,        # VS + rounding + decomposition
    #GreedyLVS,  # FactVS + rounding + decomposition
    #ProdLVS,    # FactVS + rounding + decomposition

    # Opt VS
    #OptVS,  # VS + rounding + decomposition + optimizations
    #GreedyFactVS, # FactVS + GREEDY loss
    #SquareFactVS, # FactVS + SQUARE loss
    #ProdFactVS,   # FactVS + PRODUCT loss

    # No Cat
    #GreedyFactVSNoCat,  # FactVS + PRODUCT loss + No categorical optimization
    #ProdFactVSNoCat,    # FactVS + PRODUCT loss + No categorical optimization

    # Entropy
    #Entropy,

    # SwapLearner
    Tag(SimplifiedSwapLearner, swap_iter=50, penalty=1e-4, train_sample_size=200000, retries=1, refine_max_iter=10),
    Tag(SimplifiedSwapLearner, swap_iter=50, penalty=0, train_sample_size=200000, retries=1, refine_max_iter=10),

    Swap(swap_iter=100, penalty=1e-4, train_sample_size=1000000, retries=5, refine_max_iter=10),
    Swap(swap_iter=100, penalty=1e-4, train_sample_size=1000000, retries=5, refine_max_iter=25),
    Swap(swap_iter=100, penalty=1e-4, train_sample_size=1000000, retries=5, refine_max_iter=50),
    Swap(swap_iter=100, penalty=1e-4, train_sample_size=1000000, retries=5, refine_max_iter=100),
]

# RUN PARAMS
REPEAT = 10
NUMBER_OF_ITERATIONS = 200  # number of points to be labeled by the user
SEEDS = [i for i in range(REPEAT)]

SUBSAMPLING = 50000  # None

CALLBACK_SKIP = 5
score_functions = ['true_positive', 'true_negative', 'false_positive', 'false_negative', 'precision', 'recall', 'fscore']
CALLBACKS = [
    Tag(training_classification_metrics, score_functions=score_functions, prefix='train_'),
    Tag(classification_metrics, score_functions=score_functions),
    #Tag(three_set_metric),
    Tag(compute_factorization),
]

INITIAL_SAMPLER = Tag(stratified_sampler, pos=1, neg=1, neg_in_all_subspaces=False)

CONVERGENCE_CRITERIA = [
    Tag(max_iter_reached, max_iters=NUMBER_OF_ITERATIONS),
    #Tag(metric_reached_threshold, metric='tsm', threshold=1.0),
]

NOISE_INJECTOR = None
#NOISE_INJECTOR = Tag(random_noise_injector, noise=0, skip_initial=0)

#############################################
# CHECKS AND REMINDER
#############################################
if not task_list:
    raise ValueError("Tasks list cannot be empty.")

if not active_learners_list:
    raise ValueError("Active learners list cannot be empty")

assert_positive_integer(SUBSAMPLING, 'SUBSAMPLING', allow_none=True)
assert_positive_integer(REPEAT, 'REPEAT')
assert_positive_integer(CALLBACK_SKIP, 'CALLBACK_SKIP')


print("""-----------INFO--------------
TASKS: {}
ACTIVE_LEARNERS: {}
SUBSAMPLING: {}
REPEAT: {}
INITIAL_SAMPLER: {}
CALLBACKS: {}
CALLBACK_SKIP: {}
CONVERGENCE: {}
NOISE_INJECTOR: {}
-----------------------------""".format(
    task_list, active_learners_list, SUBSAMPLING, REPEAT, INITIAL_SAMPLER,
    CALLBACKS, CALLBACK_SKIP, CONVERGENCE_CRITERIA, NOISE_INJECTOR
))

#############################################
# EXPERIMENTS
#############################################
root_folder = RootFolder()

# write configuration to disk + set folder structure
for TASK in task_list:
    for ACTIVE_LEARNER in active_learners_list:
        folder = root_folder.get_experiment_folder(TASK, str(ACTIVE_LEARNER))

        config = {
            'task': TASK,
            'active_learner': ACTIVE_LEARNER.to_json(),
            'repeat': REPEAT,
            'seeds': SEEDS,
            'initial_sampling': INITIAL_SAMPLER.to_json(),
            'subsampling': SUBSAMPLING,
            'callbacks': [c.to_json() for c in CALLBACKS],
            'callback_skip': CALLBACK_SKIP,
            'convergence_criteria': [c.to_json() for c in CONVERGENCE_CRITERIA],
            'noise_injector': None if not NOISE_INJECTOR else NOISE_INJECTOR.to_json(),
        }

        # save config to disk
        folder.write_config(config)

# run all experiments
run_all_experiments(root_folder)
