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
from aideme.experiments import run_all_experiments, Tag
from aideme.experiments.folder import RootFolder
from aideme.initial_sampling import stratified_sampler
from aideme.utils import *


def get_user_study(ls):
    assert set(ls).issubset(range(1, 19))
    process = lambda x: str(x) if x >= 10 else '0' + str(x)
    return ['user_study_' + num for num in map(process, ls)]


def get_sdss(queries):
    assert set(queries).issubset(range(1, 12))
    return ["sdss_q{}".format(q) for q in queries]


# TASKS
task_list = get_sdss([1, 2, 3])
#task_list = get_user_study([7])  # range(1,13)


# LEARNERS
# State-of-the-art VS algorithms
SM = Tag(SimpleMargin, C=1e7)
ALUMA = Tag(KernelVersionSpace, n_samples=100, warmup=1000, thin=1000, rounding=False, decompose=False)

# DSM
DSM = Tag(DualSpaceModel, active_learner=SM)
FactDSM = Tag(FactorizedDualSpaceModel, active_learner=SM)

# ICDM 2019: VS + rounding (no decomposition, no optimizations)
KVS     = Tag(KernelVersionSpace    , decompose=False, n_samples=16, warmup=100, thin=100, rounding=True, rounding_cache=False, rounding_options={'strategy': 'default'})
FactKVS = Tag(SubspatialVersionSpace, decompose=False, n_samples=16, warmup=100, thin=100, rounding=True, rounding_cache=False, rounding_options={'strategy': 'default'})

# New VS algorithm: VS + rounding + decomposition (no optimizations)
LVS =     Tag(KernelVersionSpace    , decompose=True, n_samples=16, warmup=100, thin=100, rounding=True, rounding_cache=False, rounding_options={'strategy': 'default'})
FactLVS = Tag(SubspatialVersionSpace, decompose=True, n_samples=16, warmup=100, thin=100, rounding=True, rounding_cache=False, rounding_options={'strategy': 'default'})

# LVS + Optimizations: VS + rounding + decomposition + optimizations (caching + 'opt' strategy)
OptVS =     Tag(KernelVersionSpace    , decompose=True, n_samples=16, warmup=100, thin=100, rounding=True, rounding_cache=True, rounding_options={'strategy': 'opt', 'z_cut': True, 'sphere_cuts': True})
FactOptVS = Tag(SubspatialVersionSpace, decompose=True, n_samples=16, warmup=100, thin=100, rounding=True, rounding_cache=True, rounding_options={'strategy': 'opt', 'z_cut': True, 'sphere_cuts': True})

active_learners_list = [
    # STATE-OF-THE-ART VS ALGORITHMS
    #SM,
    #ALUMA,

    # DUAL SPACE MODEL
    #DSM,
    #FactDSM,

    # OUR VS ALGORITHMS
    #KVS,    # VS + rounding
    #LVS,    # VS + rounding + decomposition
    #OptVS,  # VS + rounding + decomposition + optimizations

    # FACTORIZED VS ALGORITHMS
    #FactKVS,    # FactVS + rounding
    #FactLVS,    # FactVS + rounding + decomposition
    #FactOptVS,  # FactVS + rounding + decomposition + optimizations
]

# RUN PARAMS
REPEAT = 10
NUMBER_OF_ITERATIONS = 100  # number of points to be labeled by the user
SEEDS = [i for i in range(REPEAT)]

SUBSAMPLING = 50000  # None

CALLBACK_SKIP = 2
CALLBACKS = [
    Tag(classification_metrics, score_functions=['true_positive', 'true_negative', 'false_positive', 'false_negative', 'precision', 'recall', 'fscore']),
    #Tag(three_set_metric),
]

INITIAL_SAMPLER = Tag(stratified_sampler, pos=1, neg=1, neg_in_all_subspaces=False)

CONVERGENCE_CRITERIA = [
    Tag(max_iter_reached, max_iters=NUMBER_OF_ITERATIONS),
    #Tag(metric_reached_threshold, metric='tsm', threshold=1.0),
]

NOISE_INJECTOR = None
#NOISE_INJECTOR = Tag(random_noise_injector, noise=0, skip_initial=0)


#############################################
# REMINDER
#############################################
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
