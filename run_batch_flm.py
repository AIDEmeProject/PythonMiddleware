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
import sys
from time import perf_counter
from typing import Optional

import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from aideme.io import read_task
from aideme.active_learning.factorization import LinearFactorizationLearner, KernelFactorizationLearner
from aideme.active_learning.factorization.optimization import Adam
from aideme.active_learning.factorization.learn import prune_irrelevant_subspaces, compute_relevant_attributes, compute_factorization
from aideme.utils.random import set_random_state


def train_model_and_measure_time(linear_model, X, y, factorization, retries=1, x0=None):
    t0 = perf_counter()
    linear_model.fit(X, y, factorization, retries=retries, x0=x0)
    return perf_counter() - t0


def read_data(task: str):
    full_data = read_task(task, distinct=True, preprocess=True, read_factorization=False)
    data, labels = full_data['data'], full_data['labels']
    return data.values, labels.values


def build_train_test_sets(X, y, train_size: int, test_over_all_points: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
    if test_over_all_points:
        X_test, y_test = X, y
    return X_train, X_test, y_train, y_test


def get_optimizer(step_size: float, max_iter: int, batch_size: Optional[int] = None, adapt_step_size: bool = False,
                  adapt_every: int = 1, exp_decay: float = 0, N: Optional[int] = None):
    options = {
        'step_size': step_size, 'max_iter': max_iter, 'exp_decay': exp_decay,
        'batch_size': batch_size, 'adapt_step_size': adapt_step_size,  'adapt_every': adapt_every,
        'gtol': 0, 'rel_tol': 0, 'verbose': False
    }

    if batch_size:
        from math import ceil
        iters_per_epoch = ceil(N / batch_size)
        options['adapt_every'] *= iters_per_epoch
        options['max_iter'] *= iters_per_epoch

    return Adam(**options)


def print_results(learner, X_train, y_train, X_test, y_test, dt, prefix: str = '', file=None):
    train_score = f1_score(y_train, learner.predict(X_train))
    test_score = f1_score(y_test, learner.predict(X_test))
    print('{}train = {}, test = {}, fit time = {}'.format(prefix, train_score, test_score, dt), file=file)


# EXPERIMENT CONFIGS
TASK_LIST = ['sdss_q5', 'sdss_q9']
SUBSAMPLE = 500000
TEST_OVER_ALL = True
DO_REFINING = True
NUM_SUBSPACES = 10
RETRIES = 1
KERNEL_RETRIES = 1
SEED = 10

# print
print("""TASK_LIST = {}
SUBSAMPLE = {}
TEST_OVER_ALL = {}
DO_REFINING = {}
NUM_SUBSPACES = {}
RETRIES = {}
KERNEL_RETRIES = {}
SEED = {}
""".format(TASK_LIST, SUBSAMPLE, TEST_OVER_ALL, DO_REFINING, NUM_SUBSPACES, RETRIES, KERNEL_RETRIES, SEED))

for TASK in TASK_LIST:
    fres = open('./batch_experiments/{}.res'.format(TASK), mode='w')

    try:
        print('------ START TASK {} ------'.format(TASK), file=fres)
        L1_PENALTY = 1e-7 if TASK == 'sdss_q5' else 1e-5
        L2_SQRT_PENALTY = 1e-7 if TASK == 'sdss_q5' else 1e-5

        # RUN EXPERIMENT
        set_random_state(SEED)

        # read data
        X, y = read_data(TASK)
        X_train, X_test, y_train, y_test = build_train_test_sets(X, y, train_size=SUBSAMPLE, test_over_all_points=TEST_OVER_ALL)

        # train un-penalized model
        optimizer = get_optimizer(step_size=0.05, max_iter=200, batch_size=200, adapt_step_size=True, adapt_every=20, exp_decay=0.9, N=SUBSAMPLE)
        learner_wo_penalty = LinearFactorizationLearner(optimizer)
        dt = train_model_and_measure_time(learner_wo_penalty, X_train, y_train, factorization=NUM_SUBSPACES, retries=RETRIES)
        print_results(learner_wo_penalty, X_train, y_train, X_test, y_test, dt, prefix='LFM w/o penalty: ', file=fres)

        # train penalized model
        optimizer = get_optimizer(step_size=0.05, max_iter=200, batch_size=200, adapt_step_size=True, adapt_every=20, exp_decay=0.9, N=SUBSAMPLE)
        learner_w_penalty = LinearFactorizationLearner(optimizer, l1_penalty=L1_PENALTY, l2_sqrt_penalty=L2_SQRT_PENALTY)
        dt = train_model_and_measure_time(
            learner_w_penalty, X_train, y_train,
            factorization=NUM_SUBSPACES if DO_REFINING else learner_wo_penalty.num_subspaces,
            retries=RETRIES if DO_REFINING else 1,
            x0=None if DO_REFINING else learner_wo_penalty.weight_matrix
        )
        print_results(learner_w_penalty, X_train, y_train, X_test, y_test, dt, prefix='FLM w/ penalty: ', file=fres)

        # compute factorization
        pruned_learner = prune_irrelevant_subspaces(X_train, learner_w_penalty)
        relevant_attrs = compute_relevant_attributes(pruned_learner)
        factorization = compute_factorization(relevant_attrs)
        subspaces = [list(np.where(s)[0]) for s in relevant_attrs]
        print('# relevant subspaces:', len(relevant_attrs), file=fres)
        print('Relevant attrs:', subspaces, file=fres)
        print('Factorization:', factorization, file=fres)

        # refine pruned learner
        if DO_REFINING:
            optimizer = get_optimizer(step_size=0.05, max_iter=200, batch_size=200, adapt_step_size=True, adapt_every=20, exp_decay=0.9, N=SUBSAMPLE)
            refining_learner = LinearFactorizationLearner(optimizer)
            dt = train_model_and_measure_time(
                refining_learner, X_train, y_train,
                factorization=subspaces,
                retries=RETRIES,
                x0=pruned_learner.weight_matrix
            )
            print_results(refining_learner, X_train, y_train, X_test, y_test, dt, prefix='Refined FLM: ', file=fres)

        # kernel model
        optimizer = get_optimizer(step_size=0.05, max_iter=200, batch_size=200, adapt_step_size=True, adapt_every=20, exp_decay=0.9, N=SUBSAMPLE)
        kernel_learner = KernelFactorizationLearner(optimizer, nystroem_components=100)
        dt = train_model_and_measure_time(kernel_learner, X_train, y_train, factorization=factorization, retries=KERNEL_RETRIES)
        print_results(kernel_learner, X_train, y_train, X_test, y_test, dt, prefix='FKM: ', file=fres)
        print('------ END TASK {} ------'.format(TASK), file=fres)
    except Exception:
        print('Task {} failed\n{}'.format(TASK, sys.exc_info()[0]), file=fres)
        fres.close()
