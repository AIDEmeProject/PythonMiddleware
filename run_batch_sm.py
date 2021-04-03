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
from collections import defaultdict
from time import perf_counter
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.svm import SVC

from aideme.active_learning.factorization import LinearFactorizationLearner, KernelFactorizationLearner
from aideme.active_learning.factorization.optimization import Adam
from aideme.active_learning.factorization.penalty import SparseGroupLassoPenalty
from aideme.io import read_task


def read_data(task: str):
    full_data = read_task(task, distinct=True, preprocess=True, read_factorization=False)
    data, labels = full_data['data'], full_data['labels']
    return data.values, labels.values, full_data['factorization_info']['one_hot_groups']


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


def build_train_test_shuffle(train_size: Union[int, float], repeat: int = 1, stratify: bool = False, seed: Optional[int] = None):
    shuffle_class = StratifiedShuffleSplit if stratify else ShuffleSplit
    return shuffle_class(n_splits=repeat, train_size=train_size, random_state=seed)


def build_train_test_kfolds(k: int, repeats: int = 1, stratify: bool = False, seed: Optional[int] = None):
    kfold_class = RepeatedStratifiedKFold if stratify else RepeatedKFold
    return kfold_class(n_splits=k, n_repeats=repeats, random_state=seed)


def run_cross_validation(learner, X, y, cv: RepeatedStratifiedKFold, fit_params = None):
    if fit_params is None:
        fit_params = {}

    metrics = defaultdict(list)

    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        t0 = perf_counter()
        learner.fit(X_train, y_train, **fit_params)
        dt = perf_counter() - t0

        metrics['train_score'].append(f1_score(y_train, learner.predict(X_train)))
        metrics['test_score'].append(f1_score(y_test, learner.predict(X_test)))
        metrics['fit_time'].append(dt)

    return metrics


# EXPERIMENT CONFIGS
TASK_LIST = [
    #'sdss_q5', 'sdss_q6', 'sdss_q7', 'sdss_q8', 'sdss_q9', 'sdss_q10', 'sdss_q11',
    #'user_study_01', 'user_study_02', 'user_study_03', 'user_study_04', 'user_study_05', 'user_study_06',
    #'user_study_07', 'user_study_08', 'user_study_09', 'user_study_10', 'user_study_11', 'user_study_12',
    #'user_study_13', 'user_study_14', 'user_study_15', 'user_study_16', 'user_study_17', 'user_study_18',
]
TRAIN_SIZE = 0.7
STRATIFY = True
K_FOLD = 5  # set to 0 or None to turn off
REPEATS = 1
SEED = None

# print
print("""TASK_LIST = {}
TRAIN_SIZE = {}
STRATIFY = {}
REPEAT = {}
K_FOLD = {}
""".format(TASK_LIST, TRAIN_SIZE, STRATIFY, REPEATS, K_FOLD))

# FLM
linear_optimizer = get_optimizer(step_size=0.5, max_iter=2000, batch_size=None, adapt_step_size=False)
penalty = SparseGroupLassoPenalty(l1_penalty=1e-5, l2_sqrt_penalty=1e-5)
linear_fit_params = {'factorization': 10}

# FKM
kernel_optimizer = get_optimizer(step_size=0.5, max_iter=4000, batch_size=None, adapt_step_size=False)
kernel_fit_params = {'factorization': 2}

LEARNERS = [
    #('svm', SVC(C=1e3, gamma='auto'), None),
    #('unpen-flm', LinearFactorizationLearner(optimizer), linear_fit_params),
    #('pen-flm', LinearFactorizationLearner(optimizer, penalty_term=penalty), linear_fit_params),
    #('fkm', KernelFactorizationLearner(kernel_optimizer, nystroem_components=100), kernel_fit_params),
    #('fact-flm', LinearFactorizationLearner(optimizer), linear_fit_params),
    #('fact-fkm', KernelFactorizationLearner(optimizer, nystroem_components=100), kernel_fit_params),
]

for tag, learner, fit_params in LEARNERS:
    learner_result = {}

    for TASK in TASK_LIST:
        # read data
        X, y, groups = read_data(TASK)
        if isinstance(learner, LinearFactorizationLearner):
            learner.feature_groups = groups

        if K_FOLD is not None and K_FOLD >= 2:
            cv = build_train_test_kfolds(k=K_FOLD, repeats=REPEATS, stratify=STRATIFY, seed=SEED)
        else:
            cv = build_train_test_shuffle(train_size=TRAIN_SIZE, repeat=REPEATS, stratify=STRATIFY, seed=SEED)

        scores = run_cross_validation(learner, X, y, cv=cv, fit_params=fit_params)

        learner_result[TASK] = [
            np.mean(scores['train_score']),
            np.std(scores['train_score']),
            np.mean(scores['test_score']),
            np.std(scores['test_score']),
            np.mean(scores['fit_time']),
            np.std(scores['fit_time']),
        ]

        print(tag, TASK, np.mean(scores['train_score']), np.mean(scores['test_score']),)

    columns = ['train_score_avg', 'train_score_std', 'test_score_avg', 'test_score_std', 'fit_time_avg', 'fit_time_std']
    df = pd.DataFrame.from_dict(learner_result, orient='index', columns=columns)
    df.to_csv('./batch_experiments/{}.tsv'.format(tag), sep='\t', index_label='Query')
