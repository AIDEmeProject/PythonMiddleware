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
from aideme.active_learning.factorization.learn import prune_irrelevant_subspaces, compute_relevant_attributes, compute_factorization
from aideme.active_learning.factorization.optimization import Adam
from aideme.active_learning.factorization.penalty import SparseGroupLassoPenalty
from aideme.io import read_task


class RefinedLearner:
    def __init__(self, learner_w_penalty: LinearFactorizationLearner, refining_learner: Union[LinearFactorizationLearner, KernelFactorizationLearner], num_subspaces: int = 10):
        self.learner_w_penalty = learner_w_penalty
        self.refining_learner = refining_learner
        self.num_subspaces = num_subspaces

    @property
    def feature_groups(self):
        return self.learner_w_penalty.feature_groups

    @feature_groups.setter
    def feature_groups(self, value):
        self.learner_w_penalty.feature_groups = value

    def fit(self, X, y):
        # fit penalized learner
        self.learner_w_penalty.fit(X, y, factorization=self.num_subspaces)

        # compute factorization
        pruned_learner = prune_irrelevant_subspaces(X, self.learner_w_penalty)
        relevant_attrs = compute_relevant_attributes(pruned_learner)
        self.subspaces = [list(np.where(s)[0]) for s in relevant_attrs]
        self.factorization = compute_factorization(relevant_attrs)

        # refine pruned learner
        is_kernel = isinstance(self.refining_learner, KernelFactorizationLearner)
        fact = self.factorization if is_kernel else self.subspaces
        x0 = None if is_kernel else pruned_learner.weight_matrix
        self.refining_learner.fit(X, y, factorization=fact, x0=x0)

    def predict(self, X):
        return self.refining_learner.predict(X)


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
TRAIN_SIZE = 0.5
STRATIFY = True
K_FOLD = 5  # set to 0 or None to turn off
REPEATS = 100
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
linear_model_wo_penalty = LinearFactorizationLearner(linear_optimizer)
linear_model_w_penalty = LinearFactorizationLearner(linear_optimizer, penalty_term=penalty)

# FKM
kernel_optimizer = get_optimizer(step_size=0.5, max_iter=4000, batch_size=None, adapt_step_size=False)
kernel_fit_params = {'factorization': 3}
kernel_model = KernelFactorizationLearner(kernel_optimizer, nystroem_components=100)

LEARNERS = [
    #('svm', SVC(C=1e3, gamma='auto'), None),
    #('unpen-flm', linear_model_wo_penalty, linear_fit_params),
    #('pen-flm', linear_model_w_penalty, linear_fit_params),
    #('fact-flm', RefinedLearner(linear_model_w_penalty, linear_model_wo_penalty, 10), None),
    #('fkm', kernel_model, kernel_fit_params),
    #('fact-fkm', RefinedLearner(linear_model_w_penalty, kernel_model, 10), None),
]

for tag, learner, fit_params in LEARNERS:
    learner_result = {}

    for TASK in TASK_LIST:
        # read data
        X, y, groups = read_data(TASK)
        if hasattr(learner, 'feature_groups'):
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

        print(tag, TASK, np.mean(scores['train_score']), np.mean(scores['test_score']))

    columns = ['train_score_avg', 'train_score_std', 'test_score_avg', 'test_score_std', 'fit_time_avg', 'fit_time_std']
    df = pd.DataFrame.from_dict(learner_result, orient='index', columns=columns)
    df.to_csv('./batch_experiments/{}.tsv'.format(tag), sep='\t', index_label='Query')
