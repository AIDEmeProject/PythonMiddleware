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
from __future__ import annotations

from typing import Optional, TYPE_CHECKING, List

import numpy as np

from aideme.active_learning import ActiveLearner
from aideme.active_learning.version_space.subspace import SubspatialVersionSpace, SubspatialSimpleMargin
from aideme.utils import assert_positive_integer, assert_in_range, metric_logger, assert_positive
from .learn import prune_irrelevant_subspaces, compute_factorization_and_partial_labels
from .linear import LinearFactorizationLearner
from .optimization import FISTA

if TYPE_CHECKING:
    from aideme.explore import PartitionedDataset
    from aideme.active_learning import FactorizedActiveLearner


class SwapLearner(ActiveLearner):
    def __init__(self, active_learner: ActiveLearner,
                 swap_model: LinearFactorizationLearner, refining_model: Optional[LinearFactorizationLearner] = None, num_subspaces: int = 10, retries: int = 1,
                 swap_iter: int = 100, train_on_prediction: bool = True, train_sample_size: Optional[int] = None,
                 prune: bool = True, prune_threshold: float = 0.99,
                 fact_model: Optional[FactorizedActiveLearner] = None,  user_fact: List[List[int]] = None, compute_fact_every: int = 5, fact_repeat: int = 2,
                 fact_l1_penalty: float = 1e-4, fact_l2_sqrt_penalty: float = 1e-4):
        assert_positive_integer(swap_iter, 'swap_iter')
        assert_positive_integer(num_subspaces, 'num_subspaces')
        assert_positive_integer(retries, 'retries')
        assert_positive_integer(train_sample_size, 'train_sample_size', allow_none=True)
        assert_in_range(prune_threshold, 'prune_threshold', 0, 1)

        self._active_learner = active_learner

        self._swap_model = swap_model
        self._refining_model = swap_model if refining_model is None else refining_model
        self._swap_iter = swap_iter
        self._num_subspaces = num_subspaces
        self._retries = retries
        self._train_on_prediction = train_on_prediction
        self._train_sample_size = train_sample_size
        self._prune = prune
        self._prune_threshold = prune_threshold

        self._fact_model = fact_model
        self._fact_manager = FactorizationManager(
            user_fact=user_fact, compute_every=compute_fact_every, stable_count=fact_repeat,
            l1_penalty=fact_l1_penalty, l2_sqrt_penalty=fact_l2_sqrt_penalty
        ) if self._fact_model is not None else None

        self.__is_full_fact_phase = False
        self.__it = 0

    def clear(self) -> None:
        self._active_learner.clear()
        self._swap_model.clear()
        self._refining_model.clear()
        if self._fact_model:
            self._fact_model.clear()
            self._fact_manager.clear()
        self.__is_full_fact_phase = False
        self.__it = 0

    def set_user_factorization(self, user_factorization):
        if self._fact_manager is not None:
            self._fact_manager.set_user_fact(user_factorization)

    @property
    def is_active_learning_phase(self) -> bool:
        return self.__it <= self._swap_iter

    @property
    def is_factorization_phase(self) -> bool:
        return not self.is_active_learning_phase

    @property
    def is_swap_iter(self) -> bool:
        return self.__it == self._swap_iter + 1

    @property
    def linear_model(self) -> Optional[LinearFactorizationLearner]:
        return None if self.is_active_learning_phase else self._refining_model

    def fit_data(self, data: PartitionedDataset) -> None:
        self.__it += 1

        if self.is_active_learning_phase:
            self._active_learner.fit_data(data)

        elif self.is_swap_iter:
            self.__fit_swap_model(data)
            self._refining_model._weights = self._swap_model.weights
            self._refining_model._bias = self._swap_model.bias
            self.__update_factorization(data)

        elif self.__is_full_fact_phase:
            # TODO: retrain linear factorization? Pro: possibly more accurate partial labels, Con: retraining from scratch can be slow
            self._fact_model.fit_data(data)

        else:
            self.__fit_refining_model(data)
            self.__update_factorization(data)

    def __update_factorization(self, data: PartitionedDataset) -> None:
        if not self.__is_fact_update_iter():
            return

        factorization, y_partial = self._fact_manager.compute_factorization(data, self.linear_model)
        self._fact_manager.update(factorization)

        if self._fact_manager.can_switch_to_full_factorization(factorization):
            idx = self._fact_manager.fact_reindexing
            factorization, y_partial = [factorization[i] for i in idx], y_partial[:, idx]

            data.set_partial_labels(y_partial)
            self._fact_model.set_factorization_structure(partition=factorization, mode='numerical')

            self._fact_model.fit_data(data)
            self.__is_full_fact_phase = True

        metric_logger.log_metric('factorization', factorization)

    def __is_fact_update_iter(self) -> bool:
        return self._fact_manager is not None and (self.__it - self._swap_iter - 1) % self._fact_manager.update_every == 0

    def __fit_swap_model(self, data: PartitionedDataset) -> None:
        if self._train_on_prediction:
            X = data.sample(self._train_sample_size)
            y = self._active_learner.predict(X)
        else:
            X, y = data.training_set()

        self._swap_model.fit(X, y, self._num_subspaces, retries=self._retries, x0=None)

    def __fit_refining_model(self, data: PartitionedDataset) -> None:
        X, y = data.training_set()
        self._refining_model.fit(X, y, self._refining_model.num_subspaces, x0=self._refining_model.weight_matrix)

        if self._prune:
            self._refining_model = prune_irrelevant_subspaces(data.data, self._refining_model, threshold=self._prune_threshold)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.is_active_learning_phase:
            return self._active_learner.predict(X)
        elif self.__is_full_fact_phase:
            return self._fact_model.predict(X)
        else:
            return self._refining_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.is_active_learning_phase:
            return self._active_learner.predict_proba(X)
        elif self.__is_full_fact_phase:
            return self._fact_model.predict_proba(X)
        else:
            return self._refining_model.predict_proba(X)

    def rank(self, X: np.ndarray) -> np.ndarray:
        if self.is_active_learning_phase:
            return self._active_learner.rank(X)
        elif self.__is_full_fact_phase:
            return self._fact_model.rank(X)
        else:
            return np.abs(self._refining_model.predict_proba(X) - 0.5)


class FactorizationManager:
    def __init__(self, user_fact: List[List[int]], compute_every: int = 5, stable_count: int = 3,
                 l1_penalty: float = 1e-4, l2_sqrt_penalty: float = 1e-4):
        assert_positive_integer(compute_every, 'compute_every')
        assert_positive_integer(stable_count, 'stable_count')
        assert_positive(l1_penalty, 'l1_penalty')
        assert_positive(l2_sqrt_penalty, 'l2_sqrt_penalty')

        self.update_every = compute_every
        self._stable_count = stable_count
        self.set_user_fact(user_fact)
        self.l1_penalty = l1_penalty
        self.l2_sqrt_penalty = l2_sqrt_penalty

        self.__fact_count = 0
        self.__fact_prev = None
        self.fact_reindexing = None

    def set_user_fact(self, user_fact):
        if user_fact is not None:
            user_fact = sorted([sorted(s) for s in user_fact])
        self._user_fact = user_fact

    def clear(self) -> None:
        self.__fact_count = 0
        self.__fact_prev = None
        self.fact_reindexing = None

    def compute_factorization(self, data, linear_model):
        return compute_factorization_and_partial_labels(data, linear_model, l1_penalty=self.l1_penalty, l2_sqrt_penalty=self.l2_sqrt_penalty)

    def update(self, factorization: List[List[int]]) -> None:
        if self.__fact_prev is None or factorization != self.__fact_prev:
            self.__fact_prev = factorization
            self.__fact_count = 0
        else:
            self.__fact_count += 1

    def can_switch_to_full_factorization(self, factorization: List[List[int]]) -> bool:
        return self.__is_fact_stable() and self.__is_fact_compatible(factorization)

    def __is_fact_stable(self) -> bool:
        return self.__fact_count >= self._stable_count

    def __is_fact_compatible(self, factorization: List[List[int]]) -> bool:
        if len(factorization) != len(self._user_fact):
            return False

        mapping, inv_mapping = {}, {}
        for i, subspace in enumerate(factorization):
            for j, user_subspace in enumerate(self._user_fact):
                if set(subspace).issubset(user_subspace):
                    # assert each subspace is compatible with a single user subspace
                    if i in mapping or j in inv_mapping:
                        return False
                    mapping[i] = j
                    inv_mapping[j] = i

            if i not in mapping:  # subspace is not compatible with any user subspace
                return False

        self.fact_reindexing = [mapping[i] for i in range(len(mapping))]

        return True


class SimplifiedSwapLearner(SwapLearner):
    SWAP_DEFAULT_PARAMS = {'step_size': 0.1, 'max_iter': 200, 'batch_size': 200, 'adapt_step_size': True, 'adapt_every': 5}
    SWAP_CARS = {'step_size': 0.1, 'max_iter': 5000, 'batch_size': None, 'adapt_step_size': False}
    SWAP_EXP_DECAY = {'step_size': 0.1, 'max_iter': 250, 'batch_size': 250, 'adapt_step_size': True, 'adapt_every': 20, 'exp_decay': 0.9}
    REFINE_DEFAULT_PARAMS = {'step_size': 0.1, 'batch_size': None, 'adapt_step_size': False}
    FISTA_DEFAULT_PARAMS = {'step_size': 5, 'batch_size': None, 'adapt_step_size': False}
    FISTA_CARS = {'step_size': 0.05, 'batch_size': None, 'adapt_step_size': False}
    VS_DEFAULT_PARAMS = {'decompose': True, 'n_samples': 16, 'warmup': 100, 'thin': 100, 'rounding': True, 'rounding_cache': True, 'rounding_options': {'strategy': 'opt', 'z_cut': True, 'sphere_cuts': True}}
    FACT_SM_PARAMS = {'C': 1e5}
    FACT_VS_PARAMS = {'loss': 'PRODUCT', 'n_samples': 16, 'warmup': 100, 'thin': 100, 'rounding': True, 'rounding_cache': False}

    def __init__(self, swap_iter: int = 50, penalty: float = 1e-4, train_sample_size: Optional[int] = 500000,
                 num_subspaces: int = 10, retries: int = 1, prune: bool = True, prune_threshold: float = 0.99, refine_max_iter: int = 100,
                 use_vs: bool = True, use_fact_vs: bool = False, use_exp_decay: float = True, use_fista: bool = True, fact_penalty: float = 5e-4,
                 cars: bool = False):
        from ...active_learning import SimpleMargin, KernelVersionSpace
        if use_vs:
            active_learner = KernelVersionSpace(**self.VS_DEFAULT_PARAMS)
        else:
            active_learner = SimpleMargin(C=1e6)

        params = self.SWAP_CARS if cars else self.SWAP_EXP_DECAY if use_exp_decay else self.SWAP_DEFAULT_PARAMS
        swap_model_optimizer = self.get_optimizer(N=train_sample_size, **params)
        swap_model = LinearFactorizationLearner(optimizer=swap_model_optimizer)

        params = self.FISTA_CARS if cars else self.FISTA_DEFAULT_PARAMS if use_fista else self.REFINE_DEFAULT_PARAMS
        if use_fista:
            refined_model_optimizer = FISTA(max_iter=refine_max_iter, **params)
        else:
            refined_model_optimizer = self.get_optimizer(max_iter=refine_max_iter, **params)

        refined_model = LinearFactorizationLearner(optimizer=refined_model_optimizer, l2_sqrt_penalty=penalty, l1_penalty=penalty)

        fact_model = SubspatialVersionSpace(**self.FACT_VS_PARAMS) if use_fact_vs else SubspatialSimpleMargin(**self.FACT_SM_PARAMS)

        super().__init__(active_learner=active_learner, swap_model=swap_model, refining_model=refined_model, num_subspaces=num_subspaces, retries=retries,
                         swap_iter=swap_iter, train_sample_size=train_sample_size,
                         prune=prune, prune_threshold=prune_threshold,
                         fact_model=fact_model,
                         compute_fact_every=5, fact_repeat=2, fact_l1_penalty=fact_penalty, fact_l2_sqrt_penalty=fact_penalty)

    @staticmethod
    def get_optimizer(step_size: float, max_iter: int, batch_size: Optional[int] = None, adapt_step_size: bool = False,
                      adapt_every: int = 1, exp_decay: float = 0, N: Optional[int] = None):
        from .optimization import Adam
        options = {
            'step_size': step_size, 'max_iter': max_iter, 'exp_decay': exp_decay,
            'batch_size': batch_size, 'adapt_step_size': adapt_step_size,  'adapt_every': adapt_every,
            'gtol': 0, 'rel_tol': 0, 'verbose': False  # assert only max_iter is taken into account for convergence
        }

        if batch_size:
            from math import ceil
            iters_per_epoch = ceil(N / batch_size)
            options['adapt_every'] *= iters_per_epoch
            options['max_iter'] *= iters_per_epoch

        return Adam(**options)
