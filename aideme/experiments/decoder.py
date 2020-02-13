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

from typing import Optional, TYPE_CHECKING

from .. import PoolBasedExploration, LabeledSet, ActiveLearner, FactorizedActiveLearner
from ..io import read_task

if TYPE_CHECKING:
    from ..utils import InitialSampler, Config, Callback, Convergence


def read_training_set(task: str):
    config = read_task(task, read_factorization=True)

    df, final_labels, fact_info = config['data'], config['labels'], config.get('factorization_info', {})
    partial_labels = fact_info.pop('partial_labels', None)

    labeled_set = LabeledSet(final_labels.values, partial_labels, final_labels.index)

    return df.values, labeled_set, fact_info


def build_exploration_object(config: Config, labeled_set: LabeledSet) -> PoolBasedExploration:
    initial_sampler = decode_initial_sampler(config.get('initial_sampling', None), labeled_set)

    callbacks_config = config.get('callbacks', [])
    callbacks = [decode_callback(conf, labeled_set) for conf in callbacks_config]

    convergence_config = config.get('convergence_criteria', [])
    convergence_criteria = [decode_convergence(conf) for conf in convergence_config]

    return PoolBasedExploration(
        initial_sampler=initial_sampler, subsampling=config['subsampling'],
        callback=callbacks, callback_skip=config['callback_skip'], print_callback_result=config['print_callback_results'],
        convergence_criteria=convergence_criteria
    )


def decode_active_learner(config: Config, factorization_info: Config) -> ActiveLearner:
    import aideme.active_learning
    active_learner_class = getattr(aideme.active_learning, config['name'])

    # decode nester Tag values in params
    params = config.get('params', {})
    for k, v in params.items():
        if isinstance(v, dict) and 'name' in v:
            params[k] = decode_active_learner(v, factorization_info)

    active_learner = active_learner_class(**params)

    if isinstance(active_learner, FactorizedActiveLearner):
        active_learner.set_factorization_structure(**factorization_info)

    return active_learner


def decode_initial_sampler(config: Config, y_true: LabeledSet) -> Optional[InitialSampler]:
    if not config:
        # TODO: log warning, since this is *probably* a mistake on the user's part
        return None

    import aideme.initial_sampling

    name, params = config['name'], config.get('params', {})
    initial_sampler = getattr(aideme.initial_sampling, name)

    if name == 'stratified_sampler':
        params['labeled_set'] = y_true

    return initial_sampler if not params else initial_sampler(**params)


def decode_callback(config: Config, labeled_set: LabeledSet) -> Callback:
    import aideme.utils.metrics

    name, params = config['name'], config.get('params', {})

    callback_function = getattr(aideme.utils.metrics, name)
    if name == 'classification_metrics':
        params['y_test'] = labeled_set.labels

    return callback_function if not params else callback_function(**params)


def decode_convergence(config: Config) -> Convergence:
    import aideme.utils.convergence

    name, params = config['name'], config.get('params', {})
    convergence_function = getattr(aideme.utils.convergence, name)

    return convergence_function if not params else convergence_function(**params)
