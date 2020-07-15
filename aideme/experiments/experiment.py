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

from typing import TYPE_CHECKING, Optional, Tuple

from .decoder import read_training_set, decode_active_learner, build_exploration_object
from .logger import ExperimentLogger

if TYPE_CHECKING:
    import numpy as np
    from .folder import RootFolder, ExperimentFolder
    from ..explore import LabeledSet
    from ..utils import Config, RunsType


def run_all_experiments(root_folder: RootFolder) -> None:
    """
    Runs all experiments in the given folder
    """
    logger = ExperimentLogger(root_folder)

    for task in root_folder.get_all_tasks():
        training_set = read_training_set(task)

        for learner in root_folder.get_all_learners(task):
            exp_folder = root_folder.get_experiment_folder(task, learner)

            try:
                logger.experiment_begin(task, learner)
                runs = run_experiment(exp_folder.read_config(), training_set, return_generator=True)

                for i, run in enumerate(runs):
                    filename = 'run_{0:0=2d}'.format(i + 1)

                    # run experiment and save metrics as they are computed
                    logger.run_begin()
                    df = exp_folder.save_run(run, filename + '.tmp')

                    # remove tmp file and save csv
                    exp_folder.save(df, filename + '.tsv')
                    exp_folder.delete(filename + '.tmp')

                compute_average(exp_folder)

            except Exception as e:
                logger.error(task, learner, exception=e)
                continue  # Move to next experiment

    logger.end()


def run_experiment(config: Config, training_set: Optional[Tuple[np.ndarray, LabeledSet, Config]] = None, return_generator: bool = False) -> RunsType:
    """
    Run the exploration process from a configuration object.

    :param config: the experiment configuration object
    :param training_set: the training data. If None, it will be read from database, as specified by the 'task' key in config
    :param return_generator: If True, returns a generator returning each run in the experiment. Each run is also a generator,
    returning the metrics computed after each iteration. If False, a list of runs and metrics will be returned after all
    runs have completed
    :return: The metrics computed at each run
    """
    if training_set is None:
        training_set = read_training_set(config['task'])

    data, true_labels, factorization_info = training_set

    if config.get('disable_categorical_opt', False):
        modes = factorization_info.get('mode', [])
        for i, mode in enumerate(modes):
            if mode == 'categorical':
                modes[i] = 'persist'

    # build exploration object and active learner
    active_learner = decode_active_learner(config['active_learner'], factorization_info)
    exploration = build_exploration_object(config, data, true_labels)

    # run experiment
    return exploration.run(data, true_labels, active_learner, repeat=config['repeat'], seeds=config['seeds'], return_generator=return_generator)


def compute_average(exp_folder: ExperimentFolder) -> None:
    """
    Computes the average of metrics across all runs. Only numerical values (int, float) are considered.
    """
    runs = exp_folder.read_run_files()
    to_keep = [col for col, tp in zip(runs[0].columns, runs[0].dtypes) if tp in ('int', 'float')]
    avg = sum((df[to_keep] for df in runs)) / len(runs)

    exp_folder.save(avg, 'average.tsv')
