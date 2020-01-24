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

import logging.config

import yaml

from aideme.experiments.folder import RootFolder


class ExperimentLogger:
    def __init__(self, root_folder: RootFolder):
        self._setup_logging(root_folder)
        self.logger = logging.getLogger('experiment')

        self.total = 0
        self.errors = 0

    def _setup_logging(self, root_folder) -> None:
        with open('./resources/logging.yml') as f:
            config = yaml.safe_load(f)

        config['handlers']['file']['filename'] = root_folder.log_file
        logging.config.dictConfig(config)

    def start(self, task: str, learner: str):
        self.total += 1
        self.logger.info("Starting experiment #{0}: TASK = '{1}', LEARNER = '{2}'".format(self.total, task, learner))

    def finish(self, task: str, learner: str):
        self.logger.info("Experiment #{0} ended successfully: TASK = '{1}', LEARNER = '{2}'".format(self.total, task, learner))

    def error(self, task, learner, exception):
        self.errors += 1
        self.logger.error('Error in experiment: task = "{}", learner = "{}"'.format(task, learner), exc_info=exception)

    def end(self):
        self.logger.info("Finished all experiments! Total: {0}, Success: {1}, Fail: {2}".format(
            self.total,
            self.total - self.errors,
            self.errors,
        ))
