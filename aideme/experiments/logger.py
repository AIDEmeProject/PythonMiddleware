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

import logging
from os.path import join


class ExperimentLogger:
    def __init__(self):
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.total = 0
        self.skips = 0
        self.errors = 0

    def remove_handlers(self):
        for handler in self.logger.handlers[:]:  # loop through a copy to avoid simultaneously looping and deleting
            self.logger.removeHandler(handler)

    def set_folder(self, folder_path):
        self.remove_handlers()

        handler = logging.FileHandler(join(folder_path, 'experiment.log'))
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def begin(self, data_tag, learner_tag, run_id):
        self.total += 1

        self.logger.info("Starting experiment #{0}: TASK = {1}, LEARNER = {2}, RUN = {3}".format(
            self.total,
            data_tag,
            learner_tag,
            run_id
        ))

    def skip(self):
        self.skips += 1
        self.total += 1
        self.logger.info("Skipping experiment #{0}".format(self.total))

    def error(self, exception):
        self.errors += 1
        self.logger.error(exception, exc_info=1)

    def end(self):
        self.logger.info("Finished all experiments! " + self.end_message())

    def end_message(self):
        return "Total: {0}, Success: {1}, Fail: {2}, Skipped: {3}".format(
            self.total,
            self.total - self.errors - self.skips,
            self.errors,
            self.skips
        )
