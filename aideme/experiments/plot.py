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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


class ExperimentPlotter:
    def __init__(self, dir_manager):
        self.dir_manager = dir_manager

    def get_axis(self, nrows, ncols):
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(8 * ncols, 6 * nrows)

        if (nrows == 1): axs = [axs]
        if (ncols == 1): axs = [[ax] for ax in axs]
        return axs

    def set_axis_parameters(self, ax, data_tag, metric):
        ax.set_title(data_tag)
        ax.set_xlabel("# Labeled Samples")

        if metric.lower().endswith('time'):
            metric += " (s)"
        ax.set_ylabel(metric)

        if metric in ['average']:
            ax.set_ylim(0, 1)

    def plot_comparisons(self, dataset_tags, learner_tags, iter_lim=None):
        # get axis for plotting
        axs = self.get_axis(len(dataset_tags), 1)

        for i, data_tag in enumerate(dataset_tags):
            for k, marker, learner_tag in zip(np.linspace(0, 1, len(learner_tags)), Line2D.filled_markers, learner_tags):
                try:
                    data_folder = self.dir_manager.get_data_folder(data_tag, learner_tag)
                    avg = data_folder.read_average()

                    if 'iter' in avg.columns:
                        avg.set_index('iter', inplace=True)

                    iter_lim = len(avg) if iter_lim is None else iter_lim
                    step = max(int(len(avg) / 50), 1)

                    avg = avg.loc[avg.index.isin(range(0, iter_lim, step))]

                    for j, metric in enumerate(['average']):
                        ax = axs[i][j]

                        self.set_axis_parameters(ax, data_tag, metric)

                        ax.plot(avg.index, avg[metric], label=learner_tag, marker=marker, markersize=5, color=plt.cm.jet(k))

                        ax.legend(bbox_to_anchor=(1.1, 1.0))
                except:
                    pass
