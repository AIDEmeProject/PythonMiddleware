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
