import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


class ExperimentPlotter:
    def get_axis(self, nrows, ncols):
        fig, axs = plt.subplots(nrows, ncols)
        fig.set_size_inches(8 * ncols, 6 * nrows)

        if (nrows == 1): axs = [axs]
        if (ncols == 1): axs = [[ax] for ax in axs]
        return axs

    def plot_all_runs(self, data_tag, learner_tag):
        pass

    def set_axis_parameters(self, ax, data_tag, metric):
        ax.set_title(data_tag)
        ax.set_xlabel("# Labeled Samples")

        if metric.lower().endswith('time'):
            metric += " (s)"
        ax.set_ylabel(metric)

        if metric in ['F-Score', 'Accuracy', 'Precision', 'Recall', 'Labeled Set F-Score']:
            ax.set_ylim([0, 1])

    def plot_comparisons(self, dir_manager, dataset_tags, learner_tags, metrics=None, iter_lim=None):
        # get axis for plotting
        axs = self.get_axis(len(dataset_tags), len(metrics))

        for i, data_tag in enumerate(dataset_tags):
            for k, marker, learner_tag in zip(np.linspace(0, 1, len(learner_tags)), Line2D.filled_markers,
                                              learner_tags):
                avg = dir_manager.read_average(data_tag, learner_tag, metrics)
                iter_lim = len(avg) if iter_lim is None else iter_lim
                step = max(int(len(avg) / 50), 1)
                avg = avg.loc[avg.index.isin(range(0, iter_lim, step))]
                for j, metric in enumerate(avg.columns):
                    ax = axs[i][j]
                    self.set_axis_parameters(ax, data_tag, metric)
                    ax.plot(avg.index, avg[metric], label=learner_tag, marker=marker, markersize=5, color=plt.cm.jet(k))
                    if j == len(avg.columns) - 1:
                        ax.legend(bbox_to_anchor=(1.1, 1.0))

        # plt.savefig("test.eps", dpi=1000)
