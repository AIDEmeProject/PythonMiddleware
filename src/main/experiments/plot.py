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

    def plot_all_runs(self, data_tag, learner_tag):
        pass

    def set_axis_parameters(self, ax, data_tag, metric):
        ax.set_title(data_tag)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        ax.set_ylim([0, 1])

    def plot_comparisons(self, dataset_tags, learner_tags, metric, iter_lim=None):
        # get axis for plotting
        axs = self.get_axis(len(dataset_tags), 1)

        for i, data_tag in enumerate(dataset_tags):
            for k, marker, learner_tag in zip(np.linspace(0, 1, len(learner_tags)), Line2D.filled_markers, learner_tags):
                data_folder = self.dir_manager.get_data_folder(data_tag, learner_tag)
                avg = data_folder.read_average(metric)

                iter_lim = len(avg) if iter_lim is None else iter_lim
                step = max(int(len(avg) / 50), 1)

                avg = avg.loc[avg.index.isin(range(0, iter_lim, step))]

                ax = axs[i][0]
                self.set_axis_parameters(ax, data_tag, metric)
                ax.plot(avg.index, avg['average'], label=learner_tag, marker=marker, markersize=5, color=plt.cm.jet(k))
                ax.legend(bbox_to_anchor=(1.1, 1.0))

            yield axs[i][0]
        #plt.legend(loc='best')
        #plt.savefig(self.dir_manager.experiment_folder + '/plot_{0}.eps'.format(metric), dpi=1000)

