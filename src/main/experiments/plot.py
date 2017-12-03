import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

        if metric in ['F-Score', 'Accuracy', 'Precision', 'Recall']:
            ax.set_ylim([0, 1])

    def plot_comparisons(self, dir_manager, dataset_tags, learner_tags, metrics=None):
        # get axis for plotting
        axs = self.get_axis(len(dataset_tags), len(metrics))

        for i, data_tag in enumerate(dataset_tags):
            for marker, learner_tag in zip(Line2D.filled_markers, learner_tags):
                avg = dir_manager.read_average(data_tag, learner_tag, metrics)
                avg = avg.iloc[range(0, len(avg), int(len(avg) / 50))]
                for j, metric in enumerate(avg.columns):
                    ax = axs[i][j]
                    self.set_axis_parameters(ax, data_tag, metric)
                    ax.plot(avg.index, avg[metric], label=learner_tag, marker=marker, markersize=5)
                    ax.legend(loc='best')

        # plt.savefig("test.eps", dpi=1000)


