import matplotlib.pyplot as plt


def plot_showdown(output, metrics_list=None):
    data_tags = output.columns.levels[0]

    if metrics_list is None:
        metrics_list = output.columns.levels[1]
    als = output.columns.levels[2]

    fig, axs = plt.subplots(len(data_tags), len(metrics_list))
    if len(data_tags) == 1:
        axs = [axs]
    if len(metrics_list) == 1:
        axs = [[ax] for ax in axs]

    fig.set_size_inches(8*len(metrics_list), 6*len(data_tags))

    for i, ds in enumerate(data_tags):
        for j, name in enumerate(metrics_list):
            ax = axs[i][j]
            ax.set_title(ds.upper())
            ax.set_xlabel("# of labeled samples")
            ax.set_ylabel(name.upper())
            if name.lower() in ['fscore', 'accuracy', 'precision', 'recall']:
                ax.set_ylim([0,1])

            for al in als:
                df = output[ds][name][al]
                if not df.empty:
                    x = df.index
                    y = df['mean']
                    ax.plot(x, y, label=al)

            ax.legend(loc='best')


    plt.legend()
    plt.show()
