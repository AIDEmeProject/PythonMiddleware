from math import sqrt
import matplotlib.pyplot as plt


def plot_showdown(output, times, metrics_list=None):
    data_tags = output.columns.levels[0]

    if metrics_list is None:
        metrics_list = output.columns.levels[1]
    als = output.columns.levels[2]

    fig, axs = plt.subplots(len(data_tags), len(metrics_list))#, sharex=True, sharey=True)
    if len(data_tags) == 1:
        axs = [axs]


    fig.set_size_inches(1.6*8*len(data_tags), 8*len(metrics_list))

    for i, ds in enumerate(data_tags):
        for j, name in enumerate(metrics_list):
            ax = axs[i][j]
            ax.set_title("Dataset:  " + ds.upper() + "     -     " +  "Metric:  " + name.upper())
            #if i == len(data_tags) - 1:
            #    ax.set_xlabel(name)
            #if j == 0:
            #    ax.set_ylabel(ds)

            ax.set_xlabel("# of labeled samples")
            ax.set_ylabel("Score evolution")

            for al in als:
                df = output[ds][name][al]
                x = df.index
                y = df['mean']
                error = 1.96 * df['std'] / sqrt(times)
                ax.plot(x, y, label=al)

                ax.fill_between(x, y - error, y + error, alpha=0.1)

    plt.legend()
    plt.show()
