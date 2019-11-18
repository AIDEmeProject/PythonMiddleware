import numpy as np
import matplotlib.pyplot as plt


# TODO: change signature to be compatible with new callback interface (no labeled_indexes param)
def plot_learner_prediction(X, y, learner, labeled_indexes):
    """
        Make contour plot of learner predictions (for 2D data only). Contour lines are either probabilities (if available)
        or class label predictions otherwise.

        :param X: data matrix
        :param y: labels
        :learner: Active Learning model
        :param labeled_indexes: labeled data indexes
    """
    if X.shape[1] != 2:
        raise ValueError("Only two-dimensional datasets supported.")

    # subsample large datasets
    if len(X) > 50000:
        idx = labeled_indexes + list(np.random.choice(X.shape[0], 10000, replace=False))
        X = X[idx]
        y = y[idx]
        labeled_indexes = range(len(labeled_indexes))

    # contour plot
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 300)

    xx, yy = np.meshgrid(xs, ys)
    try:
        Z = learner.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    except:
        Z = learner.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    contour = plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu, vmin=0, vmax=1, levels=np.linspace(0, 1, 11))
    plt.colorbar(contour, ticks=np.linspace(0, 1, 11))

    # plot data points
    plt.scatter(X[:, 0], X[:, 1], c='k', s=10 / len(X))

    # plot labeled data
    plt.scatter(X[labeled_indexes, 0], X[labeled_indexes, 1],
                c=['b' if lb == 1 else 'r' for lb in y[labeled_indexes]], s=10)

    plt.show()


def plot_polytope(xlim=None, ylim=None, grid_size=800):
    def make_plot(X, y, learner):
        xlims = [X[:, 0].min(), X[:, 0].max()] if xlim is None else xlim
        ylims = [X[:, 1].min(), X[:, 1].max()] if ylim is None else ylim

        plt.figure(figsize=(10, 8))

        xx, yy = np.meshgrid(np.linspace(xlims[0], xlims[1], grid_size), np.linspace(ylims[0], ylims[1], grid_size))
        Z = learner.polytope_model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.5, vmin=0, vmax=1, levels=[0, 0.4, 0.9, 1], colors=['r', 'g', 'b', 'k'])

        # pos = learner.polytope_model.positive_region.vertices
        # plt.scatter(pos[:, 0], pos[:, 1], c='b', s=50)

        # neg = np.array([x._vertex for x in learner.polytope_model.negative_region])
        # plt.scatter(neg[:, 0], neg[:, 1], c='r', s=50)

        plt.xlim(xlims)
        plt.ylim(ylims)

        plt.show()

    return make_plot
