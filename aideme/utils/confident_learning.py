"""
Author : Enhui Huang
Time : 3/9/21 1:20 PM
This module contains helper functions for selecting confident examples from training data.
These helper functions have a fixed format of inputs and outputs as shown in the following:
    def helper(data, labels, *args):
        return (confident examples, indexes of positive examples, indexes of negative examples)
"""

import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier


def auto(data: np.ndarray, labels: np.ndarray, n_neighbors: int = 5) -> (np.ndarray, List[int], List[int]):
    """
    Utilize "auto" for detecting the label noise
    :param n_neighbors:
    :param data: data matrix
    :param labels: labels
    :return: (confident examples, indexes of positive examples, indexes of negative examples)

    References
    ----------
    .. [1] Cheng, Jiacheng, et al. "Learning with bounded instance-and label-dependent label noise.", ICML (2020).

    """
    clf = RandomForestClassifier(random_state=0)
    clf.fit(data, labels)
    psx = clf.predict_proba(data)[:, 1]  # P(y = 1|x) todo: deal with the cases where there is only one class in the

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    pos_ind, neg_ind = [], []
    for ind, neigh_ind in enumerate(neigh.kneighbors(data, return_distance=False)):
        nr_pos = sum(psx[x] for x in neigh_ind) / len(neigh_ind)
        nr_neg = 1 - sum(psx[x] for x in neigh_ind) / len(neigh_ind)
        th_pos = (1 + nr_neg) / 2
        th_neg = (1 - nr_pos) / 2

        if th_pos < psx[ind] < th_neg:
            raise ValueError('An example cannot belong to both classes!')
        if psx[ind] > th_pos:
            pos_ind.append(ind)
        if psx[ind] < th_neg:
            neg_ind.append(ind)
    return data[pos_ind], pos_ind, neg_ind


