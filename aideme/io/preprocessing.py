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
from typing import Sequence

import pandas as pd


def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """ Find the one-hot-encoding of a pandas dataframe """
    return pd.get_dummies(df, drop_first=False)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """ Standardize a pandas dataframe. """
    mean = df.mean()
    std = df.std()

    if any(std == 0):
        cte_columns = df.columns()[std == 0]
        raise ValueError("Found zero standard deviation at columns: " + ','.join(cte_columns))

    return (df - mean) / std


def preprocess_data(data: pd.DataFrame, preprocess_list: Sequence[str]) -> pd.DataFrame:
    """
    Preprocess a dataframe from a list of pre-processing steps
    :param data: pandas dataframe of data
    :param preprocess_list: list of strings containing the name of preprocessing steps to perform.
    """
    for function_name in preprocess_list:
        data = eval(function_name)(data)

    return data
