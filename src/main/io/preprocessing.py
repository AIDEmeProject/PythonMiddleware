from pandas import get_dummies


def one_hot_encoding(df):
    """ Find the one-hot-encoding of a pandas dataframe """
    return get_dummies(df, drop_first=False)


def standardize(df):
    """ Standardize a pandas dataframe. """
    mean = df.mean()
    std = df.std()

    if any(std == 0):
        cte_columns = df.columns()[std == 0]
        raise ValueError("Found zero standard deviation at columns: " + ','.join(cte_columns))

    return (df - mean) / std


def preprocess(data, preprocess_list):
    """
    Preprocess a dataframe from a list of pre-processing steps
    :param data: pandas dataframe of data
    :param preprocess_list: list of strings containing the name of preprocessing steps to perform.
    """
    for function_name in preprocess_list:
        data = eval(function_name)(data)

    return data
