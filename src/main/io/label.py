from pandas import Series
from pandasql import sqldf


def get_labels_from_query(data, predicate):
    """
    Compute the user's true labeling from his "true query predicate".

    :param data: pandas dataframe
    :param predicate: SQL predicate encoding the user's interest
    :return: pandas Series, indexed similarly to the data, of value 1 if positive and 0 if negative
    """

    # run query over dataframe
    query = "SELECT {0} FROM data WHERE {1}".format(data.index.name, predicate)
    query_result = sqldf(query, locals(), 'postgresql://postgres@localhost/')[data.index.name]

    # convert query result to 0-1 Series
    labels = Series(data=0., index=data.index)
    labels[labels.index.isin(query_result)] = 1.
    return labels


def get_labels_from_column(data):
    """
    Use the 'class' column in the data as labels.

    :param data: pandas dataframe
    :return: pandas Series, indexed similarly to the data, of value 1 if positive and 0 if negative
    """

    # extract label
    labels = data['class']

    # drop label column from data
    data.drop('class', axis=1, inplace=True)

    return labels


def read_labels(data, predicate=None, true_class=1.0):
    """
    Reads the labels from multiple possible sources: SQL query (predicate) or CSV file (label_column)

    :param data: pandas dataframe containing data
    :param predicate: true predicate of user's interest
    :param true_class: positive class value
    :return: pandas Series, indexed similarly to the data, of value 1 if positive and 0 if negative
    """
    if predicate is not None:
        labels = get_labels_from_query(data, predicate)

    else:
        labels = get_labels_from_column(data)

    return 1. * (labels == true_class)
