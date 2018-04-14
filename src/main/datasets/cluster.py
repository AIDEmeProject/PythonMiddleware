from ..config import read_dataset

def cluster_query(clusters):
    data = read_dataset('sdss_1_percent', columns=['ra', 'dec'], keep_duplicates=False)
    data = (data - data.mean())/data.std()

    from ..user import DummyUser
    from pandas import read_csv
    labels = read_csv('resources/ra_dec_cluster.csv', sep='\t', index_col='objid', dtype='float')
    labels = 2*labels['cluster'].isin(clusters) - 1
    user = DummyUser(labels, 100)

    return data, user