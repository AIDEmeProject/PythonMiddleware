class VersionSpace(object):
    """
        Version space of an Active Learning algorithm.

        SVM case: for linear kernel, an unit ball in higher-dimensional space
        Actboost: a polytope
    """

    def initialize(self, data):
        """ Sets object internal state given initial pool of unlabeled data """
        pass

    def update(self, point, label):
        """ Updates internal state given a new labeled point """
        pass

    @property
    def volume(self):
        """ Estimative of version space's volume or size """
        raise NotImplementedError

