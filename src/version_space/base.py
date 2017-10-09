class VersionSpace(object):
    """
        Version space of an Active Learning algorithm.

        SVM case: for linear kernel, an unit ball in higher-dimensional space
        Actboost: a polytope
    """
    def __init__(self, n_samples=1000):
        self.__cache = {}
        self.__scorer = VersionSpaceMetricTracker(n_samples)

    def clear(self):
       self.__scorer.clear()

    def update(self, point, label):
        """ Updates internal state given a new labeled point """
        pass

    def score(self):
        return self.__scorer.score(self)


class VersionSpaceMetricTracker:
    def __init__(self, n_samples):
        self.n_samples = int(n_samples)
        self.__cache = {}

    def clear(self):
        self.__cache = {}

    def score(self, version_space):
        scores = {}

        if hasattr(version_space, 'volume'):
            volume = version_space.volume
            scores['version_space_volume'] = volume

            if 'previous_volume' in self.__cache:
                previous_volume = self.__cache['previous_volume']
                scores["version_space_ratio"] = 100 * (1 - volume / previous_volume)

            self.__cache['previous_volume'] = volume

        if hasattr(version_space, 'sample') and hasattr(version_space, 'is_inside'):
            if 'samples' in self.__cache:
                samples = self.__cache['samples']
                number_points_inside = sum(version_space.is_inside(samples))
                scores['version_space_ratio_estimate'] = 100 * (1 - number_points_inside / len(samples))

            self.__cache['samples'] = version_space.sample(1000)

        return scores

