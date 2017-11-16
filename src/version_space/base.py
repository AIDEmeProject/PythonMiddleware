from ..convexbody.sampling import HitAndRunSampler
from .minimizer import Minimizer
from .appendable_constrain import AppendableInequalityConstrain


class VersionSpaceMixin:
    def __init__(self):
        pass

    def clear(self):
        pass

    def update(self, point, label):
        pass

    def score(self):
        return {}

class VersionSpace(VersionSpaceMixin):
    """
        Version space of an Active Learning algorithm.

        SVM case: for linear kernel, an unit ball in higher-dimensional space
        Actboost: a polytope
    """
    def __init__(self):

        self.cut_estimator = VersionSpaceCutEstimator(1000)

    def clear(self):
        self.inequality_constrain.clear()
        self.cut_estimator.clear()

    def update(self, point, label):
        """ Updates internal state given a new labeled point """
        self.cut_estimator.update_sample(self)

        constrain_vector = -label * point
        self.inequality_constrain.append(constrain_vector, 0.0)

    def score(self):
        return self.cut_estimator.estimate_cut(self)


class VersionSpaceCutEstimator:
    def __init__(self, chain_length):
        self.__sampler = HitAndRunSampler(chain_length)
        self.__sample_cache = None

    def clear(self):
        self.__sample_cache = None

    def update_sample(self, version_space):
        if hasattr(version_space, 'get_point'):
            self.__sample_cache = self.__sampler.sample_chain(version_space, version_space.get_point())

    def estimate_cut(self, version_space):
        if not hasattr(version_space, 'is_inside'):
            return {}

        old_sample = self.__sample_cache
        number_of_samples_inside = version_space.is_inside(old_sample).sum()
        return {
            'version_space_ratio_estimate': 100 * (1. - number_of_samples_inside / len(old_sample))
        }
