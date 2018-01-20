import numpy as np
from .appendable_constrain import AppendableInequalityConstrain
from .base import VersionSpace
from ..convexbody.objects import Polytope
from ..convexbody.sampling import HitAndRunSampler


class LinearVersionSpace(Polytope, VersionSpace):
    def __init__(self, dim):
        Polytope.__init__(self, l=[-1]*(dim+1), h=[1]*(dim+1))
        VersionSpace.__init__(self)
        self.inequality_constrain = AppendableInequalityConstrain(dim+1)  # add one to account for bias

    def clear(self):
        self.inequality_constrain.clear()

    def _update_single(self, point, label):
        point = np.hstack([1, point])  # add bias component
        constrain_vector = -label * point  # constrain = -y_i (1, x_i)
        self.inequality_constrain.append(constrain_vector, 0)

    def sample(self, chain_length, sample_size=-1, initial_point=None):
        if initial_point is None:
            print('falling back to linprog')
            initial_point = self.get_point()

        if sample_size > 0:
            return HitAndRunSampler.uniform(self, initial_point, chain_length, sample_size)
        return HitAndRunSampler.sample_chain(self, initial_point, chain_length)

    def get_point(self):
        return self.inequality_constrain.get_point()

