import numpy as np
import scipy.optimize


class Minimizer:
    """
       This class' objective is to help the version space classes to find an feasible point inside it.
       More specifically, it uses the Scipy's SLSQP solver to find a solution to the problem:


            minimize s, subject to Ax <= s

       As we can see, we are looking for a point x satisfying Ax <= 0. We hope that after optimization, a solution (s,x)
       with s <= 0 will be found.
    """

    def __init__(self, dim):
        if int(dim) <= 0:
            raise ValueError("Dimension must be positive.")

        self.__dim = int(dim)
        self.__constrains_list = self.get_initial_constrains()

    def __call__(self, x0):
        """
        Call solver starting at the point x0.
        :param x0: initial point of iterative method.
        :return: point inside the version space
        """
        result = scipy.optimize.minimize(
            x0=x0,
            fun=lambda x: x[0],
            jac=lambda x: np.array([1.0] + [0.0] * self.dim),
            constraints=self.__constrains_list,
            method="SLSQP"
        )

        if result.x[0] >= 0:
            raise RuntimeError(
                "Optimization failed! Success = {0}, Message = {1}, Result = {2}".format(
                    result.success,
                    result.message,
                    result.x)
                )
        return result.x[1:]

    @property
    def dim(self):
        return self.__dim

    def get_initial_constrains(self):
        return []

    def clear(self):
        self.__constrains_list = self.get_initial_constrains()

    def append(self, vector):
        """
           Include a new inequality on the form < vector, x > <= s
        """
        if vector.shape != (self.__dim, ):
            raise ValueError("Vector has wrong dimension!")

        self.__constrains_list.append(
            {
                'type': 'ineq',
                'fun': lambda s: s[0] - np.dot(vector, s[1:]),
                'jac': lambda s: np.hstack([1.0, -vector])
            }
        )
