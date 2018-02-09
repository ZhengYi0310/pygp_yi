from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.linalg as sla
from ._base import GP
from ..likelihood import gaussian

__all__ = ['ExactGP']


class ExactGP(GP):
    """
    Exact GP inference.
    This class implements exact inference for GPs. Note that exact inference
    only works with regression so an exception will be thrown if the given
    likelihood is not Gaussian.
    """
    def __init__(self, likelihood, kernel, mean):
        # NOTE: exact inference will only work with Gaussian likelihoods.
        if not isinstance(likelihood, gaussian):
            raise ValueError('exact inference requires a Gaussian likelihood')

        super(ExactGP, self).__init__(likelihood, kernel, mean)
        self._R = None
        self._a = None

    @classmethod
    def from_gp(cls, gp):
        newgp = cls(gp.likelihood_.copy(), gp.kernel_.copy(), gp.mean_)
        if gp.ndata > 0:
            X, y = gp.data
            newgp.add_data(X, y)
        return newgp

    def reset(self):
        for attr in 'Ra':
            setattr(self, attr + '_', None)
        super(ExactGP, self).reset()

    def _update(self):
        sn2 = self.likelihood_.s2
        K = self.kernel_.get(self.X_) + sn2 * np.eye(len(self.X_))
        r = self.Y_ - self.mean_
        self.L_ = sla.cholesky(K, lower=True)
        self.alpha_ = sla.solve_triangular(self.L_, r, lower=True)

    def _chol_update(self, A, B, C, a, b):
        """
        Update the cholesky decomposition of a growing matrix.
        Let `A` denote a cholesky decomposition of some matrix, and `a` is the inverse
        of `A` applied to some vector `y`. This computes the cholesky to a new
        matrix which has additional elements `B` on the non-diagonal, and `C` on
        the diagonal block. It also computes the solution to the application of the
        inverse where the vector has additional elements `b`.
        """
        n = A.shape[0]
        m = C.shape[0]

        B = sla.solve_triangular(A, B, trans=True)
        C = sla.cholesky(C - np.dot(B.T, B))
        c = np.dot(B.T, a)

        # grow the new cholesky and use then use this to grow the vector a.
        A = np.r_[np.c_[A, B], np.c_[np.zeros((m, n)), C]]
        a = np.r_[a, sla.solve_triangular(C, b - c, trans=True)]

        return A, a
    def _updateinc(self, X, y):
        sn2 = self._likelihood.s2
        Kss = self._kernel.get(X) + sn2 * np.eye(len(X))
        Kxs = self._kernel.get(self._X, X)
        r = y - self._mean

    def _full_posterior(self, X):
        mu = np.full(X.shape[0], self.mean_)
        sigma = self.kernel_.get(X)

        if self.X_ is not None:
            k = self.kernel_.get(self.X_, X)
            v = sla.solve_triangular(self.L_, k, lower=True)

            mu += np.dot(v.T, self.alpha_)
            sigma -= np.dot(v.T, v)
        return mu, sigma

    def _marg_posterior(self, X, grad=False):
        mu = np.full(X.shape[0], self._mean)
        s2 = self.kernel_.dget(X)
        k = self.kernel_.get(self.X_, X)
        v = sla.solve_triangular(self.L_, k, lower=True)

        mu += np.dot(v.T, self.alpha_)
        s2 -= np.sum(v**2, axis=0)

        if not grad:
            return (mu, s2)

    def loglikelihood(self, grad=False):
        lZ = -0.5 * np.dot(self.alpha_.T, self.alpha_)
        lZ -= np.log(np.sum(self.L_.diagonal()))
        lZ -= 0.5 * self.ndata * np.log(2 * np.pi)

        if not grad:
            return lZ

        alpha = sla.solve_triangular(self.L_.T, self.alpha_, lower=True)
        Q = -np.dot(alpha, alpha.T)
        Q += sla.cho_solve((self.L_, True), np.eye(self.ndata))

        #eq5.9, A.14, A.15 in Gaussian Process for Machine Learning
        dlZ = np.r_[-self._likelihood.s2 * np.trace(Q), #derivative wrt the likelihood's noise term.
                    [-0.5 * np.trace(Q * dk) for dk in self.kernel_.grad(self.X_)], #derivative wrt the kernel hyper,
                    np.sum(alpha)]

        return lZ, dlZ
