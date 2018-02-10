"""
The Deterministic Training Conditional (DTC) approximation
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.linalg as sla
import itertools as it

from ..likelihood import Gaussian
from ._base import GP

class DTC(GP):
    """Deterministic training conditional approximation to GP inference."""

    def __init__(self, likelihood, kernel, mean, U):
        # NOTE: exact inference will only work with Gaussian likelihoods.
        if not isinstance(likelihood, Gaussian):
            raise ValueError('exact inference requires a Gaussian likelihood')

        super(DTC, self).__init__(likelihood, kernel, mean)
        # save the pseudo-input locations.

        self.U_ = np.array(U, dtype=float, ndmin=2)
        self.luu_ = None
        self.lux_ = None
        self.alpha_ = None

    @property
    def pseudoinputs(self):
        """
        The pseudo inputs
        :return: 
        """
        return self.U_

    @classmethod
    def from_gp(cls, gp, U=None):
        if U is None:
            if hasattr(gp, 'pseudoinputs'):
                U = gp.pseudoinputs.copy()
            else:
                raise ValueError('gp has no pseudoinputs and none are given')
        newgp = cls(gp.likelihood_.copy(), gp.kernel_.copy(), gp.mean_, U)
        if gp.ndata > 0:
            X, Y = gp.data
            newgp.add_data(X, Y)
        return newgp

    def _update(self):
        """
        Update any internal parameters (ie sufficient statistics) given the
        entire set of current data.
        """
        p_dim = self.U_.shape[0]
        su2 = self.likelihood_.s2 * 1e-6

        # choleskies of Kuu and (Kuu + Kfu * Kuf / sn2), respectively,
        Kuu = self.kernel_.get(self.U_)
        self.luu_ = sla.cholesky(Kuu + su2 * np.eye(p_dim), lower=True)

        Kux = self.kernel_.get(self.X_ + self.U_)
        Sigma = Kuu + np.dot(Kux , Kux.T) / self.likelihood_.s2
        r = self.Y_ - self.mean_
        self.lux_ = sla.cholesky(Sigma + su2 * np.eye(p_dim), lower=True)
        self.alpha_ = sla.solve_triangular(self.lux_, np.dot(Kux, r), lower=True)

    def _full_posterior(self, X):
        # grab the prior mean and variance.
        mu = np.full(X.shape[0], self._mean)
        sigma = self.kernel_.get(X)

        if self.X_ is not None:
            Kux = self.kernel_.get(self.U_, X)
            a = sla.solve_triangular(self.lux_, Kux, lower=True)
            b = sla.solve_triangular(self.luu_, Kux, lower=True)
            mu += np.dot(a.T, self.alpha_) / self.likelihood_.s2
            c = np.dot(a.T, a) - np.dot(b.T, b)
            sigma += c

        return mu, sigma

    def _marg_posterior(self, X, grad=False):
        # grab the prior mean and variance.
        mu = np.full(X.shape[0], self._mean)
        s2 = self.kernel_.dget(X)

        if self.X_ is not None:
            Kux = self.kernel_.get(self.U_, X)
            a = sla.solve_triangular(self.lux_, Kux, lower=True)
            b = sla.solve_triangular(self.luu_, Kux, lower=True)
            mu += np.dot(a.T, self.alpha_) / self.likelihood_.s2
            c = np.sum(a * a, axis=0) - np.sum(b * b, axis=0)
            s2 += c

        if not grad:
            return (mu, s2)

    def loglikelihood(self, grad=False):


