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

        Kux = self.kernel_.get(self.U_, self.X_)
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

        # Get the prior gradients. Note that this assumes a constant mean and
        # stationary kernel.
        dmu = np.zeros_like(X)
        ds2 = np.zeros_like(X)

        if self.X_ is not None:
            dK = self.kernel_.grady(self.U_, X)  # M x N x D
            dK = dK.reshape(self.U_.shape[0], -1)  # M x ND
            Rdk1 = sla.solve_triangular(self.lux_, dK, lower=True)
            Rdk2 = sla.solve_triangular(self.luu_, dK, lower=True)
            dmu += np.dot(Rdk1.T, self.alpha_).reshape(X.shape)

            Rdk1 = np.rollaxis(np.reshape(Rdk1, (-1,) + X.shape), 2)  # D x M x N
            Rdk2 = np.rollaxis(np.reshape(Rdk2, (-1,) + X.shape), 2)  # D x M x N
            ds2 -= 2 * np.sum(Rdk1 * a, axis=1).T  # N x D
            ds2 += 2 * np.sum(Rdk2 * b ,axis=1).T

        return (mu, s2, dmu, ds2)
    def loglikelihood(self, grad=False):
        sn2 = self.likelihood_.s2
        su2 = sn2 * 1e-6
        sn = np.sqrt(sn2)

        Kux = self.kernel_.get(self.U_, self.X_)
        r = self.Y_.copy() - self.mean_
        r /= sn

        V = sla.solve_triangular(self.lux_, Kux, lower=True)
        V /= sn
        P = self.ndata
        A = np.eye(P) - np.dot(V.T, V)
        lZ = -0.5 * np.dot(np.dot(r.T, A), r)

        lZ -= np.sum(np.log(self.lux_.diag()))
        lZ += np.sum(np.log(self.luu_.diag()))
        lZ -= self.ndata * np.log(sn)
        lZ -= 0.5 * self.ndata * np.log(2 * np.pi)

        if not grad:
            return lZ

        # allocate space for the gradients.
        dlZ = np.zeros(self.nhyper)
        # iterator over gradients of the kernels
        dK = it.izip(self.kernel_.grad(self.U_),
                     self.kernel_.grad(self.U_, self.X_))

        # E. Snelson's Phd thesis
        B = sla.cho_solve((self.lux_, True), np.dot(Kux, r))
        Sigma_inv = sla.cho_solve((self.lux_, True), np.eye(P))
        Kuu_inv = sla.cho_solve((self.luu_, True), np.eye(P))

        # gradient w.r.t the noise variance
        dl1dsn2 = (-self.ndata + np.trace(np.dot(V.T, V))) * 0.5
        dl2dsn2 = 0.5 *  np.dot(np.dot(r.T, A.T), np.dot(A, r))
        dlZ[0] = dl1dsn2 + dl2dsn2

        i = 1
        for i, (dKuu, dKux) in enumerate(dK, i):
            temp0 = np.dot(dKux, Kux.T)
            Sigma_dot = dKuu + (temp0 + temp0.T) / sn2
            dlZ1 = -0.5 * np.sum(Sigma_inv * Sigma_dot) + 0.5 * np.sum(Kuu_inv * dKuu)

            temp1 = sla.solve_triangular(self.lux_, Sigma_dot, lower=True)
            dlZ2 = np.dot(self.lux_.T, B)
            temp2 = np.dot(temp1, B) * 0.5 - dlZ2
            dlZ2 = -1 * np.dot(dlZ2.T, temp2)

            dlZ[i] = dlZ1 + dlZ2

        # gradient w.r.t constant mean
        dlZ[-1] = np.sum(np.dot(A, r))

        return lZ, dlZ





