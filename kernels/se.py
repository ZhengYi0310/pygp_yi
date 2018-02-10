"""
squared exponential covariance function with ARD measure 
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

#global imports
import numpy as np
#from mwhutils.random import rstate

# local imports
from ._real import RealKernel
from ._distance import rescale, diff, sqdist, sqdist_foreach
from ..utils.model import printable

# exported symbols
__all__ = ['SE']

@printable
class SE(RealKernel):
    def __init__(self, sf, els, ndim=None):
        self.logsf_ = np.log(sf)
        self.logels_ = np.log(els)
        self.iso_ = False
        self.ndim_ = np.size(self.logels_)
        self.nhyper_ = 1 + self.ndim_

        if ndim is not None:
            if np.size(self.logels_) == 1:
                self.iso_= True
                self.logels_ = float(self.logels_)
                self.ndim_ = ndim
            else:
                raise ValueError('arg ndim only applicable with scalar length scales!')


    def _params(self):
        return [('signal variances sf', 1, True),
                ('length scales', self.nhyper_ - 1, True)]

    def get_hyper(self):
        return np.r_[self.logsf_, self.logels_]

    def set_hyper(self, hyper):
        self.logsf_ = hyper[0]
        self.logels_ = hyper[1] if self.iso_ else hyper[1:]

    def get(self, X1, X2=None):
        """
        
        :param X1: M x D 
        :param X2: N x D
        :return: 
        """
        X1, X2 = rescale(np.exp(self.logels_), X1, X2)
        return np.exp(self.logsf_ * 2 - 0.5 * sqdist(X1, X2))

    def grad(self, X1, X2=None):
        X1, X2 = rescale(np.exp(self.logels_), X1, X2)
        D = 0.5 * sqdist(X1, X2)
        K = np.exp(self.logsf_ * 2 - D)
        yield 2 * K # Derivative w.r.t to signal variances
        if self.iso_:
            yield K * D # Derivative w.r.t to length scales (iso)
        else:
            for D in sqdist_foreach(X1, X2):
                yield  K * D # Derivative w.r.t to length scales (ard)

    def dget(self, X1):
        return np.exp(self.logsf_ * 2) * np.ones(len(X1))

    def dgrad(self, X):
        yield 2 * self.dget(X)
        for _ in xrange(self.nhyper - 1):
            yield np.zeros(len(X))

    def gradx(self, X1, X2=None):
        els = np.exp(self.logels_)
        X1, X2 = rescale(np.exp(self.logels_), X1, X2)

        D = diff(X1, X2)
        K = np.exp(self.logsf_ * 2 - np.sum(D**2, axis=-1) * 0.5)
        G = -K[:, :, None] * D / els
        return G

    def grady(self, X1, X2=None):
        return -self.gradx(X1, X2)

    def gradxy(self, X1, X2=None):
        els = np.exp(self.logels_)
        X1, X2 = rescale(np.exp(self.logels_), X1, X2)

        D = diff(X1, X2)
        _, _, d= D.shape
        K = np.exp(self.logsf_ * 2 - np.sum(D ** 2, axis=-1) * 0.5)

        D = D / els
        M = np.eye(D) / (els ** 2) - D[:, :, :, None] * D[:, :, None, :]
        G = K[:, :, None, None] * M
        return G

    def sample_spectrum(self, N, rng=None):
        '''
        rng = rstate(rng)
        sf2 = np.exp(self._logsf * 2)
        ell = np.exp(self._logell)
            W = rng.randn(N, self.ndim) / ell
            return W, sf2
        '''
        pass # for now