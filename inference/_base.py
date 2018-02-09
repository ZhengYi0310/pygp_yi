"""
Interface for latent function inference in Gaussian process models. These
models will assume that the hyperparameters are fixed and any optimization
and/or sampling of these parameters will be left to a higher-level wrapper.
"""

# future imports
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# global imports
import numpy as np
import scipy.linalg as sla

# local imports
from ..utils.model import Parameterized
import six
import abc

# exported symbols
__all__ = ['GP']

@six.add_metaclass(abc.ABCMeta)
class GP(Parameterized):
    """
    GP inference interface
    
    The class defines the GP interface, although it implements the Parametrized 
    interface, it also defines additional abstract methods and 
    is still abstract as a result
    
    The methods that must be implemented are:
        `_update`: update internal statistics given all the data.
        `_posterior`: compute the full posterior for use with sampling.
        `posterior`: compute the marginal posterior and its gradient.
        `loglikelihood`: compute the loglikelihood of observed data.
    Additionally, the following method can be implemented for improved
    performance in some circumstances:
        `_updateinc`: incremental update given new data.
    """
    def __init__(self, likelihood, kernel, mean):
        self.likelihood_ = likelihood
        self.kernel_ = kernel
        self.mean_ = mean
        self.X_ = None
        self.Y_ = None

        # the number of hyper parameters, the additional +1
        # is due to the mean hyperparameter
        self.nhyper_ = (self.likelihood_.nhyper_
                        + self.kernel_.nhyper_ + 1)

    def reset(self):
        """
        Remove all data from the model
        :return: 
        """
        self.X_ = None
        self.Y_ = None

    def __repr__(self):
        def indent(pre, text):
            return pre + ('\n' + ' ' * len(pre)).join(text.splitlines())

        return indent(self.__class__.__name__ + '(',
            ',\n'.join([
                indent('likelihood=', repr(self._likelihood)),
                indent('kernel=', repr(self._kernel)),
                indent('mean=', str(self._mean))]) + ')')

    def _params(self):
        params = []
        params += ["likelihood.{},".format(p[0]) + p[1:] for p in self.likelihood_._params()]
        params += ["kernel.{},".format(p[0]) + p[1:] for p in self.kernel_._params()]
        params += [('mean', 1, False)]
        return params

    @abc.abstractmethod
    def from_gp(cls, gp):
        """
        Create a new GP object given another. This allows one to make a "copy"
        of a GP using the same likelihood, kernel, etc. and using the same
        data, but possibly a different inference method.
        """
        raise NotImplementedError

    def get_hyper(self):
        return np.r_[self.likelihood_.get_hyper(),
                     self.kernel.get_hyper(),
                     self.mean_]

    def set_hyper(self, hyper):
        # FIXME: should set_hyper check the number of hyperparameters?
        a = self._likelihood.nhyper
        b = self._kernel.nhyper
        assert (len(hyper) == a + b + 1), "the number of hyper parameters don't match the model, " \
                                          "should be {} + {} + 1, yet {} provided".format(a ,b ,len(hyper))

        self._likelihood.set_hyper(hyper[:a])
        self._kernel.set_hyper(hyper[a:a + b])
        self._mean = hyper[-1]

        if self.ndata > 0:
            self._update()

    @property
    def ndata(self):
        """The number of current input/output data pairs."""
        return 0 if (self._X is None) else self._X.shape[0]

    def add_data(self, X, y):
        """
        Add new data to the GP model.
        """
        X = self._kernel.transform(X)
        y = self._likelihood.transform(y)

        if self._X is None:
            self._X = X.copy()
            self._y = y.copy()
            self._update()

        else:
            try:
                self._updateinc(X, y)
                self._X = np.r_[self._X, X]
                self._y = np.r_[self._y, y]

            except NotImplementedError:
                self._X = np.r_[self._X, X]
                self._y = np.r_[self._y, y]
                self._update()

    def sample(self, X, m=None, latent=True, rng=None):
        """
        Sample values from the posterior at points `X`. Given an `(n,d)`-array
        `X` this will return an `n`-vector corresponding to the resulting
        sample.
        If `m` is not `None` an `(m,n)`-array will be returned instead,
        corresponding to `m` such samples. If `latent` is `False` the sample
        will instead be returned corrupted by the observation noise. Finally
        `rng` can be used to seed the randomness.
        """
        X = self._kernel.transform(X)

        # this boolean indicates whether we'll flatten the sample to return a
        # vector, or if we'll return a set of samples as an array.
        flatten = (m is None)

        # get the relevant sizes.
        m = 1 if flatten else m
        n = len(X)

        # if a seed or instantiated RandomState is given use that, otherwise
        # use the global object.
        rng = rstate(rng)

        # add a tiny amount to the diagonal to make the cholesky of Sigma
        # stable and then add this correlated noise onto mu to get the sample.
        mu, Sigma = self._full_posterior(X)
        Sigma += 1e-10 * np.eye(n)
        f = mu[None] + np.dot(rng.normal(size=(m, n)), sla.cholesky(Sigma))

        if not latent:
            f = self._likelihood.sample(f.ravel(), rng).reshape(m, n)

        return f.ravel() if flatten else f

    def posterior(self, X, grad=False):
        """
        Return the marginal posterior. This should return the mean and variance
        of the given points, and if `grad == True` should return their
        derivatives with respect to the input location as well (i.e. a
        4-tuple).
        """
        return self._marg_posterior(self._kernel.transform(X), grad)

    def sample_fourier(self, N, rng=None):
        """
        Approximately sample a function from the GP using a fourier-basis
        expansion with N bases. See the documentation on `FourierSample` for
        details on the returned function object.
        """
        pass #for now

    @abc.abstractmethod
    def _update(self):
        """
        Update any internal parameters (ie sufficient statistics) given the
        entire set of current data.
        """
        raise NotImplementedError

    # NOTE: the following method is not abstract since we don't require that it
    # is implemented. if it is not implemented the full _update is performed
    # when new data is added.

    def _updateinc(self, X, y):
        """
        Update any internal parameters given additional data in the form of
        input/output pairs `X` and `y`. This method is called before data is
        appended to the internal data-store and no subsequent call to `_update`
        is performed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _full_posterior(self, X):
        """
        Compute the full posterior at points `X`. Return the mean vector and
        full covariance matrix for the given inputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _marg_posterior(self, X, grad=False):
        """
        Compute the marginal posterior at points `X`. Return the mean and
        variance vectors for the given inputs. If `grad` is True return the
        gradients with respect to the inputs as well.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loglikelihood(self, grad=False):
        """
        Return the marginal loglikelihood of the data. If `grad == True` also
        return the gradient with respect to the hyperparameters.
        """
        raise NotImplementedError

def rstate(rng=None):
    """
    Return a numpy RandomState object. If an integer value is given then a new
    RandomState will be returned with this seed. If None is given then the
    global numpy state will be returned. If an already instantiated state is
    given this will be passed back.
    """
    if rng is None:
        return np.random.mtrand._rand
    elif isinstance(rng, np.random.RandomState):
        return rng
    elif isinstance(rng, int):
        return np.random.RandomState(rng)
    raise ValueError('unknown seed given to rstate')



