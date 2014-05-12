import inspect
import decorator
import numpy as np
import autodiff as ad
from collections import defaultdict, OrderedDict
from scipy.linalg import inv, LinAlgError


class HigherAD(object):
    def __init__(self, fn):
        self.func_py = fn
        self.func_ad = ad.Function(fn)
        self.grads = dict()

    @staticmethod
    def _dict_to_array(d):
        # Converts a dictionary with multi index keys and array values into a combined numpy array.
        # Output is (value_array_dims + multi_index_dims)
        dims = np.array(d.keys()).max(axis=0) + 1
        out = None
        for k, v in d.iteritems():
            if out is None:
                out = np.zeros(np.r_[v.shape, dims])
            out[[Ellipsis] + list(k)] = v
        return out
        
    def _index_wrapper(self, index):
        #TODO use operations.itemgetter instead
        def wrapper(f, *args, **kwargs):
            return f(*args, **kwargs)[index]
        return decorator.decorator(wrapper)(self.func_py)
        
    def _init_gradients(self, wrt, *args):
        if not self.grads.has_key(wrt):
            it = np.nditer(args)
            # Gather the output form of the function, list/dict/scalar
            sample_out = self.function(*it[:])
            #TODO generalize to Nd output
            if isinstance(sample_out, list):
                # Vector/list output.
                self.grads[wrt] = [ad.Gradient(self._index_wrapper(i), wrt=wrt) for i, _ in enumerate(sample_out)]
            elif isinstance(sample_out, dict):
                # Dict output.
                self.grads[wrt] = OrderedDict([(key, ad.Gradient(self._index_wrapper(key), wrt=wrt)) for key in sample_out])
            else:
                # Scalar output.
                self.grads[wrt] = [ad.Gradient(self.func_py, wrt=wrt)]
        return self.grads[wrt]

    def function(self, *args):
        # func_ad can't handle dict output
        return self.func_py(*args)
        
    def jacobian(self, *args, **kwargs):
        # TODO use decorator to fix argspec for wrt_in/wrt_out
        # Want default keywords after *args, need to use **kwargs handling instead
        wrt_in = kwargs.pop('wrt_in', None)
        wrt_out = kwargs.pop('wrt_out', None)
        wrt_index_missing = None
        if wrt_in is not None:
            wrt_index_missing = tuple(i for i, el in enumerate(wrt_in) if el not in inspect.getargspec(self.func_py).args)
            wrt_index, wrt_in = zip(*tuple((i, el) for i, el in enumerate(wrt_in) if el in inspect.getargspec(self.func_py).args))
        
        output_dict = {}
        grads = self._init_gradients(wrt_in, *args)
        if wrt_out is None:
            try:
                grads = grads.values()
            except AttributeError:
                pass
        else:
            grads = [grads[out] for out in wrt_out]
        it = np.nditer(args, flags=['multi_index'])
        input_dims = np.broadcast(*[np.atleast_1d(el) for el in args]).shape
        
        J = np.array([grad(*it[:]) for grad in grads])
        output = np.zeros(J.shape + input_dims)
        output[[Ellipsis] + list(it.multi_index)] = J
        it.iternext()
        while not it.finished:
            output[[Ellipsis] + list(it.multi_index)] = np.array([grad(*it[:]) for grad in grads])
            it.iternext()
        if wrt_index_missing:
            missing_shape = list(output.shape)
            # Change the size of the input dimension
            missing_shape[1] = len(wrt_index_missing)
            missing_output = np.empty(missing_shape)
            # Columns of missing input parameters are filled with NaN
            missing_output.fill(np.nan)
            # Since we concatenate, columns associated with missing inputs are at the end
            sort_index = wrt_index + wrt_index_missing
            sort_index = sorted(zip(range(len(sort_index)), sort_index), key=lambda e: e[1])
            sort_index = [e[0] for e in sort_index]
            output = np.concatenate((output, missing_output), axis=1)[:, sort_index, ...]
        if output.ndim == 3 and output.shape[-1] == 1:
            # Scalar input.
            return output[:, :, 0]
        return output

    def jacobian_old(self, *args, **kwargs):
        wrt_in = kwargs.pop('wrt_in', None)
        wrt_out = kwargs.pop('wrt_out', None)
        
        output_dict = {}
        grads = self._init_gradients(wrt_in, *args)
        if wrt_out is None:
            try:
                grads = grads.values()
            except AttributeError:
                pass
        else:
            grads = [grads[out] for out in wrt_out]
        it = np.nditer(args, flags=['multi_index'])
        while not it.finished:
            J = np.array([grad(*it[:]) for grad in grads])
            output_dict[it.multi_index] = J
            it.iternext()
        if len(output_dict) is 1:
            return output_dict.values()[0]
        return self._dict_to_array(output_dict)
    
    def __call__(self, *args):
        return [self.function(*args), self.jacobian(*args)]


def calc_crlb(J, Sigma=1.0):
    # CRLB for an unbiased estimator.  Sigma is noise covariance matrix.
    Sigma, _ = np.broadcast_arrays(Sigma, np.zeros(J.shape[0]))
    if Sigma.ndim == 1:
        Sigma = np.diag(Sigma)
    # TODO use solve instead
    F = (J.T).dot(inv(Sigma)).dot(J)
    try:
        return np.sqrt(np.diag(inv(F)))
    except LinAlgError:
        print 'WARNING: F is a singular matrix.'
        return np.zeros(J.shape[1])*np.nan


if __name__ == "__main__":
    def dummy_dict(a, b, c, d):
        """asdf"""
        return {'1': 2*a + 0*(b+c+d), '2': 0*a + 3*(b+c+d)}

    def dummy_list(a, b, c, d):
        return [2*a + 0*(b+c+d), 0*a + 3*(b+c+d)]

    Dummy = HigherAD(dummy_dict)
    print 'fcn', Dummy.function(*arange(4, dtype=np.float))
    J = Dummy.jacobian(*arange(4, dtype=np.float), wrt_out=['1'], wrt_in=('a', 'undef', 'b'))
    J2 = Dummy.jacobian_old(*arange(4, dtype=np.float), wrt_out=['1'], wrt_in=('a', 'b'))
    print J, J.shape
    print J2, J2.shape