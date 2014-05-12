import numpy as np
from collections import MutableMapping


class MultiOptimizationHelper(object):
    """
    Object to wrap a cost function.  Aids in handling splitting a 1-D input vector into multiple variables and their constraints.
    Handles multiple sequences with different parameters.  Expects dictionaries for all inputs.

    Inputs:
    transforms -- a dictionary of functions that take an array and return an array
    """
    def __init__(self, cost_function, n_images, params, start_range=None, transforms=None):
        self.cost_function = cost_function
        self.n_images = n_images
        self.params = params
        assert len(n_images) == len(params)
        self.n_sequences = len(n_images)
        self._start_range = {seq: dict() for seq in n_images}
        if start_range is not None:
            self.start_range = {seq: dict(zip(params_seq, start_range[seq])) for seq, params_seq in params.items()}
        if transforms is not None:
            self.transforms = {seq: dict(zip(params_seq, transforms[seq])) for seq, params_seq in params.items()}
        self.constraints = []

        self.parameter_indices = dict()
        offset = 0
        for seq, params_seq in params.iteritems():
            stride = len(params_seq)
            end = stride*n_images[seq] + offset
            self.parameter_indices[seq] = {param: range(i+offset, end, stride) for i, param in enumerate(params_seq)}
            offset += end

    def __call__(self, x, **kwargs):
        kwargs.update(self._parameter_values(x))
        return self.cost_function(**kwargs)

    def _transform_parameters(self, values):
        """
        Take a dictionary of parameter value arrays and apply corresponding functions from self.transforms on them.
        """
        # TODO do this more efficiently?
        for seq, values_seq in values.iteritems():
            try:
                transforms_seq = self.transforms[seq]
            except KeyError:
                # Missing a transform, skip.
                continue
            except AttributeError:
                # No transforms given at all.
                break
            for param, value in values_seq.iteritems():
                try:
                    values_seq[param] = transforms_seq[param](value)
                except KeyError:
                    # Missing a transform, skip.
                    continue
                except TypeError:
                    if transforms_seq[param] is None:
                        continue
        return values      

    def _parameter_values(self, x):
        """
        Takes a 1-D vector and re-organizes it into a dictionary organized like params.
        """
        params = self.params
        offset = 0
        values = dict()
        for seq, params_seq in params.iteritems():
            stride = len(params_seq)
            end = stride*self.n_images[seq] + offset
            vals = x[offset:end].reshape((-1, stride))
            values[seq] = {param: vals[:, i] for i, param in enumerate(params_seq)}
            offset += end
        return self._transform_parameters(values)

    def add_affine_constraint(self, param, constraint_type, a=1.0, b=0.0):
        """
        Adds an affine constraint function on a parameter as specified by a tuple of keys.
        """
        indices = self.parameter_indices
        try:
            for p in param:
                indices = indices[p]
        except KeyError:
            print 'Warning: Parameter %s does not exist in %s, skipping affine constraint -- %s, %s, %s' % (param, self, constraint_type, a, b)
            return
        for i in indices:
            constraint = {
                'type': constraint_type,
                'fun': lambda x, i=i, a=a, b=b: a*x[i] + b,
            }
            self.constraints.append(constraint)

    @property
    def start_range(self):
        """
        Returns the 1-D list of tuples with (min, max) bounds on the starting range of each free parameter.
        Note that this should be the start ranges for the parameter values before transformation.
        """
        def flatten(d, tag=None):
            # Flatten a nest of dictionaries into (tuple of keys, value) pairs
            l = list()
            if tag is None:
                tag = tuple()
            for k, v in d.iteritems():
                if isinstance(v, MutableMapping):
                    l.extend(flatten(v, tag+(k, )))
                else:
                    l.append((tag+(k, ), v))
            return l
        range_list = list()
        # Lookup 1-D indices and starting range
        for keys, indices in flatten(self.parameter_indices):
            val = self._start_range
            for key in keys:
                val = val[key]
            for index in indices:
                range_list.append((index, val))
        # Sort based on index.
        range_list.sort(key=lambda e: e[0])
        # Return the column of ranges.
        return zip(*range_list)[1]

    @start_range.setter
    def start_range(self, d):
        """
        Update the start ranges with a dictionary
        """
        self._start_range.update(d)


class OptimizationHelper(object):
    """
    Object to wrap a cost function.  Aids in handling splitting a 1-D input vector into multiple variables and their constraints.
    """
    def __init__(self, cost_function, n_images, params, start_range=None, single_params=[]):
        """
        start_range specifies the ranges for params+single_params
        """
        self.cost_function = cost_function
        self.n_images = n_images
        self.params = params
        self._start_range = dict()
        self.single_params = single_params
        if start_range is not None:
            self.start_range = dict(zip(params+single_params, start_range))
        self.constraints = []
        
        stride = len(params)
        N = stride*n_images
        self.parameter_indices = {param: range(i, N, stride) for i, param in enumerate(params)}
        for param, index in zip(single_params, range(N, N+len(single_params))):
            self.parameter_indices[param] = [index]

    def __call__(self, x, **kwargs):
        kwargs.update(self._parameter_values(x))
        return self.cost_function(**kwargs)

    def _parameter_values(self, x):
        """
        Takes a 1-D vector and re-organizes it into a dictionary organized like params.
        """
        params = self.params
        stride = len(params)
        N = stride*self.n_images
        vals = x[:N].reshape((-1, stride))
        d = {param: vals[:, i] for i, param in enumerate(params)}
        d.update(dict(zip(self.single_params, x[N:])))
        return d

    def add_affine_constraint(self, param, constraint_type, a=1.0, b=0.0):
        """
        Adds an affine constraint function on a parameter as specified by a tuple of keys.
        """
        indices = self.parameter_indices[param]
        for i in indices:
            constraint = {
                'type': constraint_type,
                'fun': lambda x, i=i, a=a, b=b: a*x[i] + b,
            }
            self.constraints.append(constraint)

    @property
    def start_range(self):
        """
        Returns the 1-D list of tuples with (min, max) bounds on the starting range of each free parameter.
        """
        # Lookup 1-D indices and starting range
        range_list = [
            (index, self._start_range[param])
            for param, indices in self.parameter_indices.items()
            for index in indices
        ]
        # Sort based on index.
        range_list.sort(key=lambda e: e[0])
        # Return the column of ranges.
        return zip(*range_list)[1]

    @start_range.setter
    def start_range(self, d):
        """
        Update the starting ranges with a dictionary
        """
        self._start_range.update(d)


def test_OptimizationHelper_init(C):
    pass

def test_OptimizationHelper_affine_constraint(C):
    # >0 contraints
    C.add_affine_constraint('fa', 'ineq')
    C.add_affine_constraint('nex', 'ineq')
    x = np.arange(C.n_images * len(C.params)) + 0.5
    indices = np.concatenate((C.parameter_indices['fa'], C.parameter_indices['nex']))
    for i, constraint in zip(indices, C.constraints):
        assert x[i] == constraint['fun'](x)
        
def test_OptimizationHelper_cost(C):
    fa = np.random.random(3)
    nex = np.random.random(3)
    x = np.array(zip(fa,nex)).reshape(-1)
    C_fa, C_nex, C_tr = C(x)
    assert np.all(C_fa == fa)
    assert np.all(C_nex == nex)
    assert np.all(C_tr == min_tr)
        
def test_OptimizationHelper_start_range(C):
    C.start_range = {'fa': (0, np.pi/2), 'nex': (0.0, 10.0)}
    assert C.start_range == ((0, np.pi/2), (0.0, 10.0))*n_images


def test_MultiOptimizationHelper_transform(C):
    pass


if __name__ == '__main__':
    min_tr = 5.0
    def cost_function(fa=np.pi, nex=1.0, tr=min_tr):
        return fa, nex, min_tr
    params = ['fa', 'nex']
    n_images = 3
    C = OptimizationHelper(cost_function, n_images, params)
    test_OptimizationHelper_init(C)
    test_OptimizationHelper_start_range(C)
    test_OptimizationHelper_affine_constraint(C)
    test_OptimizationHelper_cost(C)
    # TODO insert multiopt tests from ipynb (lost now)
