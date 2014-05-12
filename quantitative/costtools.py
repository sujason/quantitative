import numpy as np


def flatten_space(*args):
    # Take n 1-D arrays and broadcast them against each other producing an N-D space, then flatten
    expanded_args = [e.reshape([len(e)]+[1]*i) for i, e in enumerate(args)]
    expanded_args = np.broadcast_arrays(*expanded_args)
    flattened = [expanded_args[0].shape]
    flattened += [e.flatten() for e in expanded_args]
    return flattened


def sort_jacobians(Js):
    """
    A utility function to combine the axes of images with multi-dimensional
    output (e.g. complex data) with the overall protocol output.

    Input comes in with dimensions:
     (n_outputs_func, n_inputs_func, param_1, ..., param_m, n_out_protocol)
    where:
     n_outputs_func = the number of outputs of the function (rows in J)
     n_inputs_func = the number of inputs of the function (columns of J)
     param_1 ... param_m = vectors of test cases where J is evaluated
     n_out_protocol = the number of outputs for a given "protocol",
       i.e. the group of function evaluations that are measured
    Output combines the function outputs with protocol outputs:
     (param1, ..., param_m, n_out_protocol*n_outputs_func, n_inputs_func)
     n_out_protocol is the inner loop of the combined dimension.
    """
    # TODO make it so we don't need this function, too much hacking

    # (n_out_protocol, n_outputs_func, n_inputs_func, param_1, ..., param_m)
    out = np.rollaxis(Js, Js.ndim - 1)
    s = out.shape
    # (n_out_protocol*n_outputs_func, n_inputs_func, param_1, ..., param_m)
    # Grouped so that protocol ouputs are the inner loop/tighter grouping
    out = out.reshape((s[0]*s[1],)+s[2:], order='F')
    dim = range(out.ndim)
    # (param1, ..., param_m, n_out_protocol*n_outputs_func, n_inputs_func)
    return out.transpose(dim[2:] + dim[:2])


def test_sort_jacobians():
    # TODO
    pass


def area_underneath(y, x=None):
    # Compute the integral of a linearly interpolated y.
    if x is None or np.atleast_1d(x).size == 1:
        diff_x = 1.0
    else:
        diff_x = np.diff(x)
    integrand = np.convolve(np.atleast_1d(y), np.array([0.5, 0.5]), 'valid')*diff_x
    return integrand.sum()


def remove_small_nex(nex, data, threshold=1e-6):
    # Filters out entries of nex and data if values in nex are below a threshold.
    nex, _ = np.broadcast_arrays(nex, data[0])
    not_small = nex > threshold
    def try_filter(x):
        try:
            return x[not_small]
        except ValueError:
            # when x doesn't have enough entries
            return x
        except TypeError:
            print x
            raise
    nex = nex[not_small]
    data_out = [try_filter(el) for el in data]
    return nex, data_out