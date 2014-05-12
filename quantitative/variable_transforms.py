import numpy as np


def normalize_transform(z, match_array=None, total=1.):
    """
    Takes an array and turns them into positive weights.  This transform is scale invariant.
    """
    if match_array is not None:
        z, _ = np.broadcast_arrays(z, match_array)
    w = z**2.
    return total*w/w.sum()


def normalize_alternate_transform(z, total=1., add_element=False):
    """
    Takes an array and turns them into positive weights.  This transform is sensitive to scale.
    Due to this, the last element is determined by the rest of the array as the sum must be 1.
    add_element=False -- ignores the last given value and overwrites it with the remainder
    add_element=True -- adds an element to the end of the array for the remainder
    """
    s = np.sin(z)**2.
    if add_element:
        s = np.append(s, 1.)
    else:
        s[-1] = 1.
    c = np.cumprod(np.cos(z)**2.)
    if add_element:
        c = np.append(1., c)
    else:
        c = np.roll(c, 1)
        c[0] = 1.
    return total*s*c


def squared_transform(z, offset=0.):
    """
    Takes an array and makes it non-negative (or >= offset) via squaring.
    """
    return z**2. + offset


def ascending_nonnegative_transform(z, offset=0., nonnegative_transform=squared_transform):
    """
    Takes an array and makes it ascend in value by nonnegative_transform and incrementing by the previous element.
    """
    x = nonnegative_transform(z)
    y = np.roll(x, 1)
    y[0] = offset
    return x + y.cumsum()


def sine_bounded_transform(z, lower=-1., upper=1.):
    """
    Takes an array and makes it bound by lower and upper limits via sine:
    z = -1 corresponds to the lower and z = 1 the upper.
    """
    center = (upper+lower) / 2.
    full_width = upper - lower
    return full_width/2.*np.sin(np.pi/2.*z) + center


def sigmoid_bounded_transform(z, lower=-1., upper=1.):
    """
    Takes an array and makes it bound by lower and upper limits via sigmoid:
    z = -1 corresponds to the lower and z = 1 the upper.
    """
    center = (upper+lower) / 2.
    full_width = upper - lower
    return full_width*(1./(1.+np.exp(-z))-0.5) + center


def reciprocal_quadratic_bounded_transform(z, lower=-1., upper=1.):
    center = (upper+lower) / 2.
    full_width = upper - lower
    return full_width*(1./(z**2+1)-0.5) + center


def ascending_bounded_box_transform(z, lower=-1., upper=1., bounded_transform=sine_bounded_transform):
    """
    Takes an array and makes each element succesively lower bounded by the previous value via sine (by default).
    lower and upper correspond to the global lower and upper limit on the vector, i.e. lower affects the
    first element and upper bounds all of them.
    """
    x = np.zeros_like(z, dtype=np.float)
    for i, val in enumerate(z):
        if i == 0:
            prev = lower
        else:
            prev = x[i-1]
        x[i] = bounded_transform(val, lower=prev, upper=upper)
    return x


# TODO write unit tests
