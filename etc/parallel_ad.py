# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import multiprocessing
import autodiff as ad
from functools import partial
from numpy import arange, exp, array

def fn(x):
    return x ** 3.

f = ad.Function(fn) # compile the function
g = ad.Gradient(fn) # compile the gradient of the function

def print_g(x):
    print g(x)

def wrap_g(x):
    return g(x)

def cost_g(value=None, a=0.):
    x = value['x']
    print x-a
    return g(x[0]-a)

# <codecell>

print_g(100.)

M = multiprocessing.Pool(8)
x = arange(10.)
"""
for e in x:
    print e
    p = multiprocessing.Process(target=print_g, args=(e, ))
    p.start()
print 'done for'
p.join()
"""
y = M.map(wrap_g, x)
print 'a', y

from opt_helper import MultiOptimizationHelper
from multistart import MultiStart

n_images = {
    'value': 1,
}

params = {
    'value': ['x'],
}

start_range = {
    'value': [(-10., 10.)],
}

C = MultiOptimizationHelper(cost_g, n_images, params, start_range)

M = MultiStart(
    1000,
    C.start_range,
    method='SLSQP',
)

fun = partial(C, a=1.)
print M.solve(
    parallel_pool=4,
    fun=fun,
)
