import numpy as np
import theano.tensor
from autodiff import Symbolic, tag

# -- a vanilla function
def f1(x):
    return x + 2

# -- a function referencing a global variable
y = np.random.random(10)
def f2(x):
    return x * y

# -- a function with a local variable
def f3(x):
    z = tag(np.ones(10), 'local_var')
    return (x + z) ** 2

# -- create a general symbolic tracer and apply it to the three functions
x = np.random.random(10)
tracer = Symbolic()

out1 = tracer.trace(f1, x)
out2 = tracer.trace(f2, out1)
out3 = tracer.trace(f3, out2)

# -- compile a function representing f(x, y, z) = out3
new_fn = tracer.compile_function(inputs=[x, y, 'local_var'],
                                 outputs=out3)

# -- compile the gradient of f(x) = out3, with respect to y
fn_grad = tracer.compile_gradient(inputs=x,
                                  outputs=out3,
                                  wrt=y,
                                  reduction=theano.tensor.sum)

assert np.allclose(new_fn(x, y, np.ones(10)), f3(f2(f1(x))))
